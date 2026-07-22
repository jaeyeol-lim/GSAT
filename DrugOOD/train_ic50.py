"""Train GSAT on DrugOOD IC50 with the shared baseline protocol."""
from __future__ import annotations
import argparse, json, math, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
try:
    from .data import discover_data_root, load_splits
    from .model import GSATModel
except ImportError:
    from data import discover_data_root, load_splits
    from model import GSATModel

def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", choices=("assay","scaffold","size"), default="assay"); p.add_argument("--subset", choices=("core","general","refined"), default="core"); p.add_argument("--endpoint",choices=("ic50","ec50"),default="ic50")
    p.add_argument("--data-root", type=Path, default=discover_data_root()); p.add_argument("--output-dir", type=Path); p.add_argument("--device", default="auto"); p.add_argument("--seed",type=int,default=1)
    p.add_argument("--epochs",type=int,default=50); p.add_argument("--erm-pretrain-epochs",type=int,default=10); p.add_argument("--patience",type=int,default=10)
    p.add_argument("--batch-size",type=int,default=128); p.add_argument("--num-workers",type=int,default=4); p.add_argument("--lr",type=float,default=1e-3); p.add_argument("--weight-decay",type=float,default=0.)
    p.add_argument("--hidden-dim",type=int,default=128); p.add_argument("--num-layers",type=int,default=4); p.add_argument("--dropout",type=float,default=.1)
    p.add_argument("--causal-graph-size-ratio",type=float,default=.5,help="GSAT final r; sweep over 0.3, 0.5, 0.7."); p.add_argument("--info-loss-coef",type=float,default=1.)
    p.add_argument("--decay-interval",type=int,default=0,help="0 scales the official r schedule to 50 epochs."); p.add_argument("--selection-metric",choices=("accuracy","roc_auc"),default="accuracy"); p.add_argument("--log-every",type=int,default=1)
    a=p.parse_args()
    if not 0<a.causal_graph_size_ratio<1: p.error("ratio must be in (0,1)")
    return a
def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s); torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
def device_of(s):
    if s=="auto": return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d=torch.device(s)
    if d.type=="cuda" and not torch.cuda.is_available(): raise RuntimeError(f"CUDA unavailable: {s}")
    return d
def loader(ds,a,shuffle): return DataLoader(ds,batch_size=a.batch_size,shuffle=shuffle,num_workers=a.num_workers,pin_memory=torch.cuda.is_available(),persistent_workers=a.num_workers>0)
def auc(y,s):
    if np.unique(y).size<2:return math.nan
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y,s))
def current_r(a,epoch):
    steps=max(1,round((.9-a.causal_graph_size_ratio)/.1)); interval=a.decay_interval or max(1,a.epochs//(steps+1))
    return max(a.causal_graph_size_ratio,.9-(epoch//interval)*.1)
@torch.no_grad()
def evaluate(model,dl,dev,gsat=True):
    model.eval(); ys=[]; scores=[]; preds=[]; loss=0.; n=0
    for b in dl:
        b=b.to(dev); y=b.y.view(-1).long(); logits=model.forward_gsat(b,False)[0] if gsat else model.forward_erm(b)
        loss+=float(F.cross_entropy(logits,y,reduction="sum"));n+=y.numel();ys.append(y.cpu());scores.append(logits.softmax(-1)[:,1].cpu());preds.append(logits.argmax(-1).cpu())
    y=torch.cat(ys).numpy();s=torch.cat(scores).numpy();pr=torch.cat(preds).numpy()
    return {"loss":loss/max(n,1),"accuracy":float((pr==y).mean()),"roc_auc":auc(y,s),"count":n}
def train_epoch(model,dl,dev,opt,a,epoch,gsat):
    model.train(); total=pred_t=info_t=0.;n=0;r=current_r(a,epoch)
    for b in dl:
        b=b.to(dev);y=b.y.view(-1).long();opt.zero_grad(set_to_none=True)
        if gsat:
            logits,att,_=model.forward_gsat(b,True); pred=F.cross_entropy(logits,y); info=(att*torch.log(att/r+1e-6)+(1-att)*torch.log((1-att)/(1-r+1e-6)+1e-6)).mean(); loss=pred+a.info_loss_coef*info
        else: logits=model.forward_erm(b);pred=F.cross_entropy(logits,y);info=pred.new_zeros(());loss=pred
        loss.backward();opt.step();c=y.numel();total+=float(loss.detach())*c;pred_t+=float(pred.detach())*c;info_t+=float(info.detach())*c;n+=c
    return {"loss":total/n,"pred":pred_t/n,"info":info_t/n,"r":r}
def train(a):
    seed_all(a.seed);dev=device_of(a.device);stem,splits=load_splits(a.data_root,a.subset,a.domain,a.endpoint); tr=loader(splits["train"],a,True); ev={k:loader(v,a,False) for k,v in splits.items() if k!="train"}
    sample=splits["train"][0]; model=GSATModel(sample.x.shape[-1],sample.edge_attr.shape[-1],a.hidden_dim,a.num_layers,a.dropout).to(dev)
    out=a.output_dir or Path(__file__).resolve().parent/"outputs"/f"gsat_{stem}_seed{a.seed}_{int(time.time())}";out.mkdir(parents=True,exist_ok=True);best=out/"best.pt";history=[]
    opt=torch.optim.Adam(list(model.node_encoder.parameters())+list(model.convs.parameters())+list(model.norms.parameters())+list(model.classifier.parameters()),lr=a.lr,weight_decay=a.weight_decay)
    for e in range(1,a.erm_pretrain_epochs+1):
        tm=train_epoch(model,tr,dev,opt,a,e-1,False);v=evaluate(model,ev["ood_val"],dev,False);history.append({"phase":"erm_pretrain","epoch":e,"train":tm,"ood_val":v})
    opt=torch.optim.Adam(model.parameters(),lr=a.lr,weight_decay=a.weight_decay);bv=-math.inf;be=stale=0
    for e in range(1,a.epochs+1):
        tm=train_epoch(model,tr,dev,opt,a,e-1,True);v=evaluate(model,ev["ood_val"],dev);value=v[a.selection_metric];history.append({"phase":"main","epoch":e,"train":tm,"ood_val":v})
        if e%a.log_every==0: print(f"epoch={e:03d} loss={tm['loss']:.4f} info={tm['info']:.4f} r={tm['r']:.2f} val_acc={v['accuracy']:.4f} val_auc={v['roc_auc']:.4f}")
        if value>bv: bv,be,stale=value,e,0;torch.save({"model":model.state_dict(),"args":vars(a),"epoch":e},best)
        else:
            stale+=1
            if a.patience>0 and stale>=a.patience: break
    model.load_state_dict(torch.load(best,map_location=dev,weights_only=False)["model"]);metrics={k:evaluate(model,d,dev) for k,d in ev.items()};summary={"method":"GSAT","dataset":stem,"seed":a.seed,"best_epoch":be,"best_ood_val":bv,"selection_metric":a.selection_metric,"metrics":metrics,"args":{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}}
    (out/"history.json").write_text(json.dumps(history,indent=2));(out/"summary.json").write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=="__main__": train(parse_args())
