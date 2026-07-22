"""Search GSAT causal graph size ratio on DrugOOD IC50."""
import argparse,itertools,json,math,statistics,subprocess,sys
from pathlib import Path
RATIOS=(.3,.5,.7)
def stat(v):
    v=[float(x) for x in v if math.isfinite(float(x))];return {"mean":statistics.fmean(v),"std":statistics.pstdev(v)} if v else {"mean":None,"std":None}
def main():
    p=argparse.ArgumentParser();p.add_argument("--domains",nargs="+",default=["assay"],choices=("assay","scaffold","size"));p.add_argument("--seeds",nargs="+",type=int,default=[1,2,3,4]);p.add_argument("--subset",default="core",choices=("core","general","refined"));p.add_argument("--endpoint",choices=("ic50","ec50"),default="ic50");p.add_argument("--data-root",type=Path);p.add_argument("--output-root",type=Path,default=Path(__file__).parent/"sweeps");p.add_argument("--device",default="auto");p.add_argument("--dry-run",action="store_true");p.add_argument("--causal-graph-size-ratios",nargs="+",type=float,default=RATIOS);a,extra=p.parse_known_args();script=Path(__file__).parent/"train_ic50.py";jobs=[]
    for d,s,r in itertools.product(a.domains,a.seeds,a.causal_graph_size_ratios):
        out=a.output_root/a.endpoint/d/f"ratio_{str(r).replace('.','p')}"/f"seed_{s}";cmd=[sys.executable,str(script),"--domain",d,"--subset",a.subset,"--endpoint",a.endpoint,"--seed",str(s),"--causal-graph-size-ratio",str(r),"--device",a.device,"--output-dir",str(out)];cmd += (["--data-root",str(a.data_root)] if a.data_root else [])+extra;jobs.append((cmd,out,d,r))
    for j in jobs: print(" ".join(j[0]))
    if a.dry_run:return
    for cmd,*_ in jobs: subprocess.run(cmd,check=True)
    groups={};best={};agg={"method":"GSAT","endpoint":a.endpoint,"groups":{},"seeds":a.seeds}
    for _,out,d,r in jobs: groups.setdefault((d,r),[]).append(json.loads((out/"summary.json").read_text()))
    for (d,r),ss in sorted(groups.items()):
        key=f"{d}/causal_graph_size_ratio={r:g}";entry={"ood_val_selection":stat(x["best_ood_val"] for x in ss),"ood_test_accuracy":stat(x["metrics"]["ood_test"]["accuracy"] for x in ss),"ood_test_roc_auc":stat(x["metrics"]["ood_test"]["roc_auc"] for x in ss)};agg["groups"][key]=entry;m=entry["ood_val_selection"]["mean"]
        if m is not None and (d not in best or m>best[d][0]):best[d]=(m,key)
    agg["best_by_domain"]={d:v[1] for d,v in best.items()};root=a.output_root/a.endpoint;root.mkdir(parents=True,exist_ok=True);(root/"aggregate.json").write_text(json.dumps(agg,indent=2))
if __name__=="__main__":main()
