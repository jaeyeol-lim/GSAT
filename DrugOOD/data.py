"""Load cached DrugOOD IC50 PyG splits."""
from __future__ import annotations
import os
from pathlib import Path
import torch
from torch_geometric.data import InMemoryDataset

def discover_data_root():
    root = Path(__file__).resolve().parents[2]
    env = os.environ.get("DRUGOOD_DATA_ROOT")
    candidates = (*((Path(env).expanduser(),) if env else ()), root.parent / "Graph-OOD-Lab/data/DrugOOD",
                  Path("/workspace/Graph-OOD-Lab/data/DrugOOD"), Path("/home/jylim/project/Graph-OOD-Lab/data/DrugOOD"),
                  Path("/home/jylim/Graph-OOD-Lab/data/DrugOOD"), Path("/home/irteam/Graph-OOD-Lab/data/DrugOOD"))
    return next((p for p in candidates if p.is_dir()), candidates[0])

class CachedDrugOOD(InMemoryDataset):
    def __init__(self, path):
        super().__init__(root=None)
        if not path.is_file():
            raise FileNotFoundError(f"Missing DrugOOD IC50 split: {path}")
        self.data, self.slices = torch.load(path, map_location="cpu", weights_only=False)

def load_splits(data_root, subset, domain, endpoint="ic50"):
    stem = f"drugood_lbap_{subset}_{endpoint}_{domain}"
    splits = {s: CachedDrugOOD(data_root / f"{stem}_{s}.pt") for s in ("train", "ood_val", "ood_test")}
    for s in ("iid_val", "iid_test"):
        path = data_root / f"{stem}_{s}.pt"
        if path.is_file(): splits[s] = CachedDrugOOD(path)
    return stem, splits
