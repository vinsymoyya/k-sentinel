"""Verify default-safe checkpoint loading and finite model prediction."""
import argparse
from pathlib import Path
import torch
from ksentinel.learned import ResidualCNN

root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--directory",default="results/main")
args=parser.parse_args()
paths=sorted((root/args.directory).glob("cnn_seed*.pt"))
if not paths:
    raise SystemExit("No CNN checkpoints found in the requested directory")
torch.set_num_threads(2)
for path in paths:
    checkpoint=torch.load(path,weights_only=True,map_location="cpu")
    assert type(checkpoint["config"]["torch"]) is str
    model=ResidualCNN()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.inference_mode(): prediction=model(torch.zeros(1,2,64,64))
    assert prediction.shape==(1,2,64,64) and torch.isfinite(prediction).all()
    print(path.name,"safe load and finite prediction passed")
