"""Verify every distributed object against the deterministic acquisition source."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ksentinel.dataset import ARRAYS,load_partition,make_shard


def verify(folder):
    folder=Path(folder)
    manifest=json.loads((folder/"manifest.json").read_text())
    published=json.loads((ROOT/"results/main/splits.json").read_text())
    expected={row["identity"]:(row["seed"],row["split"]) for row in published}
    actual={}
    for entry in manifest["files"]:
        data=load_partition(folder,entry["split"],entry["seed"])
        regenerated=make_shard(entry["seed"],entry["split"],entry["count"])
        for key in ARRAYS:np.testing.assert_array_equal(data[key],regenerated[key])
        actual.update({int(identity):(entry["seed"],entry["split"]) for identity in data["identities"]})
    if actual!=expected:raise ValueError("Distributed objects do not match published subject partitions")
    return {"status":"passed","objects_verified":len(actual),"shards_verified":len(manifest["files"]),
            "all_arrays_regenerate_exactly":True,"published_split_membership_matches":True,
            "manifest_sha256":hashlib.sha256((folder/"manifest.json").read_bytes()).hexdigest(),
            "scope":"Synthetic dataset integrity, not clinical validity"}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder",nargs="?",default="dist/kaggle/data")
    parser.add_argument("--output",default="validation/dataset-0.2.0.json")
    args=parser.parse_args();report=verify(args.folder)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))
