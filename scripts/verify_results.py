"""Recompute released endpoint summaries independently from subject CSV rows."""
import csv
import argparse
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--check-only",action="store_true",help="Verify without rewriting the frozen report")
args=parser.parse_args()
result=root/"results/main"
summary=json.loads((result/"summary.json").read_text())
splits=json.loads((result/"splits.json").read_text())
assert len({r["identity"] for r in splits}) == len(splits)
membership={r["identity"]: r["split"] for r in splits}
assert len(splits) == 3*(512+256+512)
checks={"split_identities_unique": True}
for path, expected in summary["config"]["source_sha256"].items():
    relative=path.replace("\\", "/")
    snapshot=result/"source_snapshot"/relative
    source=snapshot if snapshot.is_file() else root/relative
    assert hashlib.sha256(source.read_bytes()).hexdigest()==expected, path
checks["experiment_source_hashes_match"]=True
groups=defaultdict(list)
with gzip.open(result/"subjects.csv.gz","rt",encoding="utf8",newline="") as stream:
    for r in csv.DictReader(stream):
        assert membership[int(r["identity"])]=="test"
        initial=set(map(int,r["initial_columns"].split(";")))
        audit=set(map(int,r["audit_columns"].split(";")))
        assert len(initial)==16 and len(audit)==int(r["budget"]) and not initial.intersection(audit)
        assert int(r["flagged"])==int(float(r["audit_score"])>float(r["threshold"]))
        for field in ("audit_score","native_nrmse","candidate_nrmse","fuse_all_nrmse","selective_nrmse","observed_residual"):
            assert math.isfinite(float(r[field]))
        expected_policy=float(r["fuse_all_nrmse"] if int(r["flagged"]) else r["candidate_nrmse"])
        assert float(r["selective_nrmse"])==expected_policy
        groups[(r["model"],r["case"],int(r["budget"]))].append(r)
checks["all_row_flags_masks_and_policy_consistent"]=True
for g in summary["groups"]:
    rows=groups[(g["model"],g["case"],g["budget"])]
    assert len(rows)==g["n"]==1536
    assert sum(int(r["flagged"]) for r in rows)==g["flagged"]
    assert math.isclose(sum(int(r["flagged"]) for r in rows)/len(rows),g["rejection_rate"],abs_tol=1e-14)
    for a,b in (("candidate_nrmse","mean_candidate_nrmse"),("fuse_all_nrmse","mean_fuse_all_nrmse"),("selective_nrmse","mean_selective_nrmse")):
        assert math.isclose(sum(float(r[a]) for r in rows)/len(rows),g[b],abs_tol=1e-12)
    # Independent Wilson interval calculation.
    n=len(rows); p=g["flagged"]/n; z=1.959963984540054
    lo=(p+z*z/(2*n)-z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n)
    hi=(p+z*z/(2*n)+z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n)
    assert np.allclose([lo,hi],g["wilson95"],atol=1e-14)
checks["all_84_group_counts_means_and_intervals_recomputed"]=True
for model in ("oracle","cnn","zero_fill"):
    for budget in (1,2,4,8):
        clean=groups[(model,"clean",budget)]
        blind=groups[(model,"audit_aware",budget)]
        assert [r["identity"] for r in clean]==[r["identity"] for r in blind]
        assert [r["flagged"] for r in clean]==[r["flagged"] for r in blind]
checks["audit_aware_error_has_identical_decisions_to_clean"]=True
for seed in (11,29,47):
    cal=json.loads((result/f"calibration_seed{seed}.json").read_text())
    for key,value in cal.items():
        scores=sorted(value["scores"])
        assert len(scores)==256
        assert scores[244]==value["threshold"] # ceil(257 * .95) = 245
        if key.startswith("oracle:"):
            assert 0.97 < sum(scores)/len(scores) < 1.03
checks["calibration_rank_and_oracle_noise_expectation_verified"]=True
report={"status":"passed","rows_checked":sum(map(len,groups.values())),"groups_checked":len(groups),"checks":checks,
        "scope":"Structural and numerical verification of released simulation; not independent clinical validation"}
if not args.check_only:
    (result/"verification.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
