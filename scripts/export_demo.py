"""Export the first 16 released test phantoms with synthetic raw measurements.

These belong to the published test set. Do not use them as additional training
data and then evaluate against the same test results.
"""
from pathlib import Path
import numpy as np
from ksentinel.core import phantom, acquisition, noisy_kspace

root=Path(__file__).resolve().parents[1]
destination=root/"data"; destination.mkdir(exist_ok=True)
identities=np.arange(1120000,1120016,dtype=np.int64)
truths=[]; measurements=[]; masks=[]; audits=[]
for identity in identities:
    rng=np.random.default_rng(int(identity)+70000000)
    truth=phantom(int(identity))
    mask,audit=acquisition(rng)
    measured=noisy_kspace(truth,rng)
    truths.append(truth); measurements.append(measured); masks.append(mask); audits.append(audit)
np.savez_compressed(destination/"synthetic-demo.npz",identities=identities,
                    truth=np.stack(truths),kspace=np.stack(measurements),
                    initial_mask=np.stack(masks),audit_order=np.stack(audits),sigma=np.array(0.02))
print(destination/"synthetic-demo.npz")
