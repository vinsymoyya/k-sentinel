"""Audit an external complex reconstruction without access to reference anatomy.

NPZ contract: candidate (H,W) complex; columns (B,) integer;
audit_values (H,B) complex; calibration_scores (N,) float; sigma scalar.
"""
import argparse
import json
import numpy as np
from .core import fft, threshold


def audit_reconstruction(candidate, columns, audit_values, calibration_scores, sigma, alpha=0.05):
    candidate = np.asarray(candidate)
    columns = np.asarray(columns)
    audit_values = np.asarray(audit_values)
    calibration_scores = np.asarray(calibration_scores)
    if candidate.ndim != 2 or not np.iscomplexobj(candidate) or not np.isfinite(candidate).all():
        raise ValueError("Candidate must be a finite 2D complex image")
    if columns.ndim != 1 or columns.dtype.kind not in "iu" or not len(columns):
        raise ValueError("Columns must be a nonempty integer vector")
    if len(np.unique(columns)) != len(columns) or columns.min() < 0 or columns.max() >= candidate.shape[1]:
        raise ValueError("Audit columns must be unique and within the candidate width")
    if audit_values.shape != (candidate.shape[0], len(columns)) or not np.isfinite(audit_values).all():
        raise ValueError("Audit values must have shape (height, number of audit columns)")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Sigma must be finite and positive")
    limit = threshold(calibration_scores, alpha)
    residual = fft(candidate)[:,columns]-audit_values
    value = float(np.mean(np.abs(residual)**2)/sigma**2)
    p = float((1+np.count_nonzero(calibration_scores >= value))/(len(calibration_scores)+1))
    return {"score": value, "threshold": limit if np.isfinite(limit) else None,
            "p_value": p, "flagged": bool(value > limit), "alpha": alpha,
            "meaning": "Reconstruction disagreement with held-out measurements relative to supplied clean calibration; not an anatomical correctness certificate"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="NPZ using the documented contract; object arrays are forbidden")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as item:
        result = audit_reconstruction(item["candidate"],item["columns"],item["audit_values"],
                                      item["calibration_scores"],float(item["sigma"]),args.alpha)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__ == "__main__": main()
