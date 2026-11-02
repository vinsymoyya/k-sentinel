"""Run the complete frozen synthetic experiment and save subject-level evidence."""
import argparse
import csv
import gzip
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
from .core import (phantom, acquisition, noisy_kspace, ifft, fft, project,
                   null_template, line_error, score, threshold, nrmse, wilson, paired_auc)

BUDGETS = (1, 2, 4, 8)
CASES = ("clean", "insertion", "removal", "single_line", "audit_aware", "noise_shift", "anatomy_shift")


def candidates(truth, measured, mask, model):
    zf = ifft(measured*mask[None, :])
    result = {"oracle": project(truth, measured, mask), "zero_fill": zf}
    if model is not None:
        from .learned import predict
        result["cnn"] = predict(model, zf, measured, mask)
    return result


def summarize(rows, training, config):
    groups = {}
    for row in rows:
        key = (row["model"], row["case"], row["budget"])
        groups.setdefault(key, []).append(row)
    output = []
    rng = np.random.default_rng(1981)
    for (model, case, budget), group in sorted(groups.items()):
        flagged = sum(r["flagged"] for r in group)
        delta = np.array([r["selective_nrmse"]-r["fuse_all_nrmse"] for r in group])
        # Descriptive paired bootstrap: calibration/model are held fixed.
        boot = np.array([delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(500)])
        clean = groups[(model, "clean", budget)]
        entry = {
            "model": model, "case": case, "budget": budget, "n": len(group),
            "flagged": flagged, "rejection_rate": flagged/len(group),
            "wilson95": wilson(flagged, len(group)),
            "per_seed": {str(seed): np.mean([r["flagged"] for r in group if r["seed"] == seed])
                         for seed in config["seeds"]},
            "mean_native_nrmse": float(np.mean([r["native_nrmse"] for r in group])),
            "mean_candidate_nrmse": float(np.mean([r["candidate_nrmse"] for r in group])),
            "mean_fuse_all_nrmse": float(np.mean([r["fuse_all_nrmse"] for r in group])),
            "mean_selective_nrmse": float(np.mean([r["selective_nrmse"] for r in group])),
            "selective_minus_fuse_all": float(delta.mean()),
            "paired_bootstrap95": np.quantile(boot, [0.025, 0.975]).tolist(),
            "mean_observed_residual": float(np.mean([r["observed_residual"] for r in group])),
            "max_observed_residual": float(max(r["observed_residual"] for r in group)),
            "score_auc_vs_clean": paired_auc([r["audit_score"] for r in group], [r["audit_score"] for r in clean]),
            "sampled_fraction": (16+budget)/config["size"],
        }
        output.append(entry)
    return {"config": config, "training": training, "groups": output,
            "interval_scope": "Descriptive phantom-level intervals conditional on shared fitted model/calibration; not population or clinical intervals"}


def run(args):
    start = time.perf_counter()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    config = {"seeds": args.seeds, "train": args.train, "calibration": args.calibration,
              "test_per_seed": args.test, "epochs": args.epochs, "size": args.size,
              "sigma": 0.02, "relative_injected_error": 0.10, "alpha": 0.05,
              "budgets": BUDGETS, "python": platform.python_version(), "numpy": np.__version__,
              "device": "CPU", "learned": not args.no_cnn}
    root = Path(__file__).resolve().parent.parent
    source_files = sorted((root/"ksentinel").glob("*.py")) + [root/"RESEARCH_PROTOCOL.md"]
    config["source_sha256"] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    config["protocol_status"] = "Locally recorded before first run; not registered externally"
    rows, training, manifest = [], {}, []
    examples = {}
    for seed in args.seeds:
        model = None
        if not args.no_cnn:
            import torch
            from .learned import train
            model, history = train(seed, args.train, args.epochs, args.size)
            config["torch"] = str(torch.__version__)
            training[str(seed)] = history
            checkpoint = out/f"cnn_seed{seed}.pt"
            torch.save({"state_dict": model.state_dict(), "seed": seed, "history": history,
                        "config": config}, checkpoint)
        scores = {}
        for i in range(args.calibration):
            identity = seed*100000+10000+i
            manifest.append({"identity": identity, "seed": seed, "split": "calibration"})
            rng = np.random.default_rng(identity+70000000)
            truth = phantom(identity, args.size)
            mask, audit = acquisition(rng, args.size)
            measured = noisy_kspace(truth, rng)
            for name, candidate in candidates(truth, measured, mask, model).items():
                for budget in BUDGETS:
                    scores.setdefault((name, budget), []).append(score(candidate, measured, audit[:budget]))
        thresholds = {key: threshold(values) for key, values in scores.items()}
        (out/f"calibration_seed{seed}.json").write_text(json.dumps({
            f"{name}:{budget}": {"threshold": thresholds[(name, budget)], "scores": values}
            for (name, budget), values in scores.items()}, indent=2))
        for i in range(args.train):
            manifest.append({"identity": seed*100000+i, "seed": seed, "split": "train"})
        for i in range(args.test):
            identity = seed*100000+20000+i
            manifest.append({"identity": identity, "seed": seed, "split": "test"})
            rng = np.random.default_rng(identity+70000000)
            truth = phantom(identity, args.size)
            mask, audit = acquisition(rng, args.size)
            measured = noisy_kspace(truth, rng)
            base = candidates(truth, measured, mask, model)
            template = null_template(truth, mask, rng)
            concentrated = line_error(truth, mask, rng)
            forbidden = mask.copy()
            forbidden[audit] = True
            evasive = line_error(truth, forbidden, rng)
            shifted = phantom(identity, args.size, shifted=True)
            shifted_measured = noisy_kspace(shifted, rng)
            noisy = noisy_kspace(truth, rng, sigma=0.04)
            shifted_base = candidates(shifted, shifted_measured, mask, model)
            noisy_base = candidates(truth, noisy, mask, model)
            for name, native in base.items():
                for case in CASES:
                    candidate = {"clean": native, "insertion": native+template,
                                 "removal": native-template, "single_line": native+concentrated,
                                 "audit_aware": native+evasive,
                                 "noise_shift": noisy_base[name], "anatomy_shift": shifted_base[name]}[case]
                    case_truth = shifted if case == "anatomy_shift" else truth
                    case_measured = shifted_measured if case == "anatomy_shift" else noisy if case == "noise_shift" else measured
                    native_error = nrmse(shifted_base[name] if case == "anatomy_shift" else noisy_base[name] if case == "noise_shift" else native, case_truth)
                    candidate_error = nrmse(candidate, case_truth)
                    observed = np.linalg.norm((fft(candidate)-case_measured)[:, mask])/max(np.linalg.norm(case_measured[:, mask]), 1e-12)
                    for budget in BUDGETS:
                        columns = audit[:budget]
                        value = score(candidate, case_measured, columns)
                        flagged = int(value > thresholds[(name, budget)])
                        fused_mask = mask.copy()
                        fused_mask[columns] = True
                        fused = project(candidate, case_measured, fused_mask)
                        fused_error = nrmse(fused, case_truth)
                        rows.append({"seed": seed, "identity": identity, "model": name,
                                     "case": case, "budget": budget, "audit_score": value,
                                     "threshold": thresholds[(name, budget)], "flagged": flagged,
                                     "native_nrmse": native_error, "candidate_nrmse": candidate_error,
                                     "fuse_all_nrmse": fused_error,
                                     "selective_nrmse": fused_error if flagged else candidate_error,
                                     "observed_residual": float(observed),
                                     "initial_columns": ";".join(map(str, np.flatnonzero(mask))),
                                     "audit_columns": ";".join(map(str, columns))})
                    if seed == args.seeds[0] and i == 0 and name in ("oracle", "cnn") and case in ("clean", "insertion", "audit_aware"):
                        examples[f"{name}_{case}"] = candidate
            if seed == args.seeds[0] and i == 0:
                examples.update(truth=truth, template=template, measured=measured, initial_mask=mask, audit_columns=audit)
            if (i+1) % 128 == 0:
                print(f"seed={seed} tested={i+1}/{args.test}", flush=True)
    manifest = sorted(manifest, key=lambda x: x["identity"])
    (out/"splits.json").write_text(json.dumps(manifest, indent=2))
    with gzip.open(out/"subjects.csv.gz", "wt", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(out/"example.npz", **examples)
    summary = summarize(rows, training, config)
    summary["elapsed_seconds"] = time.perf_counter()-start
    (out/"summary.json").write_text(json.dumps(summary, indent=2))
    (out/"config.json").write_text(json.dumps(config, indent=2))
    print(json.dumps({"rows": len(rows), "seconds": summary["elapsed_seconds"], "output": str(out)}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/main")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 29, 47])
    parser.add_argument("--train", type=int, default=512)
    parser.add_argument("--calibration", type=int, default=256)
    parser.add_argument("--test", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--no-cnn", action="store_true")
    args = parser.parse_args()
    if args.size != 64:
        parser.error("This protocol fixes size=64; create a separate protocol for other sizes")
    if min(args.train, args.calibration, args.test, args.epochs) < 1:
        parser.error("All sample counts and epochs must be positive")
    if max(args.train, args.calibration, args.test) > 10000:
        parser.error("Split identity ranges permit at most 10000 subjects per split")
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds):
        parser.error("Seeds must be unique nonnegative integers")
    run(args)


if __name__ == "__main__":
    main()
