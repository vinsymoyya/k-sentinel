"""Split-preserving synthetic raw-measurement shards for open distribution."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .core import phantom, acquisition, noisy_kspace

SPLITS = {"train": 0, "calibration": 10000, "test": 20000}
ARRAYS = ("identities", "truth", "initial_columns", "initial_values", "audit_order", "audit_values")


def make_shard(seed, split, count):
    if split not in SPLITS or not isinstance(seed, int) or seed < 0 or not 1 <= count <= 10000:
        raise ValueError("Invalid seed, split or count")
    shared_rng = np.random.default_rng(seed*100000+17)
    identities = np.arange(seed*100000+SPLITS[split], seed*100000+SPLITS[split]+count, dtype=np.int64)
    records = {key: [] for key in ARRAYS if key != "identities"}
    for identity in identities:
        rng = shared_rng if split == "train" else np.random.default_rng(int(identity)+70000000)
        truth = phantom(int(identity))
        mask, audit = acquisition(rng)
        measured = noisy_kspace(truth, rng)
        records["truth"].append(truth)
        records["initial_columns"].append(np.flatnonzero(mask))
        records["initial_values"].append(measured[:, mask])
        records["audit_order"].append(audit)
        records["audit_values"].append(measured[:, audit])
    return {"identities": identities, **{key: np.stack(value) for key, value in records.items()},
            "split": np.array(split), "seed": np.array(seed), "sigma": np.array(0.02)}


def validate_shard(data):
    required = set(ARRAYS) | {"split", "seed", "sigma"}
    if not required.issubset(data):
        raise ValueError("Missing required shard arrays")
    ids = np.asarray(data["identities"])
    if ids.ndim != 1 or not len(ids) or ids.dtype.kind not in "iu" or len(np.unique(ids)) != len(ids):
        raise ValueError("Shard identities must be unique integers")
    n = len(ids)
    if np.shape(data["split"]) != () or str(data["split"]) not in SPLITS or np.shape(data["seed"]) != ():
        raise ValueError("Invalid shard split or seed")
    seed = int(data["seed"])
    if seed < 0 or np.asarray(data["seed"]).dtype.kind not in "iu":
        raise ValueError("Seed must be a nonnegative integer")
    lower = seed*100000+SPLITS[str(data["split"])]
    if np.any(ids < lower) or np.any(ids >= lower+10000):
        raise ValueError("Identity does not belong to declared seed/split")
    shapes = {"truth": (n,64,64), "initial_columns": (n,16), "initial_values": (n,64,16),
              "audit_order": (n,8), "audit_values": (n,64,8)}
    for key, shape in shapes.items():
        value = np.asarray(data[key])
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"Invalid shape or nonfinite values in {key}")
        if key in ("truth", "initial_values", "audit_values") and not np.iscomplexobj(value):
            raise ValueError(f"Complex values required in {key}")
    for a, b in zip(data["initial_columns"], data["audit_order"]):
        if a.dtype.kind not in "iu" or b.dtype.kind not in "iu" or a.min()<0 or b.min()<0 or a.max()>=64 or b.max()>=64:
            raise ValueError("Column indices must be integers in [0,64)")
        if len(set(a)) != 16 or len(set(b)) != 8 or set(a).intersection(b):
            raise ValueError("Initial and audit columns must be unique and disjoint")
    sigma = np.asarray(data["sigma"])
    if sigma.shape != () or not np.isrealobj(sigma) or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Invalid shard noise scale")
    return data


def export_dataset(folder, seeds=(11,29,47), counts=(512,256,512)):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if len(counts) != 3 or len(set(seeds)) != len(seeds):
        raise ValueError("Require three split counts and distinct seeds")
    files = []
    for seed in seeds:
        for split, count in zip(SPLITS, counts):
            data = validate_shard(make_shard(seed, split, count))
            path = folder/f"seed{seed}-{split}.npz"
            np.savez_compressed(path, **data)
            files.append({"file": path.name, "seed": seed, "split": split, "count": count,
                          "identities": data["identities"].tolist(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = {"schema_version": 1, "license": "CC0-1.0", "size": 64,
                "operator": "centered orthonormal 2D Fourier; complete phase-encoding columns",
                "precision": "complex128", "files": files,
                "total_objects": sum(entry["count"] for entry in files)}
    (folder/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf8")
    return manifest


def load_partition(folder, split, seed, limit=None):
    folder = Path(folder)
    manifest = json.loads((folder/"manifest.json").read_text(encoding="utf8"))
    if split not in SPLITS or manifest.get("schema_version") != 1:
        raise ValueError("Unsupported split or manifest schema")
    all_ids = []
    selected = []
    for entry in manifest["files"]:
        filename = entry["file"]
        if Path(filename).name != filename or "/" in filename or "\\" in filename or not filename.endswith(".npz"):
            raise ValueError("Shard paths must be local NPZ filenames")
        if len(entry["identities"]) != entry["count"]:
            raise ValueError("Manifest count disagrees with identity list")
        all_ids.extend(entry["identities"])
        if entry["split"] == split and entry["seed"] == seed:
            selected.append(entry)
    if len(set(all_ids)) != len(all_ids) or len(all_ids) != manifest["total_objects"]:
        raise ValueError("Dataset has duplicate identities or inconsistent total")
    if len(selected) != 1:
        raise ValueError("Expected exactly one shard for requested seed and split")
    entry = selected[0]
    path = folder/entry["file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        raise ValueError("Dataset shard SHA256 mismatch")
    with np.load(path, allow_pickle=False) as loaded:
        data = validate_shard({name: loaded[name] for name in loaded.files})
    if data["identities"].tolist() != entry["identities"] or str(data["split"]) != split or int(data["seed"]) != seed:
        raise ValueError("Shard contents disagree with manifest membership")
    if limit is not None:
        if not isinstance(limit,int) or limit <= 0:
            raise ValueError("Limit must be a positive integer")
        data = {key: value[:limit] if key in ARRAYS else value for key,value in data.items()}
    return data


def observed_kspace(data, index):
    """Construct model input without touching truth or audit values."""
    measured = np.zeros((64,64), dtype=complex)
    mask = np.zeros(64, dtype=bool)
    columns = data["initial_columns"][index]
    mask[columns] = True
    measured[:, columns] = data["initial_values"][index]
    return measured, mask
