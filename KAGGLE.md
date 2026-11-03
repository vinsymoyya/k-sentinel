# Public Kaggle distribution

The distribution contains three separate reusable artifacts:

- Dataset: `vinsymoyya/k-sentinel-synthetic-mri-3840` - all 3,840 synthetic subjects in nine split-preserving NPZ shards; CC0-1.0.
- Model: `vinsymoyya/k-sentinel/pyTorch/residual-cnn/4` - the three open v0.1.0 CNN checkpoints with v0.2.0 source and loading tools; MIT.
- Notebook: `vinsymoyya/k-sentinel-45-checks-and-open-training` - executes all 45 checks before training three fixed-epoch CNNs on the attached dataset.

The dataset and model are public. The model version 4 download was verified byte for byte against the staged source bundle and all three original checkpoints. See `release_status.json` for the notebook execution state and retained reports. This document does not promise downloads, forks, votes or higher clinical performance.

## Download and train

Use the official Kaggle CLI or the dataset/model page's download control:

```bash
kaggle datasets download -d vinsymoyya/k-sentinel-synthetic-mri-3840 --unzip -p data/kaggle
python -m pip install -e ".[learned]"
python scripts/verify_kaggle_data.py data/kaggle
python -m ksentinel.training --data data/kaggle --output results/training --seed 11 --epochs 8
python scripts/verify_checkpoints.py --directory results/training
```

The dataset files preserve complex128 precision. Each shard stores `truth` (N,64,64), `initial_columns` (N,16), `initial_values` (N,64,16), `audit_order` (N,8), `audit_values` (N,64,8), `identities`, scalar `seed`, `split`, and `sigma`. All arrays are procedurally generated, with no patient data. The manifest binds every file to its SHA256 and identity allocation. Regeneration uses a 1e-12 numerical tolerance across platforms while file hashes and discrete membership remain exact.

`observed_kspace` builds inputs solely from initially acquired values. It does not read truth or audits. Training uses only training truth as targets; calibration and test data are first read by inference after the fixed training schedule. The notebook runs on CPU with internet disabled, writes safe checkpoints plus per-subject test errors, and does not select a checkpoint using test results. Editing QUICK=True creates a smoke run and must not be presented as the full experiment.

## Prepare and publish as the owner

```bash
python scripts/prepare_kaggle.py --dataset --bundle
kaggle datasets create -p dist/kaggle/data --public --keep-tabular
kaggle models create -p dist/kaggle/model
kaggle models instances create -p dist/kaggle/variation
kaggle kernels push -p dist/kaggle/notebook --timeout 3600
```

Preparing files does not publish them. Creating a notebook version queues execution; wait for its completed status and inspect output before calling it successful. Never put API tokens or passwords in this repository or notebook. Authentication remains in the normal Kaggle CLI/browser flow outside the checkout.

The notebook's model attachment includes the original checkpoints and source archive, so no download from GitHub is needed during execution. Code licensing and generated-data licensing remain separate. Dataset reproduction and negative audit results are useful contributions; download totals measure observed adoption rather than scientific validity.

## Portable source bundle

Kaggle model version 4 carries `k-sentinel-0.2.0-source.bundle`, a standard ZIP payload with a neutral suffix. This preserves the original compressed CSV results: Kaggle recursively expanded the earlier `.zip`/`.gz` distribution and check KS043 correctly failed. The notebook opens the payload with Python `zipfile`; no network access is needed. For standalone use after downloading the model:

```bash
python -m zipfile -e k-sentinel-0.2.0-source.bundle .
cd k-sentinel-0.2.0
python -m pip install -e ".[learned]"
python scripts/run_validation.py --output validation/local
```

The successful round-trip integrity record is `validation/kaggle-model-0.2.0.json`. Earlier notebook attempts and the Windows timeout are retained under `validation/attempts/`; they do not count as passing runs.
