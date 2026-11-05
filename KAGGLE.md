# Public Kaggle distributions

## Preserved synthetic workflow (v0.2.0)

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

## Completed hosted run

[Notebook version 3](https://www.kaggle.com/code/vinsymoyya/k-sentinel-45-checks-and-open-training) completed on CPU with internet disabled: **45 passed, zero failed, zero skipped**, then three complete eight-epoch training runs with QUICK=False. Runtime: Python 3.13.18, NumPy 2.1.3, PyTorch 2.11.0+cpu. Each seed used 512 training, 256 calibration and 512 held-out test subjects.

| Seed | Mean CNN complex NRMSE | Mean zero-fill complex NRMSE |
|---|---:|---:|
| 11 | 0.212878 | 0.231545 |
| 29 | 0.191866 | 0.232343 |
| 47 | 0.189403 | 0.232962 |

The downloaded reports were independently checked for eight finite epoch losses, declared split counts, 3,840 distinct identities across all seeds, manifest SHA256, checkpoint SHA256, exact test identity membership and recomputed mean errors. Core module and test hashes match the current checkout. Reports/checkpoints are retained in `results/kaggle-0.2.0/`; validation JSON/JUnit and the non-quick run summary are in `validation/kaggle-run-0.2.0/`. The compact integrity record is `validation/kaggle-run-0.2.0.json`.

These retrained checkpoints are separate from the immutable original `results/main/` weights. This is synthetic reconstruction evidence, not clinical efficacy or an improvement to the original paper's low added-error audit sensitivity.

## Completed real-acquisition workflow (v0.3.0)

[Real-MRI notebook](https://www.kaggle.com/code/vinsymoyya/k-sentinel-real-mri-variations-and-failures), version 1, completed the original 45 plus 12 new contracts with zero skips, downloaded/verified all 180 source H5 files, fine-tuned eight fixed epochs, and completed all 2,880 result rows. Its attached original phantom weights remain unchanged in model source variation version 6. The separately fine-tuned pilot is documented in `research/real-acquisition/MODEL_CARD.md`.

This notebook needs internet for the frozen 2.22 GB CC BY 4.0 M4Raw source selection. It does not attach or relabel M4Raw as the older synthetic CC0 dataset. Actual repeated acquisitions and T2/FLAIR contrast are distinguished from retrospective R/mask changes. Source references contain noise/artifacts. The primary fine-tuned model did not outperform the frozen model. All 100 participant identities remain disjoint across train/calibration/test/motion-risk roles.

```bash
python -m pip install -e ".[learned,real]"
python scripts/run_validation.py --output validation/local
python scripts/run_real_validation.py --output validation/local/real
python scripts/download_m4raw.py
python scripts/run_real_acquisition.py --output results/my-real-reproduction
python scripts/verify_real_results.py --results results/my-real-reproduction --output validation/my-real-verification.json
```

Use a new result directory; the runner refuses to overwrite an existing experiment. The immutable protocol/manifest and source notices live in `research/real-acquisition/`. The independently reconciled original run is `results/real-acquisition-0.3.0/`. Add `.[figures,paper]` for plotting/PDF generation and retain source-based figure attribution.

The separate [M4Raw pilot weight variation](https://www.kaggle.com/models/vinsymoyya/k-sentinel/pyTorch/m4raw-pilot/1) contains the verified fine-tuned checkpoint, model card, source bundle and addendum. Original phantom weights remain in their original variation.
