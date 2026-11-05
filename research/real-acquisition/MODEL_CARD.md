# M4Raw fine-tuned pilot CNN

Version 0.3.0. Author: Vinsy Moyya, ORCID https://orcid.org/0009-0003-0042-3556 . Research only; clinical validation not performed.

Checkpoint: `m4raw_finetuned_seed20261105.pt`, SHA256 `7413aafb80012c9e82e2166bb34bbb355e092bb6619435cbbb7a01f9abd2140d`.

Architecture: residual 2-16-16-16-2 CNN, applied separately to complex coil images represented by real/imaginary channels, followed by initial-measurement projection. Initialization: original K-Sentinel phantom seed-11 checkpoint, SHA256 `4d6995305dbadc7a3a4516203783cf76e4c16dac742729e8c6d1bc5624465d2d`.

## Training and data separation

Eight fixed fine-tuning epochs on T2 repeat 1 from twenty M4Raw training volunteers, slices 7/8/9, four coils, 240 coil-slices. Adam learning rate 0.0001, batch size 8, seed 20261105, two CPU threads. No calibration or test subject trains the model; no early stopping or test-based checkpoint selection. Thirty other participants calibrate thresholds; expanded-test and motion-risk groups contain twenty-five other participants each. Full selection and source hashes: `manifest.json`.

## Input and loading

Native centered complex `[coil,FE,PE]`, evaluated at four coils and 256 x 256. Phase encoding is last. Boolean initial and ACS masks select columns; ACS is a subset of initial measurements. Normalization uses initial ACS only. Source support is 195 columns for T2 and 198 for FLAIR, not all 256 stored columns. Unavailable zero padding is not measured data.

```python
from ksentinel.model_io import load_model
from ksentinel.real_acquisition import reconstruct_coils, rss

model = load_model('m4raw_finetuned_seed20261105.pt', expected_sha256='7413aafb80012c9e82e2166bb34bbb355e092bb6619435cbbb7a01f9abd2140d')
coils = reconstruct_coils(model, measured_kspace, initial_mask, acs_mask)
display_image = rss(coils)
```

Unacquired values are discarded. Audits and source RSS do not determine inference or normalization. Use this native centered adapter rather than passing M4Raw k-space to the older synthetic `model_io.reconstruct` convention. The small per-coil model does not estimate sensitivity maps or exploit joint multicoil encoding.

## Evaluated outcomes and limits

Primary: twenty-five held-out T2 repetition-1 participants, random retrospective R=4, eight audit columns. Mean RSS source-reference NRMSE: fine-tuned 0.31023, frozen phantom CNN 0.30943, zero fill 0.31171. Fine-tuning did not outperform the frozen model. One primary case exceeded the calibration-defined high-error cutoff and was flagged; one event cannot establish detection sensitivity or stable AUROC generalization. In secondary FLAIR/random/R4, both high-reference-error cases remained unflagged. All other conditions and natural-error cases remain in the complete record.

Source references are noisy all-available reconstructions and may contain motion. Reference NRMSE is not a clinical endpoint or proof of unsupported focal anatomy. Data are healthy-volunteer acquisitions from one 0.3-T scanner/four-channel coil. No cross-vendor, pathology, prospective sequence, reader-study or regulatory evaluation was completed. The model is not a state-of-the-art reconstruction or motion-correction system.

## License and intended use

Code/weights MIT, with M4Raw training provenance and attribution retained. M4Raw data, selection metadata and source-based figures: CC BY 4.0, see `SOURCE_NOTICE.md`. The older CC0 synthetic license does not apply. Lyu et al., Scientific Data 10:264 (2023), https://doi.org/10.1038/s41597-023-02181-4 ; https://doi.org/10.5281/zenodo.8056074 ; https://huggingface.co/datasets/mylyu/M4Raw_brain .

Intended use: open reconstruction-audit research/reproduction. Excluded uses: diagnosis, treatment, patient triage, scanner control and clinical quality assurance. See `REPORT.md` for every outcome and conditional uncertainty.
