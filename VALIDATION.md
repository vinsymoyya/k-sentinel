# The 45 validation contracts

Version 0.2.0 validates the open residual CNN and its synthetic research workflow. It does not assign a medical reliability score. Passing software, numerical and synthetic-data checks cannot establish clinical safety, natural pathology sensitivity, patient benefit or regulatory authorization.

Run the full suite with the learned dependencies installed:

```bash
python -m pip install -e ".[learned]"
python scripts/run_validation.py --output validation/local
```

The runner requires exactly **KS001-KS045**, fails on missing or duplicate cases, and counts neither skips nor setup failures as passes. It emits per-case JSON evidence, runtime and dependency versions, source hashes, and JUnit XML. The same suite runs on hosted Linux with Python 3.11 and 3.12; 45 distinct cases executed twice are still 45 distinct contracts. Local evidence is in `validation/release-0.2.0.json`. Hosted Linux execution passed all 45 cases with zero skips under Python 3.11 and 3.12 in [run 41551084455](https://github.com/vinsymoyya/k-sentinel/actions/runs/41551084455), source `d0a2b7607de086d7986a6442f968a40fdf91bdff`. Downloaded JSON/JUnit evidence is retained in `validation/hosted-python-3.11/` and `validation/hosted-python-3.12/`. A prepared Kaggle notebook is not a completed remote run.

| ID | Contract | Evidence type |
|---|---|---|
| KS001 | Fourier inverse and energy preservation | Analytic identity |
| KS002 | Added projected structure remains invisible to initial measurements | Numerical negative control |
| KS003 | A perturbation avoiding the entire audit remains invisible | Threat-model failure control |
| KS004 | Initial/audit masks have correct counts and disjoint support | Repeated acquisition invariant |
| KS005 | Finite-sample calibration rank, ties and insufficient calibration | Known-order examples |
| KS006 | Wilson interval boundaries and tied AUROC | Independent known values |
| KS007 | Noiseless extra-measurement fusion reduces squared error | Orthogonal-projection property |
| KS008 | Constant-image Fourier audit gives analytic score and p-value | Analytic public-interface example |
| KS009 | An unflagged audit is not labelled anatomically safe | Public output contract |
| KS010 | Invalid audit masks and values are rejected | Input boundary cases |
| KS011 | Negative calibration scores are rejected | Regression against frozen baseline |
| KS012 | Complex calibration scores are not silently cast to real | Regression against frozen baseline |
| KS013 | Empty spatial dimensions fail explicit input validation | Boundary behavior |
| KS014 | Numerical overflow fails with an error instead of an infinite score | Regression against frozen baseline |
| KS015 | Invalid alpha values are rejected | Statistical parameter boundary |
| KS016 | Invalid, nonfinite or nonscalar noise scales are rejected | Normalization boundary |
| KS017 | Tied rank p-values remain conservative | Known rank example |
| KS018 | Insufficient calibration gives strict JSON and no false certification | Public API/serialization boundary |
| KS019 | Equal-magnitude, opposite-phase disagreement is detected | Independent complex-valued example |
| KS020 | Changing image/data/noise units together preserves the score | Scaling invariant |
| KS021 | Valid dataset shards retain declared subject/split identities | Actual file round trip |
| KS022 | Overlapping initial/audit columns are rejected | Leakage negative control |
| KS023 | Duplicate subjects across manifest splits are rejected | Leakage negative control |
| KS024 | Declared split agrees with identity allocation | Data contract |
| KS025 | Modified dataset bytes fail SHA256 verification | Corruption negative control |
| KS026 | Manifest counts agree with subject membership | Truncation/integrity control |
| KS027 | Model input construction ignores truth and audit arrays | Poisoned-unused-input control |
| KS028 | Exported raw data match the frozen published test example | Preservation against released artifact |
| KS029 | All three open checkpoints load with the default-safe loader | Real artifact loading |
| KS030 | Incorrect checkpoint digest fails before loading | Integrity boundary |
| KS031 | Nonfinite weights are rejected | Real corrupted checkpoint |
| KS032 | Incompatible architectures are rejected | Strict checkpoint contract |
| KS033 | Model reconstruction preserves acquired coefficients | Inference integration |
| KS034 | Unacquired values cannot influence model predictions | Leakage negative control |
| KS035 | Full acquisition preserves complex phase | Physical/operator boundary |
| KS036 | Checkpoint reload preserves prediction | Real artifact round trip |
| KS037 | Actual optimization updates weights with finite loss | Small real training run |
| KS038 | Repeated fixed-seed training agrees | Deterministic training reproduction |
| KS039 | External audit CLI emits strict JSON with known output | Real subprocess |
| KS040 | CLI refuses object-array unpickling | Real subprocess/input boundary |
| KS041 | Small benchmark produces portable JSON and complete records | End-to-end benchmark |
| KS042 | Invalid experiment selectors fail before creating output | Real CLI failure paths |
| KS043 | All 129,024 frozen result rows and 84 groups reaggregate | Independent saved-evidence check |
| KS044 | Public notebook is parseable, offline and attached to intended inputs | Notebook/configuration inspection |
| KS045 | Downloaded shards train, calibrate, evaluate and reload a model | End-to-end data-driven training |

Three accepted-invalid-input defects are demonstrated against the preserved v0.1.0 implementation in `validation/legacy-regression-evidence.json`. That comparison does not falsify the v0.1.0 results, which used valid inputs. Historical source files and results remain under `results/main/` and the immutable v0.1.0 tag. The new source snapshot prevents later library updates from silently changing the provenance of the original study.

Cross-platform regeneration compares continuous complex values at `rtol=atol=1e-12`, allowing last-bit libm/FFT roundoff (the first Linux run differed from Windows by about 1e-16). File SHA256 digests, subject identities, masks and split membership remain exact checks. Within-host fixed-seed training repeatability is checked exactly. Tolerances do not permit relabelled subjects, changed masks or substantive image changes.

Dataset publication has an additional distribution check, outside this 45-case suite: every exported complex array and acquisition is regenerated and all 3,840 identities are compared with the published split record. See `validation/dataset-0.2.0.json`. The check count is deliberately separate from population sizes, repeated matrix runs and dataset rows.

Not covered: real multicoil scanner physics, motion, external hospital data, demographic performance, biological lesion realism, clinician review, prospective acquisition timing and clinical outcomes. The trained CNN's previously reported weak added-error audit sensitivity remains part of the scientific result.

## Execution history

The first Kaggle notebook attempt failed before validation because Kaggle expanded the uploaded ZIP. The second ran all 45 cases: 44 passed and KS043 correctly failed because recursive decompression removed the expected frozen `.csv.gz` artifact. Version 4 of the model source uses a byte-verified ZIP-format `.bundle` payload to preserve it. Neither attempt was counted as successful.

A later Windows run passed 44 cases and timed out in KS045's real training subprocess after 180 seconds; the report is retained in `validation/attempts/windows-subprocess-timeout.json`. The cause of that environment-specific timeout was not isolated. An isolated rerun of KS045 then passed in 55.612 seconds; its record is alongside the timeout report. This is a targeted retry, not a new complete Windows suite. Earlier local 45/45 evidence remains versioned separately, and hosted Linux runs completed the same training case. Latest verified states are in `release_status.json`.
