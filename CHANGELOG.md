# Changelog

## 0.3.0 - 2026-11-05

- Completed a source-pinned M4Raw real-acquisition study: 100 disjoint participants, 180 native four-coil acquisitions, actual repeats/contrast/motion-risk variation, six retrospective sampling settings and 2,880 result rows.
- Added centered raw-data adapters, support-aware masks, initial-only normalization, participant-separated fine-tuning/calibration and natural reference-error analysis without injected structures.
- Retained 45 legacy contracts and added 12 real-acquisition contracts; hosted GitHub/Kaggle runs pass without skips.
- Published independently reconciled outcomes, fine-tuned weights, source notices, native case figures and an unreviewed addendum. Fine-tuning did not outperform the frozen CNN; secondary FLAIR high-reference-error cases can remain unflagged.
- Preserved original synthetic paper/results/checkpoints and versioned licenses; M4Raw is separately attributed CC BY 4.0.


## 0.2.0 - 2026-11-03

Expand validation to 45 named contracts with JSON/JUnit evidence and no skipped passes. Add default-safe open-model loading, checksum validation, finite-weight checks and measurement-only inference; reject invalid audit calibration and numerical overflow; export the full 3,840-object acquisition dataset with integrity manifests; provide data-driven training and an offline public Kaggle notebook. Preserve original experimental source snapshots and all v0.1.0 numerical results. Distribution/hosted-run status is recorded separately and must be verified.

## 0.1.0 - 2026-11-03

Initial synthetic methods release: independent measurement audit API, procedural complex phantoms, three trained small CNNs, calibration and blind-spot controls, complete per-subject results, source/split provenance, scientific contract tests, independent reaggregation, preprint and market analysis. Weak learned-model sensitivity, audit evasion and the equal-budget fusion comparison are reported explicitly. No patient-data or clinical validation claim.
