# K-Sentinel real-acquisition exploratory protocol

Recorded 2026-11-05 before downloading scientific H5 payloads or inspecting model results. This is a frozen local analysis plan, not an external preregistration. Version 0.2.0 synthetic results remain immutable.

## Question and boundaries

Do independent raw measurements expose naturally occurring reconstruction errors on real MRI acquisitions, without adding artificial image structures? How do outcomes change across repeated acquisitions, T2/FLAIR contrast, retrospective sampling, and a participant-level motion-risk cohort?

Source: Lyu et al., *M4Raw*, Scientific Data 10:264 (2023), DOI 10.1038/s41597-023-02181-4; data DOI 10.5281/zenodo.8056074; original-creators' byte-preserving Hugging Face mirror `mylyu/M4Raw_brain`, pinned revision `46cb84455c0d6653b3b4afe27d56b10fbd4cd424`. Data license CC BY 4.0, separate from MIT code and prior CC0 synthetic data. The manifest records every selected source file, SHA256, size, participant, sequence and repetition before evaluation.

## Fixed selection

100 participants, 180 H5 acquisitions, 2,221,536,429 bytes. Twenty original training participants selected by ascending SHA256 of `20261105:<scan_id>`; their T2 repetition 1 is used only for fine-tuning. All 30 original validation participants supply calibration T2/FLAIR repetition 1. All 25 expanded-test participants supply T2 repetitions 1/2 and FLAIR repetition 1. All 25 motion-risk participants supply T2 repetition 1. Participants and all their repetitions remain in one role. No test participant enters training or calibration. The fixed stored slices are 7, 8, 9, selected without viewing images. Slices and repeats are never counted as independent people.

## Models and actual measurements

Compare zero filling; the frozen released phantom CNN `results/main/cnn_seed11.pt`; and that same CNN fine-tuned only on the twenty training participants, T2 repetition 1, three slices, all four coils. Fine-tune eight fixed epochs, Adam learning rate 0.0001, batch size 8, seed 20261105, two CPU threads. No early stopping, model search, test-based selection or real-data updates to the frozen baseline. Targets use all *available* source lines; they are noisy source reconstructions, not noise-free anatomy.

Retain native 256 x 256 complex four-coil data. Use centered orthonormal FFT/IFFT compatible with the creators' stored RSS reference. Verify recomputed RSS against stored RSS. Derive source PE support from nonzero raw samples and reconcile against the manifest: T2 has 195 acquired lines, FLAIR 198; zero-padded unavailable columns are ineligible for initial and audit sampling.

Retrospectively retain round(available_lines/R) columns for R=2,4,8, with 16 available central ACS columns and either uniform-random or approximately equispaced remaining columns. All conditions share a deterministic ordering of remaining lines; eight independent audit columns are hidden from reconstruction and normalization. Record exact masks/counts. Retrospective acceleration is relative to source support, not 256; it is not prospective scan-time evidence. Models see only acquired coefficients. Normalize each coil by its initial-ACS low-resolution magnitude 99th percentile, never source RSS or audit data. Project model predictions back onto initial measurements exactly. No inserted structures, synthetic noise or synthetic motion in this experiment.

## Audit and natural-error endpoints

Audit score: squared complex residual on the eight audit columns divided by their squared observed signal norm, pooled across coils. It is an empirical relative residual, not a whitened chi-square statistic or known-noise likelihood. Average scores over the three fixed slices to obtain one acquisition/participant score. Fit separate rank-based upper thresholds at alpha=0.05 using 30 calibration participants per contrast/model/sampling condition. Test and motion-risk thresholds remain frozen. Do not claim nominal coverage under cohort or acquisition shift.

Measure complex coil NRMSE and magnitude RSS NRMSE against the same acquisition's all-available source reconstruction. The latter can contain noise and motion; consistency with it cannot establish correct anatomy. Report all-source reference limitations even if residuals are small. Compute acquired residual, source support, audit support and reference error for every case. No biological lesion, unsupported focal-feature or diagnostic labels are available.

Primary stratum: T2 repetition 1, random R=4, fine-tuned CNN, all 25 expanded-test participants. Primary descriptive endpoints: audit flag fraction; RSS reference NRMSE; and count of unflagged high-reference-error cases. Define high reference error using the calibration 90th percentile (order statistic with `higher` interpolation), never test-selected thresholds. Report error-detection AUROC only when both reference-error classes exist; call these reference-error classes, not disease labels. Show the five highest-reference-error cases and the five worst unflagged cases by a fixed sorting rule, including every original result row.

Secondary: other models and sampling conditions, FLAIR, paired T2 repetition changes, motion-risk inter/intra participant subsets, and same-budget audit fusion applied to everyone versus only flagged cases. Compute primary paired reference-error changes with participant bootstrap intervals (2,000 resamples, seed 20261105), conditional on this fitted model/calibration. Repeats are paired within participant. No aggregate treats repeated slices as independent. No corrected multiple-comparison superiority claims; all secondary analyses exploratory.

Scores are rounded to 12 decimal places; numerical ranks conservatively treat relative/absolute differences up to 1e-12 as ties. This prevents FFT round-off in the analytically constant zero-fill score from creating artificial discrimination. These implementation details were set before the first full model experiment; the first cloud submission was rejected for a notebook title/slug mismatch and executed no experiment.

## Release criteria

Retain source/data/checkpoint hashes, fixed selection and settings, training history, all acquisition-level rows, calibration thresholds, numerical summaries, source-reference checks and failure-case identifiers. Run new real-acquisition operator/leakage/data tests in a separate suite and preserve the existing 45-case synthetic suite. A real-data run must finish before outcomes are called demonstrated. Publish a clearly labeled exploratory real-data extension only after verification. No clinical, multi-vendor, motion-correction or prospective acquisition benefit is established by this single-scanner healthy-volunteer study.
