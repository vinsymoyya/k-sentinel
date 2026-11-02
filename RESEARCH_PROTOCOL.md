# K-Sentinel: frozen exploratory protocol

Recorded before the first experiment, 2026-11-03. This is a local protocol, not an external preregistration.

## Question
When can held-out Cartesian Fourier measurements reveal a controlled reconstruction error that an acquired-data consistency check cannot see? What remains invisible, and what does the audit cost?

## Primary experiment
Synthetic complex-valued 64 x 64 randomized ellipse phantoms; single coil; unitary centered Fourier operator; 16/64 initially acquired phase-encoding columns including six central columns; independent circular complex Gaussian noise with E|noise|^2 = 0.0004. Independent train/calibration/test phantom identities, with 512/256/512 subjects per seed. Three seeds: 11, 29, 47. Three reconstruction controls: oracle with observed noisy coefficients (mechanistic control, not deployable), zero filling, and a small supervised residual CNN with exact acquired-coefficient projection. CNN trained eight epochs with fixed settings and no test-based selection.

Reserve 1, 2, 4, or 8 randomly selected unacquired columns, independent of the candidate reconstruction. Do not feed audit measurements or mask identities to the CNN or the ordinary injection mechanism. Audit score is mean squared complex Fourier residual divided by known measurement noise power. Calibrate a separate split-conformal upper threshold for each reconstruction and budget on clean calibration subjects, at alpha=0.05. Thresholds control clean rejection only under exchangeability; they do not guarantee anatomical correctness or error detection.

## Controlled challenges
Clean; projected positive Gaussian template; projected negative Gaussian template; a single unacquired-column perturbation; and a deliberately audit-aware perturbation in a column outside both initial and audit masks. Each perturbation has relative complex-image L2 norm 0.10. The projected Gaussian is a lesion-like error template, not a biological lesion. Counterfactual insertion/removal describes error sign, not measured disease sensitivity. Single-line and audit-aware errors are nonanatomical mathematical stress controls. Native CNN error is reported separately from injected error.

## Endpoints
Primary: clean rejection, insertion rejection, and audit-aware rejection for oracle control at four extra columns. Secondary: all budgets, other models, removal, single-line rejection, native NRMSE, exact observed residual, same-subject score AUROC, paired changes in reconstruction NRMSE when the same audit measurements are fused for everyone versus only flagged subjects. Neither audit arm receives additional measurements beyond its stated budget. No scan-time or clinical outcome endpoint.

Report numerator/denominator, Wilson 95% intervals as descriptive within-simulation intervals, per-seed rates, and paired subject bootstrap intervals for policy NRMSE differences. Shared calibration/model uncertainty is not captured by pooled Wilson intervals. No claims about generalization from three training seeds. No post-hoc removal of failed cases. Save every per-subject row. No tuning against test data.

## Failure and interpretation requirements
Explicitly show that acquired consistency is blind by construction, single-line detection is limited by inclusion probability b/48, and an audit-aware perturbation can evade this audit. Test higher-noise and shifted-anatomy clean inputs with frozen thresholds; these are distribution-shift probes, without nominal coverage guarantees. Audit fusion can worsen noise or fail to repair errors. Acquisition fractions count Fourier columns, not actual scan duration. Compare all-subject fusion at the same audit budget before claiming any selective policy benefit.

## Release boundary
An open synthetic methods benchmark and unreviewed preprint. No patient data, no diagnosis, no demonstrated reduction in invasive procedures, no regulatory authorization, and no claim of a new null-space theorem, hold-out method, conformal method, or state-of-the-art MRI model. Real multicoil raw-data validation and clinician-reviewed endpoints are future work.
