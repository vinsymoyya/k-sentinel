K-Sentinel is a CPU-reproducible synthetic benchmark asking when independent Fourier measurements can expose errors hidden by an AI MRI reconstruction's acquired-data consistency step.

This is an **unreviewed synthetic methods release**. It has no patient-data evaluation or clinical authorization.

The release contains an eight-page preprint, a five-page market analysis, a Numpy-only external audit API, a small optional PyTorch baseline, three trained checkpoints, a downloadable 16-object synthetic test subset, and all 129,024 evaluation records from 1,536 independent test phantoms. Code is MIT; generated data and numerical results are CC0-1.0.

At four extra audit columns, the small CNN flagged 191/1,536 controlled insertions (12.4%) and 86/1,536 clean cases (5.6%). The privileged oracle flagged 78.1% of insertions and is not deployable. Audit-aware errors evade the test, and using the same additional measurements for every reconstruction outperformed selective fusion for the tested CNN. These limits are central findings, not exclusions.

Start with the repository quickstart, read the protocol, and use the paper's full result table before interpreting the mechanism. The four-column audit adds 25% to the initial sample count; no actual scan-time or invasive-procedure reduction was measured. No comparison with a commercial reconstruction system was performed.

Checks include scientific contract tests, an end-to-end CPU smoke run, default-safe checkpoint loading, independent reaggregation of all 84 result groups, package creation and rendered-document inspection. Hosted Linux checks run Python 3.11 and 3.12 for installation, Numpy-based contracts, smoke execution and saved-result verification; the full CNN experiment was run locally on CPU.

Author: [Vinsy Moyya](https://orcid.org/0009-0003-0042-3556). Citation metadata and an importable BibTeX record are included. This release does not assert a DOI, peer-review acceptance or an updated ORCID work record. ResearchGate is excluded at the author's request.

Useful contributions: an independently reproduced run, a tested reconstruction-framework adapter, real multicoil raw-data validation under appropriate data terms, or a new reproducible audit failure case.
