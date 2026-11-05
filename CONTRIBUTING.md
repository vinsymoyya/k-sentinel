# Contributing

The most valuable contribution is evidence someone else can reproduce. Report Python/library versions, platform, exact command, seed, data provenance and the observed result. Include a small synthetic example rather than private data. Do not upload patient images, identifiers, credentials or data you are not permitted to redistribute.

## Start here

1. Run the CPU quickstart and existing tests.
2. Open an issue describing the question and intended evidence before a large change.
3. Keep one experiment or interface change per pull request.
4. Install `.[learned]`, then run `python scripts/run_validation.py --output validation/local`. All 45 cases must run with zero skips. Add new cases when material behavior needs coverage and update the catalogue/count contract explicitly; do not conceal new behavior to preserve a number.
5. For experimental changes, use a new output directory and protocol version; preserve the released record.

Useful small contributions: reproduce the analytic Fourier example; add a test for a new input-contract boundary; document installation on another operating system; or create a small adapter from a public reconstruction framework using synthetic values.

Research contributions: natural-error evaluation, correctly whitened multicoil noise, raw-data operator adapters, prospective acquisition accounting, and challenging errors that evade the audit. Compare the same acquisition budget, preserve subject-level splits, and label oracle information. Include negative results and every evaluated challenge category. Do not tune against `results/main` and then describe it as an untouched test set.

Do not infer diagnostic performance, vendor failure rates, patient benefit or regulatory authorization from this benchmark. Code contributions use MIT; newly contributed data must identify their own provenance and compatible license. Maintainers may reject data with unclear rights or patient privacy risk.

For the real-MRI extension, preserve participant roles across every contrast/repetition, actual PE support, hidden audit separation and M4Raw CC BY 4.0 attribution. Run the separate real-acquisition suite and independently reconcile full results. The published test record has been inspected: new model comparisons must disclose reuse and cannot describe it as a fresh untouched clinical validation set. Use independent data for new generalization claims.
