# Evidence-driven research roadmap

## Released synthetic baseline

- Hidden random audits, model-specific rank calibration, controlled measurement-blind errors.
- Three seeded small CNNs, independent split identities, complete results and negative controls.
- Same-budget universal versus selective fusion; no claimed selective-policy advantage.
- External NPZ audit interface and CPU smoke workflow.

## Next: external reconstruction evidence

Acceptance criteria: one tested DIRECT or DeepInverse adapter, an independently reproduced run, a frozen competitive multicoil baseline, documented data-use terms, subject-level partitioning, and all evaluated results. These adapters and data evaluations are not yet implemented. Do not redistribute restricted raw MRI data.

## Next: natural pathology and operator uncertainty

Measure native reconstruction failures near annotated pathology; compare acquisition strategies at equal budgets; vary coil sensitivity and noise correlation; freeze thresholds before external-site testing. Use clinician-defined endpoint relevance and retain errors the audit misses. Separate synthetic perturbation detection from clinical disease sensitivity.

## Prospective task-directed acquisition

Investigate randomized measurements that distinguish reconstructions disagreeing on a quantitative task. Preserve a hidden audit, account for repeated testing or stopping, and measure sequence timing. Compare the simplest all-data fusion policy and published uncertainty/adaptive-acquisition methods. Require an improvement on an appropriate endpoint before making efficiency claims.

## Clinical translation gate

Requires separately approved prospective protocols, independent sites, reader studies, failure handling, intended-use definition and applicable regulatory review. No part of the present release meets that gate. Avoid replacing evidence with a larger model or a broader title.
