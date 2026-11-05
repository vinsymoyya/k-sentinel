# Additional exploratory interpretation

Recorded during implementation on 2026-11-05, after the main protocol and cloud execution were started, before inspecting held-out outcome summaries. This supplements rather than changes the primary protocol; it is not an external preregistration.

The main experiment calibrates separately for every contrast/model/sampling condition. This evaluates within-condition natural-error discrimination but does not by itself demonstrate calibration robustness to acquisition changes.

Therefore also report a **frozen-threshold transfer analysis**: for each model, retain its T2 / random / R=4 calibration scores from the same thirty calibration participants, and apply those exact scores to all held-out contrasts, repetitions and sampling conditions. Do not recalibrate on test subjects or adjust the threshold after inspecting outcomes. Report participant counts and flag fractions. No nominal 5% guarantee is asserted under these acquisition or cohort shifts.

This secondary analysis reuses the completed, immutable acquisition-level score record; it neither retrains models nor changes primary labels, settings or reported outcomes. It must be labeled exploratory and separated from condition-matched calibration results. No diagnostic labels or clinician-reviewed focal-feature findings can be inferred from these scores.
