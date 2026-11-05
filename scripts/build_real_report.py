"""Write a data-driven, explicitly exploratory real-MRI research addendum."""
import argparse
import json
from pathlib import Path
import numpy as np

LABELS = {'zero_fill':'Zero fill','phantom_cnn':'Frozen phantom CNN','m4raw_finetuned_cnn':'Fine-tuned CNN'}


def interval(values):
    values = np.asarray(values,dtype=float)
    rng = np.random.default_rng(20261105)
    sampled = values[rng.integers(0,len(values),(2000,len(values)))].mean(axis=1)
    lo,hi = np.quantile(sampled,[.025,.975])
    return f'{values.mean():.5f} [{lo:.5f}, {hi:.5f}]'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',default='results/real-acquisition-0.3.0')
    parser.add_argument('--output',default='research/real-acquisition/REPORT.md')
    args = parser.parse_args()
    folder = Path(args.results)
    config = json.loads((folder/'config.json').read_text(encoding='utf-8'))
    summary = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    verification = json.loads(Path('validation/real-acquisition-verification.json').read_text(encoding='utf-8'))
    transfer = json.loads((folder/'exploratory_transfer.json').read_text(encoding='utf-8'))
    if config['complete'] is not True or verification['status']!='passed':
        raise ValueError('A complete, independently reconciled experiment is required')
    rows = [json.loads(line) for line in (folder/'records.jsonl').read_text(encoding='utf-8').splitlines()]
    groups = summary['groups']
    primary = [g for g in groups if g['role']=='test' and g['sequence']=='T2' and g['repetition']==1 and g['pattern']=='random' and g['acceleration']==4]
    learned = next(g for g in primary if g['method']=='m4raw_finetuned_cnn')
    calibration = next(c for c in summary['calibrations'] if c['condition']==['T2','random',4,'m4raw_finetuned_cnn'])
    out=[]
    def line(text=''):out.append(text+'\n')
    line('# Real acquisition variations and natural reference-error failures')
    line('## An exploratory K-Sentinel methods addendum')
    line('Vinsy Moyya | ORCID: https://orcid.org/0009-0003-0042-3556 | 2026-11-05 | Unreviewed research')
    line('## Main result')
    line(f"The complete experiment used **180 real M4Raw acquisitions from 100 participant-disjoint volunteers**. In the predeclared primary condition (T2, repeat 1, random retrospective R=4, fine-tuned CNN), mean RSS reference NRMSE was **{learned['mean_rss_nrmse']:.4f}**. The audit flagged **{learned['flagged']}/25** participants. **{learned['high_reference_error_cases']}** exceeded the calibration-defined high-reference-error cutoff; **{learned['unflagged_high_reference_error']}** of those remained unflagged.")
    line('These endpoints concern agreement with the same acquisition\'s all-available, noisy source reconstruction. They do not establish anatomical correctness, detection of unsupported focal features, motion correction, diagnosis or patient benefit. Every deliberately injected image-structure challenge from the earlier synthetic study is absent here.')
    line('## Why this extension was needed')
    line('Community feedback asked whether measurement audits would remain useful under real acquisition variation and naturally occurring reconstruction failures. The original v0.1.0 paper and v0.2.0 synthetic results are preserved. This extension evaluates actual scanner-exported, source-corrected four-coil MRI measurements, rather than treating a Fourier transform of a magnitude image as raw acquisition data.')
    line('## Data, selection and licensing')
    line('M4Raw was acquired from healthy volunteers on one 0.3-T scanner with a four-channel head coil. We pinned the original creators\' byte-preserving Hugging Face mirror to revision `46cb84455c0d6653b3b4afe27d56b10fbd4cd424`. The selected 180 H5 files total 2,221,536,429 bytes; every download was checked against its source SHA256 and size.')
    line('| Role | Distinct participants | Acquisitions used | Use |')
    line('|---|---:|---:|---|')
    line('| Training | 20 | 20 | T2 repeat 1, fixed fine-tuning |')
    line('| Calibration | 30 | 60 | T2 and FLAIR repeat 1, threshold fitting only |')
    line('| Expanded test | 25 | 75 | T2 repeats 1/2 and FLAIR repeat 1 |')
    line('| Motion-risk stress | 25 | 25 | T2 repeat 1; 8 inter-scan / 17 intra-scan participants |')
    line('The three fixed stored slices 7/8/9 are averaged to one acquisition-level score and error. All repeats stay in the same participant role. **2,880 result rows and repeated slices are not 2,880 independent people.** Motion-risk assignment is participant-level: an individual H5 is not thereby proven visibly corrupted. The data contain no disease or focal-lesion truth labels.')
    line('M4Raw data, derived selection metadata and source-based figures are CC BY 4.0, attributed to Lyu et al. Code and original analysis prose are MIT. The older CC0 synthetic dataset has a separate license. Public raw H5 payloads are downloaded from the creators, not relabeled as K-Sentinel synthetic data.')
    line('## Reconstruction and acquisition design')
    line('We compare zero filling, the immutable released phantom CNN seed 11, and that same architecture fine-tuned on twenty separate T2 training participants. Fine-tuning uses eight fixed epochs, Adam learning rate 0.0001, batch size 8 and seed 20261105. There are 240 training coil-slices, but only twenty training people. No calibration or test data update weights or select a checkpoint. The compact per-coil CNN does not exploit joint coil-sensitivity encoding and is not a state-of-the-art reconstruction comparator.')
    line('Native complex 256 x 256 arrays and all four coils are retained. Centered orthonormal FFT/IFFT is reconciled against the source RSS images. T2 has 195 acquired PE columns and FLAIR 198; the other stored columns are unavailable zero padding. Retrospective masks and audit samples are drawn only from actual source support.')
    line('Six retrospective settings combine nominal R=2/4/8 with random or approximately equispaced sampling. Sixteen initially acquired central ACS columns determine per-coil normalization. Reference images and hidden audit samples do not determine normalization or CNN input. Eight other acquired source columns are reserved for the independent audit. Initial measurements are projected back into every candidate exactly.')
    line('For primary T2 R=4, initial sampling retains 49/195 available columns; auditing raises this to 57/195. Eight columns are a 16.3% increase over the initial retained count. This is retrospective sample accounting, not measured scanner time or a prospective sequence experiment.')
    line('Actual variations are repeated scans, T2 versus FLAIR acquisitions, and participant-level motion-risk cohorts. Random/equispaced masks and R changes are retrospective simulations. No synthetic motion, added noise or artificial image structure is used in the present experiment.')
    line('## Calibration and reference-error definitions')
    line('The audit averages, over three slices, squared complex residual on hidden columns divided by their observed signal energy. It is not a whitened noise-likelihood statistic. Thirty separate calibration participants fit a threshold for each contrast/model/sampling condition. Rank p-values count conservative ties; scores are rounded to twelve decimal places to suppress numerical discrimination in the analytically constant zero-fill control. With thirty participants the minimum p-value is 1/31, and alpha=0.05 flags only scores above the largest calibration score, apart from conservative numerical ties.')
    line(f"High reference error uses the calibration 90th-percentile order statistic; for the primary fine-tuned condition the RSS NRMSE cutoff is **{calibration['high_reference_error_cutoff']:.5f}**. This is a benchmark error class, not a clinical failure or disease label. AUROC is reported only when both reference-error classes exist. No nominal coverage guarantee is asserted under cohort/acquisition shift.")
    line('## Primary results: 25 independent held-out participants')
    line('| Method | Mean RSS NRMSE | Audit flags | High-error cases | High-error unflagged | Reference-error AUROC |')
    line('|---|---:|---:|---:|---:|---:|')
    for g in primary:
        auc='n/a' if g['reference_error_auroc'] is None else f"{g['reference_error_auroc']:.3f}"
        line(f"| {LABELS[g['method']]} | {g['mean_rss_nrmse']:.5f} | {g['flagged']}/25 | {g['high_reference_error_cases']} | {g['unflagged_high_reference_error']} | {auc} |")
    by_method = {m:{r['scan_id']:r['rss_nrmse'] for r in rows if r['role']=='test' and r['sequence']=='T2' and r['repetition']==1 and r['pattern']=='random' and r['acceleration']==4 and r['method']==m} for m in LABELS}
    ids=sorted(by_method['m4raw_finetuned_cnn'])
    for baseline in ('zero_fill','phantom_cnn'):
        differences=[by_method['m4raw_finetuned_cnn'][s]-by_method[baseline][s] for s in ids]
        line(f"Fine-tuned minus {LABELS[baseline].lower()} paired RSS NRMSE difference: **{interval(differences)}**, mean [95% participant-bootstrap interval]. Negative differences favor the fine-tuned model. These intervals condition on the fitted model and do not capture training/calibration-set uncertainty.")
    line('## Acquisition changes and motion-risk stress')
    line('| Actual acquisition / cohort | Participants | Mean RSS NRMSE | Matched-calibration flags | Frozen T2/R4 flags |')
    line('|---|---:|---:|---:|---:|')
    for g in groups:
        if g['method']!='m4raw_finetuned_cnn' or g['pattern']!='random' or g['acceleration']!=4:
            continue
        t=next(t for t in transfer['groups'] if all(t[k]==g[k] for k in ('role','motion_type','sequence','repetition','pattern','acceleration','method')))
        label=f"{g['sequence']} repeat {g['repetition']}" if g['role']=='test' else f"T2 {g['motion_type']} risk"
        line(f"| {label} | {g['participants']} | {g['mean_rss_nrmse']:.5f} | {g['flagged']}/{g['participants']} | {t['frozen_T2_R4_flags']}/{g['participants']} |")
    paired=summary['paired_repetition_error_change']
    line(f"Actual T2 repeat 2 minus repeat 1 paired mean RSS NRMSE change: **{paired['mean']:.5f} [{paired['interval_95'][0]:.5f}, {paired['interval_95'][1]:.5f}]**, 95% participant-bootstrap interval, n=25 paired people. No repeat was selected as a cleaner anatomical truth.")
    line('The frozen-threshold column is an additional exploratory analysis, specified after the main protocol during implementation and before inspecting held-out summaries. It reuses the exact T2/random/R4 calibration scores across changes, without recalibration or model updates. It does not change primary outcomes. Condition-matched and transferred thresholds answer different questions; neither supplies clinical validation.')
    line('## Native sampling failures and same-budget fusion')
    line('| Pattern | Nominal R | Zero-fill RSS NRMSE | Frozen CNN RSS NRMSE | Fine-tuned RSS NRMSE | Fine-tuned flags |')
    line('|---|---:|---:|---:|---:|---:|')
    for pattern in ('random','equispaced'):
        for r in (2,4,8):
            selected={g['method']:g for g in groups if g['role']=='test' and g['sequence']=='T2' and g['repetition']==1 and g['pattern']==pattern and g['acceleration']==r}
            line(f"| {pattern} | {r} | {selected['zero_fill']['mean_rss_nrmse']:.5f} | {selected['phantom_cnn']['mean_rss_nrmse']:.5f} | {selected['m4raw_finetuned_cnn']['mean_rss_nrmse']:.5f} | {selected['m4raw_finetuned_cnn']['flagged']}/25 |")
    fusion=learned['all_fusion_minus_no_fusion_rss'];selective=learned['selective_minus_all_fusion_rss']
    line(f"Primary all-subject audit fusion minus no fusion: mean RSS NRMSE difference **{fusion['mean']:.5f} [{fusion['interval_95'][0]:.5f}, {fusion['interval_95'][1]:.5f}]**. Selective minus all-subject fusion: **{selective['mean']:.5f} [{selective['interval_95'][0]:.5f}, {selective['interval_95'][1]:.5f}]**. Both policies receive the same eight audit columns. The source reference includes those same measurements, so improved agreement is not independent evidence of anatomical or clinical improvement.")
    line('## Largest unflagged reference errors')
    line('| Public scan ID | Mean RSS NRMSE | Mean audit score | Rank p-value | High-reference-error class |')
    line('|---|---:|---:|---:|---|')
    for r in summary['worst_unflagged_cases']:
        line(f"| {r['scan_id']} | {r['rss_nrmse']:.5f} | {r['audit_score']:.5f} | {r['p_value']:.5f} | {'yes' if r['high_reference_error'] else 'no'} |")
    line('Cases are selected by the fixed rule highest mean reference error among unflagged primary cases, with scan-ID tie breaking. The figure shows stored slice 8; ranking averages slices 7/8/9. No model or threshold was tuned on these cases. Pixel differences alone do not prove an unsupported lesion or a missed clinical finding.')
    line('## Reproducibility and verification')
    line(f"The complete record contains **{verification['rows_checked']}** acquisition/condition/model rows, **{verification['calibration_strata']}** calibration strata and **{verification['reported_groups']}** reported test/stress groups. Every saved mask, per-slice mean, flag, high-error label and group summary is reconciled independently. Maximum acquired relative residual is **{verification['max_acquired_relative_residual']:.3g}**; maximum stored-source RSS reconciliation error among 160 evaluation/calibration acquisitions is **{verification['max_stored_source_rss_relative_error']:.3g}**.")
    line(f"The completed cloud run used Python {config['python']}, NumPy {config['numpy']} and PyTorch {config['torch']}. The original 45 scientific/software contracts plus 12 new real-acquisition contracts pass without skips on GitHub and in Kaggle. Downloaded JSON/JUnit evidence, all result rows, fixed protocol/manifest, source hashes and the fine-tuned checkpoint accompany the release. The original synthetic results remain byte-preserved.")
    line('Reproduce with `scripts/download_m4raw.py`, then `scripts/run_real_acquisition.py` in a fresh output directory. Independently reconcile with `scripts/verify_real_results.py`. The public Kaggle notebook is https://www.kaggle.com/code/vinsymoyya/k-sentinel-real-mri-variations-and-failures .')
    line('## Limits and next validation')
    line('This is a single-scanner, four-coil, low-field healthy-volunteer pilot, with twenty fine-tuning people and twenty-five primary test people. Three fixed slices do not characterize an entire volume. Source references contain noise, partial PE support and possible motion. The per-coil CNN is a compact baseline; modern sensitivity-aware/unrolled reconstruction systems require separate comparisons. Thirty calibration people yield coarse tail ranks; repeated subjects are not new independent evidence. Bootstrap intervals condition on fitted models and calibration. Secondary comparisons are exploratory and do not support multiplicity-corrected superiority claims.')
    line('Actual contrast/repeat/motion-risk variation has been studied; new vendors, field strengths, prospective sequences, measured scan time, pathology cohorts and clinician-reviewed unsupported-feature labels remain untested. Next steps are stronger open multicoil models, independent-site raw data, registered clinical endpoints and reader review of native failure cases. The present result supports open error analysis and reproducibility, not clinical deployment or a proven reduction in invasive procedures.')
    line('## Figures')
    for title,name in [('Sampling and audit','sampling_and_audit'),('Actual acquisition variations','actual_acquisition_variations'),('Audit versus natural reference error','audit_vs_natural_error'),('Fixed-rule native failure cases','natural_failure_cases')]:
        line(f'![{title}](../../assets/real-acquisition/{name}.png)')
    line('Source-based figures: M4Raw, Lyu et al. (2023), CC BY 4.0. They retain their own attribution and are not licensed as the older CC0 synthetic data.')
    line('## References')
    line('Lyu M, Mei L, Huang S, et al. M4Raw: A multi-contrast, multi-repetition, multi-channel MRI k-space dataset for low-field MRI research. Scientific Data 10, 264 (2023). https://doi.org/10.1038/s41597-023-02181-4 .')
    line('M4Raw original data, version 1.6: https://doi.org/10.5281/zenodo.8056074 . Original-creators\' source and data card: https://huggingface.co/datasets/mylyu/M4Raw_brain/tree/46cb84455c0d6653b3b4afe27d56b10fbd4cd424 . License: https://creativecommons.org/licenses/by/4.0/ .')
    line('K-Sentinel fixed protocol, manifest and attribution: research/real-acquisition/PROTOCOL.md, manifest.json, SOURCE_NOTICE.md. Additional transfer analysis: EXPLORATORY_ANALYSES.md. Original unreviewed synthetic paper: v0.1.0; synthetic validation/data/model release: v0.2.0. Neither is rewritten by this addendum.')
    Path(args.output).write_text('\n'.join(out),encoding='utf-8')
    print(args.output)


if __name__=='__main__':
    main()
