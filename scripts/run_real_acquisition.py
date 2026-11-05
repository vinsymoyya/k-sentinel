"""Run the frozen, participant-disjoint M4Raw natural-failure experiment."""
import argparse
from collections import defaultdict
import copy
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ksentinel.real_acquisition import (SLICES, audit_score, dump_json, fftc, file_sha256,
    fuse, ifftc, load_acquisition, observed_input, rank_pvalue, reconstruct_coils,
    relative_error, rss, sampling, sampling_seed, validate_manifest)
from ksentinel.model_io import load_model
from ksentinel.learned import channels
from ksentinel.core import paired_auc, threshold, wilson

SEED = 20261105
METHODS = ('zero_fill', 'phantom_cnn', 'm4raw_finetuned_cnn')


def fine_tune(model, manifest, data, output):
    inputs, targets, identities = [], [], []
    for item in manifest['files']:
        if item['role'] != 'train':
            continue
        kspace, _, support, _ = load_acquisition(data, item)
        mask, _, acs = sampling(support, 4, 'random', sampling_seed(item['scan_id'], 'T2', 4, 'random'))
        for frame in kspace:
            measured, scales = observed_input(frame, mask, acs)
            inputs.extend(ifftc(measured) / scales[:, None, None])
            targets.extend(ifftc(frame) / scales[:, None, None])
        identities.append(item['scan_id'])
    x, y = channels(np.stack(inputs)), channels(np.stack(targets))
    del inputs, targets
    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(SEED)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    history = []
    model.train()
    for epoch in range(8):
        total = 0.
        for ids in torch.randperm(len(x), generator=generator).split(8):
            optimizer.zero_grad()
            loss = ((model(x[ids]) - y[ids]) ** 2).mean()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite fine-tuning loss')
            loss.backward()
            optimizer.step()
            total += loss.item() * len(ids)
        history.append(total / len(x))
        print(f'Fine-tune epoch {epoch+1}/8 MSE={history[-1]:.7f}', flush=True)
    model.eval()
    checkpoint = output / 'm4raw_finetuned_seed20261105.pt'
    torch.save({'state_dict': model.state_dict(), 'history': history,
                'config': {'seed': SEED, 'epochs': 8, 'learning_rate': .0001, 'batch_size': 8,
                           'torch': str(torch.__version__), 'participants': identities,
                           'coil_slice_samples': len(x), 'slices': list(SLICES),
                           'reference': 'All available noisy source lines; not noise-free anatomy'}}, checkpoint)
    return model, {'participants': identities, 'coil_slice_samples': len(x), 'training_mse': history,
                   'checkpoint_sha256': file_sha256(checkpoint)}


def condition_key(row):
    return (row['sequence'], row['pattern'], int(row['acceleration']), row['method'])


def evaluate(item, kspace, reference, support, models, rss_check):
    records = []
    for pattern in ('random', 'equispaced'):
        for acceleration in (2, 4, 8):
            mask, audit, acs = sampling(support, acceleration, pattern,
                sampling_seed(item['scan_id'], item['sequence'], acceleration, pattern))
            for method, model in models.items():
                metrics = []
                for index, (frame, target_rss) in enumerate(zip(kspace, reference)):
                    candidate = reconstruct_coils(model, frame, mask, acs)
                    audit_values = frame[..., audit]
                    fused = fuse(candidate, audit, audit_values)
                    target_coils = ifftc(frame)
                    metrics.append({'slice': SLICES[index], 'audit_score': audit_score(candidate, audit, audit_values),
                        'rss_nrmse': relative_error(rss(candidate), target_rss),
                        'coil_nrmse': relative_error(candidate, target_coils),
                        'fused_rss_nrmse': relative_error(rss(fused), target_rss),
                        'fused_coil_nrmse': relative_error(fused, target_coils),
                        'acquired_relative_residual': relative_error(fftc(candidate)[..., mask], frame[..., mask])})
                means = {key: float(np.mean([m[key] for m in metrics])) for key in metrics[0] if key != 'slice'}
                records.append({'scan_id': item['scan_id'], 'path': item['relative_path'], 'role': item['role'],
                    'motion_type': item['motion_type'], 'sequence': item['sequence'], 'repetition': int(item['repetition']),
                    'pattern': pattern, 'acceleration': acceleration, 'method': method,
                    'source_acquired_columns': int(support.sum()), 'stored_columns': len(support),
                    'initial_columns': np.flatnonzero(mask).tolist(), 'audit_columns': audit.tolist(),
                    'source_rss_check_relative_error': rss_check, **means, 'slice_metrics': metrics})
    return records


def bootstrap_mean(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None
    rng = np.random.default_rng(SEED)
    estimates = values[rng.integers(0, len(values), size=(2000, len(values)))].mean(axis=1)
    return {'mean': float(values.mean()), 'interval_95': np.quantile(estimates, [.025, .975]).tolist(),
            'participants': len(values), 'resamples': 2000}


def summarize(rows):
    calibration = defaultdict(list)
    for row in rows:
        if row['role'] == 'calibration':
            calibration[condition_key(row)].append(row)
    calibrations = []
    for key, items in sorted(calibration.items()):
        if len(items) != len({r['scan_id'] for r in items}):
            raise ValueError('Repeated calibration participant within condition')
        scores = [r['audit_score'] for r in items]
        error_cutoff = float(np.quantile([r['rss_nrmse'] for r in items], .9, method='higher'))
        calibrations.append({'condition': list(key), 'subjects': len(items), 'threshold': threshold(scores),
                             'high_reference_error_cutoff': error_cutoff, 'scores': scores})
    lookup = {tuple(c['condition']): c for c in calibrations}
    groups = defaultdict(list)
    for row in rows:
        if row['role'] in ('train', 'calibration'):
            continue
        calibration_record = lookup[condition_key(row)]
        row['p_value'] = rank_pvalue(row['audit_score'], calibration_record['scores'])
        row['flagged'] = row['p_value'] <= .05
        row['high_reference_error'] = row['rss_nrmse'] > calibration_record['high_reference_error_cutoff']
        row['selective_rss_nrmse'] = row['fused_rss_nrmse'] if row['flagged'] else row['rss_nrmse']
        key = (row['role'], row['motion_type'], row['sequence'], row['repetition'], row['pattern'], row['acceleration'], row['method'])
        groups[key].append(row)
    summaries = []
    for key, items in sorted(groups.items()):
        if len(items) != len({r['scan_id'] for r in items}):
            raise ValueError('Repeated participant inside a reported group')
        high = [r for r in items if r['high_reference_error']]
        low = [r for r in items if not r['high_reference_error']]
        summaries.append({'role': key[0], 'motion_type': key[1], 'sequence': key[2], 'repetition': key[3],
            'pattern': key[4], 'acceleration': key[5], 'method': key[6], 'participants': len(items),
            'flagged': sum(r['flagged'] for r in items),
            'flag_fraction_interval_95_wilson': wilson(sum(r['flagged'] for r in items), len(items)),
            'mean_rss_nrmse': float(np.mean([r['rss_nrmse'] for r in items])),
            'mean_coil_nrmse': float(np.mean([r['coil_nrmse'] for r in items])),
            'high_reference_error_cases': len(high), 'high_reference_error_flagged': sum(r['flagged'] for r in high),
            'unflagged_high_reference_error': sum(not r['flagged'] for r in high),
            'reference_error_auroc': paired_auc([r['audit_score'] for r in high], [r['audit_score'] for r in low]) if high and low else None,
            'all_fusion_minus_no_fusion_rss': bootstrap_mean([r['fused_rss_nrmse'] - r['rss_nrmse'] for r in items]),
            'selective_minus_all_fusion_rss': bootstrap_mean([r['selective_rss_nrmse'] - r['fused_rss_nrmse'] for r in items])})
    primary = [r for r in rows if r['role']=='test' and r['sequence']=='T2' and r['repetition']==1
               and r['pattern']=='random' and r['acceleration']==4 and r['method']=='m4raw_finetuned_cnn']
    failures = sorted(primary, key=lambda r: (-r['rss_nrmse'], r['scan_id']))[:5]
    unflagged = sorted((r for r in primary if not r['flagged']), key=lambda r: (-r['rss_nrmse'], r['scan_id']))[:5]
    repeats = []
    second = {r['scan_id']: r for r in rows if r['role']=='test' and r['sequence']=='T2' and r['repetition']==2
              and r['pattern']=='random' and r['acceleration']==4 and r['method']=='m4raw_finetuned_cnn'}
    for row in primary:
        other = second[row['scan_id']]
        repeats.append({'scan_id': row['scan_id'], 'rss_nrmse_rep1': row['rss_nrmse'],
                        'rss_nrmse_rep2': other['rss_nrmse'], 'flagged_rep1': row['flagged'], 'flagged_rep2': other['flagged']})
    return {'calibrations': calibrations, 'groups': summaries, 'primary_rows': primary,
            'highest_error_cases': failures, 'worst_unflagged_cases': unflagged,
            'paired_repeats': repeats, 'paired_repetition_error_change': bootstrap_mean([r['rss_nrmse_rep2']-r['rss_nrmse_rep1'] for r in repeats])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default='research/real-acquisition/manifest.json')
    parser.add_argument('--data', default='tmp/m4raw/data')
    parser.add_argument('--output', default='results/real-acquisition-0.3.0')
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    roles = validate_manifest(manifest)
    if len(manifest['files']) != 180 or len(roles) != 100:
        raise ValueError('This frozen run requires all 180 acquisitions and 100 participants')
    output = Path(args.output)
    if output.exists():
        raise ValueError('Use a new output directory; do not overwrite an experimental record')
    output.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED)
    start = time.perf_counter()
    frozen_path = ROOT / 'results/main/cnn_seed11.pt'
    frozen = load_model(frozen_path)
    models = {'zero_fill': None, 'phantom_cnn': frozen}
    config = {'protocol_sha256': file_sha256('research/real-acquisition/PROTOCOL.md'),
        'manifest_sha256': file_sha256(args.manifest), 'frozen_checkpoint_sha256': file_sha256(frozen_path),
        'source_sha256': {str(p.relative_to(ROOT)).replace('\\','/'): file_sha256(p)
                          for p in [ROOT/'ksentinel/real_acquisition.py', Path(__file__).resolve()]},
        'seed': SEED, 'slices': list(SLICES), 'audit_columns': 8, 'acs_columns': 16,
        'python': platform.python_version(), 'numpy': np.__version__, 'torch': str(torch.__version__),
        'platform': platform.platform(), 'complete': False, 'dataset_revision': manifest['revision'],
        'claim_scope': 'Exploratory single-scanner healthy-volunteer source-reference experiment; no clinical validation'}
    dump_json(output/'config.json', config)
    tuned, training = fine_tune(copy.deepcopy(frozen), manifest, args.data, output)
    models['m4raw_finetuned_cnn'] = tuned
    dump_json(output/'training.json', training)
    rows, source_checks = [], []
    evaluated = [item for item in manifest['files'] if item['role'] != 'train']
    with (output/'acquisitions.jsonl').open('w', encoding='utf-8') as stream:
        for index, item in enumerate(evaluated, 1):
            kspace, reference, support, check = load_acquisition(args.data, item)
            new = evaluate(item, kspace, reference, support, models, check)
            rows.extend(new)
            source_checks.append({'path': item['relative_path'], 'source_sha256': item['sha256'],
                                  'acquired_columns': int(support.sum()), 'rss_relative_check': check})
            for row in new:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
            stream.flush()
            print(f"Evaluated {index}/{len(evaluated)} {item['relative_path']}", flush=True)
    summary = summarize(rows)
    dump_json(output/'summary.json', summary)
    dump_json(output/'source_checks.json', source_checks)
    with (output/'records.csv').open('w', newline='', encoding='utf-8') as stream:
        fields = list(dict.fromkeys(k for row in rows for k in row
                      if k not in ('initial_columns','audit_columns','slice_metrics')))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k:r.get(k) for k in fields} for r in rows)
    with (output/'records.jsonl').open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
    config.update(complete=True, seconds=time.perf_counter()-start, rows=len(rows), participants=len(roles))
    dump_json(output/'config.json', config)
    print(json.dumps({'complete': True, 'rows': len(rows), 'seconds': config['seconds'],
        'primary_participants': len(summary['primary_rows'])}, indent=2), flush=True)


if __name__ == '__main__':
    main()
