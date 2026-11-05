"""Independently reconcile the completed real-MRI record without rerunning training."""
import argparse
from collections import defaultdict, Counter
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', default='results/real-acquisition-0.3.0')
    parser.add_argument('--output', default='validation/real-acquisition-verification.json')
    args = parser.parse_args()
    folder = Path(args.results)
    config = read(folder/'config.json')
    summary = read(folder/'summary.json')
    training = read(folder/'training.json')
    manifest_path = ROOT/'research/real-acquisition/manifest.json'
    manifest = read(manifest_path)
    assert config['complete'] is True
    assert config['manifest_sha256'] == digest(manifest_path)
    assert config['protocol_sha256'] == digest(ROOT/'research/real-acquisition/PROTOCOL.md')
    for name, expected in config['source_sha256'].items():
        assert digest(ROOT/name) == expected, name
    assert digest(ROOT/'results/main/cnn_seed11.pt') == config['frozen_checkpoint_sha256']
    checkpoint = folder/'m4raw_finetuned_seed20261105.pt'
    assert digest(checkpoint) == training['checkpoint_sha256']
    expected_train = sorted({i['scan_id'] for i in manifest['files'] if i['role']=='train'})
    assert sorted(training['participants']) == expected_train
    assert len(expected_train) == 20 and training['coil_slice_samples'] == 240
    assert len(training['training_mse']) == 8 and np.isfinite(training['training_mse']).all()
    by_path = {i['relative_path']:i for i in manifest['files']}
    checks = read(folder/'source_checks.json')
    assert len(checks)==160 and len({r['path'] for r in checks})==160
    assert {r['path'] for r in checks}=={p for p,i in by_path.items() if i['role']!='train'}
    for row in checks:
        item = by_path[row['path']]
        assert row['source_sha256']==item['sha256']
        assert row['acquired_columns']==int(item['acquired_pe_lines'])
        assert 0 <= row['rss_relative_check'] <= 1e-5
    rows = [json.loads(line) for line in (folder/'records.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows)==2880 and config['rows']==2880 and config['participants']==100
    assert Counter(r['role'] for r in rows)=={'calibration':1080,'test':1350,'motion_risk':450}
    keys = [(r['path'],r['pattern'],r['acceleration'],r['method']) for r in rows]
    assert len(set(keys))==len(keys)
    calibration, groups = defaultdict(list), defaultdict(list)
    for row in rows:
        item = by_path[row['path']]
        for field in ('scan_id','sequence','role','motion_type'):
            assert row[field]==item[field]
        assert row['repetition']==int(item['repetition'])
        assert len(row['initial_columns'])==round(int(item['acquired_pe_lines'])/row['acceleration'])
        assert len(set(row['audit_columns']))==8
        assert not set(row['initial_columns']) & set(row['audit_columns'])
        assert len(row['slice_metrics'])==3 and [s['slice'] for s in row['slice_metrics']]==[7,8,9]
        for metric in ('audit_score','rss_nrmse','coil_nrmse','fused_rss_nrmse','fused_coil_nrmse','acquired_relative_residual'):
            expected = sum(s[metric] for s in row['slice_metrics'])/3
            assert np.isclose(row[metric],expected,rtol=1e-12,atol=1e-12),metric
            assert np.isfinite(row[metric]) and row[metric]>=0
        assert row['acquired_relative_residual'] < 1e-5
        assert row['fused_coil_nrmse'] <= row['coil_nrmse'] + 1e-6
        if row['method']=='zero_fill':
            assert abs(row['audit_score']-1)<1e-10
        key = (row['sequence'],row['pattern'],row['acceleration'],row['method'])
        if row['role']=='calibration':
            calibration[key].append(row)
    assert len(calibration)==36
    lookup = {tuple(c['condition']):c for c in summary['calibrations']}
    for key, values in calibration.items():
        assert len(values)==30 and len({r['scan_id'] for r in values})==30
        record = lookup[key]
        scores = [r['audit_score'] for r in values]
        assert sorted(scores)==sorted(record['scores'])
        assert record['threshold']==max(scores)  # ceil((30+1)*.95) is 30.
        assert record['high_reference_error_cutoff']==sorted(r['rss_nrmse'] for r in values)[27]
    for row in rows:
        if row['role']=='calibration':
            continue
        key = (row['sequence'],row['pattern'],row['acceleration'],row['method'])
        scores = [r['audit_score'] for r in calibration[key]]
        greater_or_tied = sum(x>=row['audit_score'] or abs(x-row['audit_score'])<=1e-12+1e-12*abs(row['audit_score']) for x in scores)
        expected_p = (1+greater_or_tied)/31
        assert row['p_value']==expected_p
        assert row['flagged']==(expected_p<=.05)
        assert row['high_reference_error']==(row['rss_nrmse']>lookup[key]['high_reference_error_cutoff'])
        assert row['selective_rss_nrmse']==(row['fused_rss_nrmse'] if row['flagged'] else row['rss_nrmse'])
        group = (row['role'],row['motion_type'],row['sequence'],row['repetition'],row['pattern'],row['acceleration'],row['method'])
        groups[group].append(row)
    assert len(groups)==90 and len(summary['groups'])==90
    for group in summary['groups']:
        key = tuple(group[k] for k in ('role','motion_type','sequence','repetition','pattern','acceleration','method'))
        values = groups[key]
        assert group['participants']==len(values)==len({r['scan_id'] for r in values})
        assert group['flagged']==sum(r['flagged'] for r in values)
        assert group['high_reference_error_cases']==sum(r['high_reference_error'] for r in values)
        assert group['unflagged_high_reference_error']==sum(r['high_reference_error'] and not r['flagged'] for r in values)
        assert np.isclose(group['mean_rss_nrmse'],sum(r['rss_nrmse'] for r in values)/len(values),rtol=1e-12)
        high = [r['audit_score'] for r in values if r['high_reference_error']]
        low = [r['audit_score'] for r in values if not r['high_reference_error']]
        auc = np.mean([float(a>b)+.5*float(a==b) for a in high for b in low]) if high and low else None
        assert group['reference_error_auroc']==auc or (auc is not None and np.isclose(group['reference_error_auroc'],auc,atol=1e-15))
    primary = summary['primary_rows']
    assert len(primary)==25 and len({r['scan_id'] for r in primary})==25
    expected_top = sorted((r for r in primary if not r['flagged']),key=lambda r:(-r['rss_nrmse'],r['scan_id']))[:5]
    assert [r['scan_id'] for r in summary['worst_unflagged_cases']]==[r['scan_id'] for r in expected_top]
    assert len(summary['paired_repeats'])==25
    report = {'status':'passed','rows_checked':len(rows),'participant_roles':{'train':20,'calibration':30,'test':25,'motion_risk':25},
        'acquisition_source_checks':len(checks),'calibration_strata':len(calibration),'reported_groups':len(groups),
        'manifest_sha256':digest(manifest_path),'finetuned_checkpoint_sha256':digest(checkpoint),
        'max_acquired_relative_residual':max(r['acquired_relative_residual'] for r in rows),
        'max_stored_source_rss_relative_error':max(r['rss_relative_check'] for r in checks),
        'zero_fill_scores_equal_one':True,'all_flags_and_high_error_labels_reconciled':True,
        'source_reference_caveat':'References contain noise and possible artifacts; clinical anatomy not validated'}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
