"""Exploratory frozen-T2-threshold transfer; never alter the primary record."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', default='results/real-acquisition-0.3.0')
    args = parser.parse_args()
    folder = Path(args.results)
    config = json.loads((folder/'config.json').read_text(encoding='utf-8'))
    if config['complete'] is not True:
        raise ValueError('A complete primary record is required')
    rows = [json.loads(line) for line in (folder/'records.jsonl').read_text(encoding='utf-8').splitlines()]
    fixed = {}
    for method in ('zero_fill','phantom_cnn','m4raw_finetuned_cnn'):
        calibration = [r for r in rows if r['role']=='calibration' and r['sequence']=='T2'
                       and r['pattern']=='random' and r['acceleration']==4 and r['method']==method]
        if len(calibration)!=30 or len({r['scan_id'] for r in calibration})!=30:
            raise ValueError('Thirty distinct fixed calibration participants required')
        fixed[method] = [r['audit_score'] for r in calibration]
    groups = defaultdict(list)
    for row in rows:
        if row['role']=='calibration':
            continue
        greater = sum(x>=row['audit_score'] or abs(x-row['audit_score'])<=1e-12+1e-12*abs(row['audit_score'])
                      for x in fixed[row['method']])
        p = (1+greater)/31
        key = tuple(row[k] for k in ('role','motion_type','sequence','repetition','pattern','acceleration','method'))
        groups[key].append((row,p))
    output = []
    for key, cases in sorted(groups.items()):
        if len(cases)!=len({r['scan_id'] for r,_ in cases}):
            raise ValueError('Repeated participant within analysis group')
        output.append({**dict(zip(('role','motion_type','sequence','repetition','pattern','acceleration','method'),key)),
            'participants':len(cases),'condition_matched_flags':sum(r['flagged'] for r,_ in cases),
            'frozen_T2_R4_flags':sum(p<=.05 for _,p in cases),
            'mean_rss_reference_nrmse':float(np.mean([r['rss_nrmse'] for r,_ in cases]))})
    result = {'analysis_status':'Additional exploratory analysis; primary protocol and record unchanged',
        'fixed_calibration':'T2, random R=4, original thirty calibration participants for each model',
        'records_sha256':hashlib.sha256((folder/'records.jsonl').read_bytes()).hexdigest(),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'no_coverage_guarantee_under_shift':True,'groups':output}
    (folder/'exploratory_transfer.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'groups':len(output),'status':'complete'},indent=2))


if __name__=='__main__':
    main()
