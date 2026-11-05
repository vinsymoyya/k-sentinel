"""Publication figures from complete real-MRI records and source-verified cases."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ksentinel.real_acquisition import (load_acquisition, reconstruct_coils, rss,
                                       sampling, sampling_seed, relative_error, file_sha256)
from ksentinel.model_io import load_model
from download_m4raw import fetch_one

COLORS = {'zero_fill':'#64748b', 'phantom_cnn':'#dd8428', 'm4raw_finetuned_cnn':'#007f86'}
LABELS = {'zero_fill':'Zero fill', 'phantom_cnn':'Frozen phantom CNN', 'm4raw_finetuned_cnn':'M4Raw fine-tuned CNN'}


def finish(fig, folder, name, footer):
    fig.text(.02, .015, footer, fontsize=8, color='#475569', va='bottom')
    fig.tight_layout(rect=(0,.06,1,.95))
    for suffix in ('png','svg'):
        fig.savefig(folder/f'{name}.{suffix}', dpi=180, facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', default='results/real-acquisition-0.3.0')
    parser.add_argument('--data', default='tmp/m4raw/data')
    parser.add_argument('--output', default='assets/real-acquisition')
    args = parser.parse_args()
    results, folder = Path(args.results), Path(args.output)
    folder.mkdir(parents=True,exist_ok=True)
    config = json.loads((results/'config.json').read_text(encoding='utf-8'))
    if config['complete'] is not True:
        raise ValueError('Only a complete frozen experiment can generate publication figures')
    summary = json.loads((results/'summary.json').read_text(encoding='utf-8'))
    rows = [json.loads(line) for line in (results/'records.jsonl').read_text(encoding='utf-8').splitlines()]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.titleweight':'bold'})
    groups = [g for g in summary['groups'] if g['role']=='test' and g['sequence']=='T2' and g['repetition']==1 and g['pattern']=='random']
    fig,axes = plt.subplots(1,2,figsize=(11,4.5))
    for method,color in COLORS.items():
        selected = sorted((g for g in groups if g['method']==method),key=lambda g:g['acceleration'])
        x = np.arange(len(selected))
        axes[0].plot(x,[g['mean_rss_nrmse'] for g in selected],'-o',color=color,label=LABELS[method])
        p = np.array([g['flagged']/g['participants'] for g in selected])
        bounds = np.array([g['flag_fraction_interval_95_wilson'] for g in selected]).T
        lengths=np.array([p-bounds[0],bounds[1]-p])
        if lengths.min() < -1e-12:
            raise ValueError('Reported interval does not contain its point estimate')
        axes[1].errorbar(x,p,yerr=np.maximum(0,lengths),fmt='-o',color=color,capsize=3,label=LABELS[method])
    for ax in axes:
        ax.set_xticks([0,1,2],['2','4','8'])
        ax.set_xlabel('Nominal retrospective acceleration (source support)')
        ax.grid(alpha=.15)
    axes[0].set(ylabel='Mean RSS reference NRMSE',title='Native reconstruction error')
    axes[1].set(ylabel='Audit flag fraction',title='Eight hidden audit columns')
    axes[1].set_ylim(-.025,1.025)
    axes[0].legend(fontsize=8)
    fig.suptitle('Real T2 acquisitions: 25 held-out participants, repetition 1',fontsize=13,fontweight='bold')
    finish(fig,folder,'sampling_and_audit','Three fixed slices per participant. Wilson intervals conditional on fitted model/calibration. Source reference is noisy, not anatomical ground truth.')

    primary = summary['primary_rows']
    repeats = summary['paired_repeats']
    fig,axes = plt.subplots(1,2,figsize=(11,4.5))
    for row in repeats:
        axes[0].plot([1,2],[row['rss_nrmse_rep1'],row['rss_nrmse_rep2']],color='#9ca3af',alpha=.6,lw=1)
    axes[0].plot([1,2],[np.mean([r['rss_nrmse_rep1'] for r in repeats]),np.mean([r['rss_nrmse_rep2'] for r in repeats])],'-o',color=COLORS['m4raw_finetuned_cnn'],lw=3,label='Mean')
    axes[0].set(xticks=[1,2],xticklabels=['Actual repeat 1','Actual repeat 2'],ylabel='RSS reference NRMSE',title='Paired real repeat acquisitions (n=25)')
    cohorts = [
        ('T2 test\nn=25',lambda r:r['role']=='test' and r['sequence']=='T2' and r['repetition']==1),
        ('FLAIR test\nn=25',lambda r:r['role']=='test' and r['sequence']=='FLAIR'),
        ('Inter-scan risk\nn=8',lambda r:r['role']=='motion_risk' and r['motion_type']=='inter_scan'),
        ('Intra-scan risk\nn=17',lambda r:r['role']=='motion_risk' and r['motion_type']=='intra_scan')]
    selected = [r for r in rows if r['method']=='m4raw_finetuned_cnn' and r['pattern']=='random' and r['acceleration']==4]
    arrays = [[r['rss_nrmse'] for r in selected if predicate(r)] for _,predicate in cohorts]
    axes[1].boxplot(arrays,tick_labels=[name for name,_ in cohorts],showfliers=False)
    rng = np.random.default_rng(20261105)
    for i, values in enumerate(arrays,1):
        axes[1].scatter(i+rng.uniform(-.08,.08,len(values)),values,s=12,color=COLORS['m4raw_finetuned_cnn'],alpha=.7)
    axes[1].set(ylabel='RSS reference NRMSE',title='Contrast and participant-level motion-risk groups')
    axes[1].tick_params(axis='x',labelsize=8)
    fig.suptitle('Actual acquisition changes: fine-tuned CNN, random R=4',fontsize=13,fontweight='bold')
    finish(fig,folder,'actual_acquisition_variations','Motion-risk membership is participant-level, not an artifact label for every scan. One scanner / 0.3 T / four coils; no cross-vendor claim.')

    fig,axes = plt.subplots(1,2,figsize=(11,4.5))
    for ax,method in zip(axes,['phantom_cnn','m4raw_finetuned_cnn']):
        selected = [r for r in rows if r['role']=='test' and r['sequence']=='T2' and r['repetition']==1 and r['pattern']=='random' and r['acceleration']==4 and r['method']==method]
        calibration = next(c for c in summary['calibrations'] if c['condition']==['T2','random',4,method])
        for flagged,label,color in [(False,'Unflagged','#007f86'),(True,'Flagged','#d06725')]:
            points = [r for r in selected if r['flagged']==flagged]
            if points:
                ax.scatter([r['rss_nrmse'] for r in points],[r['audit_score'] for r in points],color=color,label=label,s=35,alpha=.8)
        ax.axvline(calibration['high_reference_error_cutoff'],color='#475569',ls='--',lw=1,label='Calibrated high-error cutoff')
        ax.axhline(calibration['threshold'],color='#b45309',ls=':',lw=1,label='Calibration audit threshold')
        ax.set(xlabel='RSS reference NRMSE',ylabel='Relative complex audit residual',title=LABELS[method])
        ax.legend(fontsize=7)
    fig.suptitle('Does audit disagreement track natural reference error?',fontsize=13,fontweight='bold')
    finish(fig,folder,'audit_vs_natural_error','25 participant-level points per model. High-error cutoff from separate calibration participants; no clinical or focal-feature labels are available.')

    manifest = json.loads((ROOT/'research/real-acquisition/manifest.json').read_text(encoding='utf-8'))
    by_path = {r['relative_path']:r for r in manifest['files']}
    cases = (summary['worst_unflagged_cases'] or summary['highest_error_cases'])[:3]
    pending = [by_path[case['path']] for case in cases
               if not (Path(args.data)/case['path']).exists()]
    if pending:
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(lambda item:fetch_one(ROOT/'tmp/m4raw/figure-data',manifest,item),pending))
    torch.set_num_threads(2)
    models = {'zero_fill':None, 'phantom_cnn':load_model(ROOT/'results/main/cnn_seed11.pt',config['frozen_checkpoint_sha256']),
        'm4raw_finetuned_cnn':load_model(results/'m4raw_finetuned_seed20261105.pt',json.loads((results/'training.json').read_text(encoding='utf-8'))['checkpoint_sha256'])}
    fig,axes = plt.subplots(len(cases),5,figsize=(13,3.1*len(cases)),squeeze=False)
    displayed = []
    for index,case in enumerate(cases):
        item = by_path[case['path']]
        data_root = Path(args.data)
        if not (data_root/item['relative_path']).exists():
            data_root = ROOT/'tmp/m4raw/figure-data'
            fetch_one(data_root,manifest,item)
        raw, target, support, _ = load_acquisition(data_root,item)
        mask,audit,acs = sampling(support,4,'random',sampling_seed(item['scan_id'],'T2',4,'random'))
        images = {name:rss(reconstruct_coils(model,raw[1],mask,acs)) for name,model in models.items()}
        local_error = relative_error(images['m4raw_finetuned_cnn'],target[1])
        hosted_error = case['slice_metrics'][1]['rss_nrmse']
        if abs(local_error-hosted_error)>1e-4:
            raise ValueError('Local displayed reconstruction disagrees with the hosted numerical record')
        displayed.append({'scan_id':item['scan_id'],'source_sha256':item['sha256'],'slice':8,
                          'hosted_slice_rss_nrmse':hosted_error,'local_slice_rss_nrmse':local_error,
                          'absolute_agreement_tolerance':1e-4})
        maximum = float(np.quantile(target[1],.99))
        displays = [('All-available\nsource',target[1]),('Zero fill',images['zero_fill']),('Frozen phantom\nCNN',images['phantom_cnn']),('Fine-tuned\nCNN',images['m4raw_finetuned_cnn']),('Absolute RSS\ndifference',abs(images['m4raw_finetuned_cnn']-target[1]))]
        for column,(title,image) in enumerate(displays):
            ax = axes[index,column]
            ax.imshow(image,cmap='gray' if column<4 else 'magma',vmin=0,vmax=maximum if column<4 else maximum*.3)
            ax.set_xticks([]);ax.set_yticks([])
            if index==0:ax.set_title(title,fontsize=15)
            if column==0:ax.set_ylabel(f"{item['scan_id']}\nmean NRMSE {case['rss_nrmse']:.3f}\naudit {'flagged' if case['flagged'] else 'unflagged'}",fontsize=13)
        # The displayed slice is fixed (8); ranking used the predeclared mean across 7/8/9.
    fig.suptitle('Predeclared worst unflagged native reconstruction cases' if summary['worst_unflagged_cases'] else 'Predeclared highest-error native reconstruction cases',fontsize=18,fontweight='bold')
    finish(fig,folder,'natural_failure_cases','Source MRI: M4Raw, Lyu et al. (2023), CC BY 4.0. Display-only reference windowing. No inserted structures; source reference may contain noise or motion.')
    (folder/'case_provenance.json').write_text(json.dumps({'cases':displayed,
        'checkpoint_sha256':file_sha256(results/'m4raw_finetuned_seed20261105.pt'),
        'numpy':np.__version__,'torch':str(torch.__version__)},indent=2)+'\n',encoding='utf-8')
    print('Wrote four PNG/SVG figure pairs',flush=True)


if __name__=='__main__':
    main()
