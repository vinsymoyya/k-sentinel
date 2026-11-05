"""Exploratory display of every unflagged high-error FLAIR/R4 test case."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ksentinel.model_io import load_model
from ksentinel.real_acquisition import (file_sha256,load_acquisition,reconstruct_coils,
                                        relative_error,rss,sampling,sampling_seed)
from download_m4raw import fetch_one


def main():
    folder=ROOT/'results/real-acquisition-0.3.0'
    out=ROOT/'assets/real-acquisition'
    out.mkdir(parents=True,exist_ok=True)
    config=json.loads((folder/'config.json').read_text(encoding='utf-8'))
    assert config['complete'] is True
    rows=[json.loads(line) for line in (folder/'records.jsonl').read_text(encoding='utf-8').splitlines()]
    cases=sorted([r for r in rows if r['role']=='test' and r['sequence']=='FLAIR' and r['repetition']==1
        and r['pattern']=='random' and r['acceleration']==4 and r['method']=='m4raw_finetuned_cnn'
        and r['high_reference_error'] and not r['flagged']],key=lambda r:r['scan_id'])
    if not cases:
        raise ValueError('No qualifying secondary cases; do not invent failure examples')
    manifest=json.loads((ROOT/'research/real-acquisition/manifest.json').read_text(encoding='utf-8'))
    lookup={r['relative_path']:r for r in manifest['files']}
    data=ROOT/'tmp/m4raw/contrast-figure-data'
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda r:fetch_one(data,manifest,lookup[r['path']]),cases))
    torch.set_num_threads(2)
    training=json.loads((folder/'training.json').read_text(encoding='utf-8'))
    frozen=load_model(ROOT/'results/main/cnn_seed11.pt',config['frozen_checkpoint_sha256'])
    model=load_model(folder/'m4raw_finetuned_seed20261105.pt',training['checkpoint_sha256'])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12})
    fig,axes=plt.subplots(len(cases),4,figsize=(12,3.6*len(cases)),squeeze=False)
    provenance=[]
    for index,row in enumerate(cases):
        item=lookup[row['path']]
        raw,reference,support,_=load_acquisition(data,item)
        mask,audit,acs=sampling(support,4,'random',sampling_seed(row['scan_id'],'FLAIR',4,'random'))
        prediction=rss(reconstruct_coils(model,raw[1],mask,acs))
        baseline=rss(reconstruct_coils(frozen,raw[1],mask,acs))
        local=relative_error(prediction,reference[1])
        hosted=row['slice_metrics'][1]['rss_nrmse']
        assert abs(local-hosted)<=1e-4
        maximum=float(np.quantile(reference[1],.99))
        displays=[('All-available source',reference[1]),('Frozen phantom CNN',baseline),
                  ('Fine-tuned CNN',prediction),('Absolute RSS difference',abs(prediction-reference[1]))]
        for column,(title,image) in enumerate(displays):
            ax=axes[index,column]
            ax.imshow(image,cmap='gray' if column<3 else 'magma',vmin=0,vmax=maximum if column<3 else maximum*.3)
            ax.set_xticks([]);ax.set_yticks([])
            if index==0:ax.set_title(title,fontsize=15,fontweight='bold')
            if column==0:ax.set_ylabel(f"{row['scan_id']}\nmean NRMSE {row['rss_nrmse']:.3f}\np={row['p_value']:.3f}, unflagged",fontsize=13)
        provenance.append({'scan_id':row['scan_id'],'source_sha256':item['sha256'],'slice':8,
            'mean_rss_nrmse':row['rss_nrmse'],'p_value':row['p_value'],'high_reference_error':True,
            'hosted_slice_rss_nrmse':hosted,'local_slice_rss_nrmse':local,'absolute_agreement_tolerance':1e-4})
    fig.suptitle('Real FLAIR: both high-reference-error cases remained unflagged',fontsize=17,fontweight='bold')
    fig.text(.02,.015,'Post-outcome exploratory illustration, random R=4. Source: M4Raw, Lyu et al. (2023), CC BY 4.0. Source reference contains noise/artifacts; no diagnostic labels.',fontsize=8)
    fig.tight_layout(rect=(0,.06,1,.94))
    for suffix in ('png','svg'):
        fig.savefig(out/f'flair_unflagged_high_error.{suffix}',dpi=180,facecolor='white')
    plt.close(fig)
    (out/'contrast_case_provenance.json').write_text(json.dumps({'selection':'All qualifying FLAIR/R4/random cases; post-outcome exploratory',
        'cases':provenance,'checkpoint_sha256':file_sha256(folder/'m4raw_finetuned_seed20261105.pt'),
        'numpy':np.__version__,'torch':str(torch.__version__)},indent=2)+'\n',encoding='utf-8')
    print(f'Displayed all {len(cases)} qualifying cases')


if __name__=='__main__':
    main()
