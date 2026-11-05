"""Package the independent real-acquisition experiment, without MRI payloads."""
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT/'dist/real-kaggle'


def main():
    variation = STAGE/'variation'
    notebook = STAGE/'notebook'
    variation.mkdir(parents=True, exist_ok=True)
    notebook.mkdir(parents=True, exist_ok=True)
    source = variation/'k-sentinel-0.3.0-source.bundle'
    for name in ('11','29','47'):
        shutil.copy2(ROOT/f'results/main/cnn_seed{name}.pt',variation/f'cnn_seed{name}.pt')
    shutil.copy2(ROOT/'LICENSE',variation/'LICENSE')
    shutil.copy2(ROOT/'MODEL_CARD.md',variation/'MODEL_CARD.md')
    metadata = {'ownerSlug':'vinsymoyya','modelSlug':'k-sentinel','instanceSlug':'residual-cnn',
        'framework':'pyTorch','licenseName':'MIT','fineTunable':True,
        'overview':'Original open phantom CNN weights with separate real-acquisition experiment source. Original weights unchanged.',
        'usage':'Use the real-acquisition notebook to fine-tune on its participant-disjoint M4Raw training selection. The attached weights remain the original phantom baseline.'}
    (variation/'model-instance-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    cells=[]
    def md(text):cells.append({'cell_type':'markdown','metadata':{},'source':text.splitlines(keepends=True)})
    def code(text):cells.append({'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':text.splitlines(keepends=True)})
    md('''# K-Sentinel: real MRI acquisition variations and natural failures

Author: Vinsy Moyya, ORCID 0009-0003-0042-3556. An exploratory extension of [K-Sentinel](https://github.com/vinsymoyya/k-sentinel).

Uses original-creators' M4Raw MRI acquisitions (Lyu et al., Scientific Data 10:264, DOI 10.1038/s41597-023-02181-4), licensed CC BY 4.0, pinned source revision. Code MIT. The older synthetic dataset remains CC0; M4Raw is separately attributed and licensed.

100 disjoint participants; 180 source acquisitions; actual repeated scans and T2/FLAIR contrast; participant-level motion-risk stress cohort. Retrospective masks are simulated acquisition changes. No artificial structures, synthetic motion or synthetic noise are added. All-available source references contain noise and possible motion. This experiment does not validate clinical anatomy, diagnosis or prospective scanning benefit.
''')
    code('''from pathlib import Path
import os, sys, zipfile, subprocess, json
archive = next(Path('/kaggle/input').rglob('k-sentinel-0.3.0-source.bundle'))
destination = Path('/kaggle/working')
with zipfile.ZipFile(archive) as z:
    for member in z.namelist():
        if not (destination/member).resolve().is_relative_to(destination.resolve()):
            raise ValueError('Unsafe source member')
    z.extractall(destination)
root = destination/'k-sentinel-0.3.0'
os.chdir(root); sys.path.insert(0,str(root))
import numpy as np, torch, h5py, requests
print({'numpy':np.__version__,'torch':str(torch.__version__),'h5py':h5py.__version__})
''')
    md('## Validate both workflows\nThe existing 45 cases remain separate from the new real-acquisition contracts. No skips count as passes.')
    code('''subprocess.run([sys.executable,'scripts/run_validation.py','--output','/kaggle/working/validation/legacy'],check=True)
subprocess.run([sys.executable,'scripts/run_real_validation.py','--output','/kaggle/working/validation/real'],check=True)
''')
    md('## Fetch the frozen, byte-verified real acquisition selection\nAbout 2.22 GB, 180 files. Every source SHA256 is checked. Internet is required for this step. All selection and settings are recorded in research/real-acquisition/PROTOCOL.md before model evaluation.')
    code('''subprocess.run([sys.executable,'scripts/download_m4raw.py','--output','/kaggle/temp/m4raw','--workers','4'],check=True)
''')
    md('## Train and evaluate\nEight fixed fine-tuning epochs on twenty training participants. Thirty separate calibration participants. Twenty-five expanded-test participants and twenty-five motion-risk participants. Three fixed native slices per acquisition, all four coils. Test and calibration data never select weights.')
    code('''subprocess.run([sys.executable,'scripts/run_real_acquisition.py','--data','/kaggle/temp/m4raw','--output','/kaggle/working/real-results'],check=True)
''')
    md('## Inspect every outcome\nA negative audit is not an anatomical certificate. Motion-subset membership is not a file-level artifact label. Natural failures here are reference-reconstruction errors; no disease or focal-lesion labels are supplied.')
    code('''report = json.loads(Path('/kaggle/working/real-results/summary.json').read_text())
primary = [g for g in report['groups'] if g['role']=='test' and g['sequence']=='T2' and g['repetition']==1 and g['pattern']=='random' and g['acceleration']==4]
for group in primary:
    print({key:group[key] for key in ('method','participants','mean_rss_nrmse','flagged','high_reference_error_cases','unflagged_high_reference_error')})
print('Paired real repetitions:',report['paired_repetition_error_change'])
print('Worst unflagged cases:',[(r['scan_id'],r['rss_nrmse']) for r in report['worst_unflagged_cases']])
''')
    md('## Reproduce and review\nDownload all result rows, source checks, checkpoint and configuration from output. Preserve the participant groups and original dataset attribution. Review statistical calibration, source-reference limitations, native failure cases and actual versus retrospective acquisition changes. This is a single-scanner, healthy-volunteer exploratory study.')
    doc={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':cells}
    (notebook/'real-acquisition.ipynb').write_text(json.dumps(doc,indent=2)+'\n',encoding='utf-8')
    meta={'id':'vinsymoyya/k-sentinel-real-mri-variations-and-failures','title':'K-Sentinel Real MRI Variations and Failures',
        'code_file':'real-acquisition.ipynb','language':'python','kernel_type':'notebook','is_private':False,
        'enable_gpu':False,'enable_internet':True,'dataset_sources':[],
        'model_sources':['vinsymoyya/k-sentinel/pyTorch/residual-cnn/6'],'competition_sources':[],'kernel_sources':[]}
    (notebook/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
    shutil.copytree(notebook, ROOT/'kaggle/real-acquisition', dirs_exist_ok=True)
    files = subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard'],cwd=ROOT,text=True).splitlines()
    with zipfile.ZipFile(source,'w',zipfile.ZIP_DEFLATED) as z:
        for name in files:
            path = ROOT/name
            if path.is_file():
                z.write(path,'k-sentinel-0.3.0/'+name.replace('\\','/'))
    print(source)


if __name__ == '__main__':
    main()
