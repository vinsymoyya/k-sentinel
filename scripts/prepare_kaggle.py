"""Prepare public Kaggle data, model and executable notebook without credentials."""
import argparse
import json
import shutil
import subprocess
import zipfile
from pathlib import Path
from ksentinel.dataset import export_dataset

ROOT=Path(__file__).resolve().parents[1]
OWNER="vinsymoyya"
DATASET=f"{OWNER}/k-sentinel-synthetic-mri-3840"
MODEL=f"{OWNER}/k-sentinel"
MODEL_SOURCE=f"{MODEL}/pyTorch/residual-cnn/3"
NOTEBOOK=f"{OWNER}/k-sentinel-45-checks-and-open-training"


def notebook_document():
    cells=[]
    def markdown(text): cells.append({"cell_type":"markdown","metadata":{},"source":text.splitlines(keepends=True)})
    def code(text): cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":text.splitlines(keepends=True)})
    markdown("""# K-Sentinel: 45 checks and open training

Train an open residual CNN on split-preserving synthetic complex MRI data, reproduce the released failure mechanism, and inspect the evidence. **Synthetic research only.** Forty-five passing software/scientific contract checks do not establish clinical safety, natural pathology detection or state-of-the-art performance.

Author: [Vinsy Moyya](https://orcid.org/0009-0003-0042-3556). [Code and paper](https://github.com/vinsymoyya/k-sentinel). Code/model: MIT. Generated data: CC0-1.0.

This notebook runs on CPU with internet disabled. Add the public K-Sentinel dataset and model variation if you fork it. Every original training, calibration and test identity is preserved. The original unreviewed preprint remains a v0.1.0 experimental record; this notebook is the v0.2.0 reproducibility workflow.
""")
    code("""from pathlib import Path
import sys, os, zipfile, json, subprocess, shutil
destination = Path('/kaggle/working')
root = destination / 'k-sentinel-0.2.0'
archives = list(Path('/kaggle/input').rglob('k-sentinel-0.2.0-source.zip'))
if len(archives) == 1:
    with zipfile.ZipFile(archives[0]) as bundle:
        for item in bundle.namelist():
            if not (destination / item).resolve().is_relative_to(destination.resolve()):
                raise ValueError('Archive member escapes working directory')
        bundle.extractall(destination)
else:
    # Kaggle expands uploaded model ZIP files into directories.
    sources = [p.parents[1] for p in Path('/kaggle/input').rglob('run_validation.py')
               if p.parent.name == 'scripts' and (p.parents[1] / 'ksentinel').is_dir()]
    if len(sources) != 1:
        raise ValueError('Attach exactly one K-Sentinel model source bundle')
    shutil.copytree(sources[0], root, dirs_exist_ok=True)
os.chdir(root)
sys.path.insert(0, str(root))
import numpy as np, torch
print({'numpy': np.__version__, 'torch': str(torch.__version__), 'source': str(root)})
""")
    markdown("## 1. Run every validation case\n\nNo skips are counted as passes. The runner refuses a suite other than the documented 45 cases and writes JSON plus JUnit evidence.")
    code("""subprocess.run([sys.executable, 'scripts/run_validation.py', '--output', '/kaggle/working/validation'], check=True)
validation = json.loads(Path('/kaggle/working/validation/report.json').read_text())
assert validation['passed'] == 45 and validation['failed'] == 0 and validation['skipped'] == 0
print({key: validation[key] for key in ('total', 'passed', 'failed', 'skipped')})
""")
    markdown("## 2. Inspect the downloadable dataset\n\nEach shard stores complex truth, initially acquired columns/values, independent audit columns/values, identities and noise scale. Keep audit values out of reconstruction. Ground truth is used only as a training target or for final evaluation.")
    code("""from ksentinel.dataset import load_partition
candidates = [p for p in Path('/kaggle/input').rglob('manifest.json') if (p.parent / 'seed11-train.npz').exists()]
if len(candidates) != 1:
    raise ValueError('Attach exactly one K-Sentinel synthetic MRI dataset')
data_dir = candidates[0].parent
manifest = json.loads(candidates[0].read_text())
assert manifest['total_objects'] == 3840
print({'objects': manifest['total_objects'], 'shards': len(manifest['files']), 'license': manifest['license']})
sample = load_partition(data_dir, 'test', 11, limit=1)
import matplotlib.pyplot as plt
plt.figure(figsize=(4, 4))
plt.imshow(abs(sample['truth'][0]), cmap='gray', vmin=0, vmax=1)
plt.title('Synthetic test phantom; no patient data')
plt.axis('off'); plt.show()
""")
    markdown("## 3. Train with fixed settings, then evaluate\n\nEight epochs, Adam at 0.001, batch size 32, CPU. Three seeds. Calibration and test subjects never train the network; test scores never select a checkpoint. Set QUICK=True only for a small smoke run; it does not reproduce the full experiment.")
    code("""from ksentinel.training import train_dataset
QUICK = False
reports = []
for seed in ([11] if QUICK else [11, 29, 47]):
    reports.append(train_dataset(data_dir, '/kaggle/working/training', seed=seed,
                                 epochs=1 if QUICK else 8, limit=16 if QUICK else None))
print([{key: r[key] for key in ('test_count', 'mean_cnn_nrmse', 'mean_zero_fill_nrmse')} for r in reports])
""")
    markdown("## 4. Compare reconstruction error\n\nLower complex NRMSE is better. These are results for the current hosted run, not clinical endpoints. Inspect natural reconstruction error before interpreting an added-error audit.")
    code("""fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for report in reports:
    axes[0].plot(range(1, len(report['training_mse']) + 1), report['training_mse'], label=f\"seed {report['config']['seed']}\")
axes[0].set(xlabel='Epoch', ylabel='Training channel MSE', title='Training history')
axes[0].legend()
positions = np.arange(len(reports))
axes[1].bar(positions - .17, [r['mean_zero_fill_nrmse'] for r in reports], width=.34, label='Zero fill')
axes[1].bar(positions + .17, [r['mean_cnn_nrmse'] for r in reports], width=.34, label='CNN')
axes[1].set(xticks=positions, xticklabels=[str(r['config']['seed']) for r in reports], xlabel='Seed', ylabel='Mean test NRMSE', title='Held-out synthetic test subjects')
axes[1].legend(); plt.tight_layout(); plt.show()
""")
    markdown("## 5. Load and use the open checkpoint\n\nOnly the initial measured coefficients enter reconstruction. Values from the unacquired grid are ignored. The audit remains a separate check and a negative flag does not certify correct anatomy.")
    code("""from ksentinel.model_io import load_model, reconstruct
from ksentinel.dataset import observed_kspace
checkpoint = Path('/kaggle/working/training/cnn_seed11.pt')
model = load_model(checkpoint, expected_sha256=reports[0]['checkpoint_sha256'])
measured, initial_mask = observed_kspace(sample, 0)
candidate = reconstruct(model, measured, initial_mask)
plt.figure(figsize=(8, 3))
plt.subplot(1, 2, 1); plt.imshow(abs(sample['truth'][0]), cmap='gray', vmin=0, vmax=1); plt.title('Synthetic truth'); plt.axis('off')
plt.subplot(1, 2, 2); plt.imshow(abs(candidate), cmap='gray', vmin=0, vmax=1); plt.title('Trained CNN reconstruction'); plt.axis('off')
plt.tight_layout(); plt.show()
Path('/kaggle/working/run-summary.json').write_text(json.dumps({'quick': QUICK, 'validation': {'passed': validation['passed'], 'total': validation['total']}, 'training_reports': [str(p) for p in Path('/kaggle/working/training').glob('*.json')], 'scope': 'Synthetic methods workflow; no clinical validation'}, indent=2))
""")
    markdown("## Reuse and contribution\n\nDownload the generated checkpoints and JSON/JUnit records from notebook output. Fork this notebook for independent reproduction; preserve split identities and report failures. The attached model contains the original open weights and a source archive. The Kaggle model page documents licensing and intended use. Downloads, forks and votes reflect actual community use; no growth outcome is promised.")
    return {"nbformat":4,"nbformat_minor":5,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.11"}},"cells":cells}


def write_notebook(folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    metadata={"id":NOTEBOOK,"title":"K-Sentinel 45 Checks and Open Training","code_file":"k-sentinel-training.ipynb",
              "language":"python","kernel_type":"notebook","is_private":False,"enable_gpu":False,
              "enable_internet":False,"dataset_sources":[DATASET],"model_sources":[MODEL_SOURCE],
              "competition_sources":[],"kernel_sources":[]}
    (folder/"kernel-metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    (folder/metadata["code_file"]).write_text(json.dumps(notebook_document(),indent=2)+"\n")
    return metadata


def prepare(folder, dataset=False, bundle=False):
    stage=Path(folder)
    for name in ("data","model","variation","notebook"):(stage/name).mkdir(parents=True,exist_ok=True)
    title="K-Sentinel Synthetic MRI: 3840 Complex Phantoms"
    description="""# K-Sentinel synthetic MRI benchmark

3,840 independent procedurally generated complex 64x64 objects: 1,536 training, 768 calibration and 1,536 test subjects across seeds 11, 29 and 47. Nine NPZ shards preserve the exact published subject partitions and raw-measurement generation. Includes truth, initial measurement columns/values, withheld audit columns/values and a SHA256 manifest. This is synthetic single-coil Cartesian MRI, not patient data or biological lesions.

Use the paired open model and executable training notebook for a CPU workflow with 45 validation cases. Model input must exclude audit and unacquired coefficients. Code/model: MIT; this generated dataset: CC0-1.0. Paper, limitations and full results: https://github.com/vinsymoyya/k-sentinel. Author: Vinsy Moyya, ORCID 0009-0003-0042-3556.

Passing software tests does not establish clinical efficacy. The published CNN audit detected only 12.4% of controlled insertions at four audit columns, with 5.6% clean rejection. Preserve negative results and do not describe this dataset as clinical ground truth.
"""
    meta={"id":DATASET,"title":title,"subtitle":"Split-preserving raw measurements, independent audits and open training",
          "description":description,"licenses":[{"name":"CC0-1.0"}],"keywords":["health","artificial intelligence","computer science"],
          "resources":[{"path":f"seed{s}-{split}.npz","description":f"Seed {s}; {split} subjects; complex128 raw synthetic measurements"} for s in (11,29,47) for split in ("train","calibration","test")]}
    (stage/"data/dataset-metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
    (stage/"data/README.md").write_text(description)
    (stage/"data/LICENSE-DATA.txt").write_text("Generated synthetic data dedicated to the public domain under CC0-1.0.\nhttps://creativecommons.org/publicdomain/zero/1.0/legalcode\nNo patient data or third-party images are included.\n")
    model={"ownerSlug":OWNER,"slug":"k-sentinel","title":"K-Sentinel: Open MRI Reconstruction Audit Baseline",
           "subtitle":"Three small synthetic MRI CNNs with reproducible checks and open training",
           "isPrivate":False,"description":(ROOT/"MODEL_CARD.md").read_text()+"\n\n# Open workflow\n\nSoftware release 0.2.0 supplies 45 executable validation cases, default-safe loading and split-preserving Kaggle training. This count is software evidence, not a clinical reliability rating. Original weights and v0.1.0 experimental claims are unchanged.\n\nhttps://github.com/vinsymoyya/k-sentinel\n",
           "provenanceSources":"Original training and synthetic data generator: https://github.com/vinsymoyya/k-sentinel"}
    (stage/"model/model-metadata.json").write_text(json.dumps(model,indent=2)+"\n")
    instance={"ownerSlug":OWNER,"modelSlug":"k-sentinel","instanceSlug":"residual-cnn","framework":"pyTorch",
              "overview":"Three default-safe residual CNN checkpoints trained only on synthetic MRI phantoms. Native reconstruction baseline; no clinical validation.",
              "usage":"# Load\n\nExtract k-sentinel-0.2.0-source.zip and run the README quickstart.\n\n```python\nfrom ksentinel.model_io import load_model, reconstruct\nmodel = load_model('cnn_seed11.pt')\nimage = reconstruct(model, measured_kspace, initial_mask)\n```\n\nKeep audit values and test subjects out of training. Complete CPU training and 45-check notebook are linked from the repository.\n",
              "licenseName":"MIT","fineTunable":True,"trainingData":[f"https://www.kaggle.com/datasets/{DATASET}"],"modelInstanceType":"Unspecified"}
    (stage/"variation/model-instance-metadata.json").write_text(json.dumps(instance,indent=2)+"\n")
    write_notebook(ROOT/"kaggle/notebook")
    shutil.copytree(ROOT/"kaggle/notebook",stage/"notebook",dirs_exist_ok=True)
    if dataset:
        manifest=export_dataset(stage/"data")
        print(f"Exported {manifest['total_objects']} synthetic subjects",flush=True)
    if bundle:
        for path in (ROOT/"results/main").glob("cnn_seed*.pt"):shutil.copy2(path,stage/"variation"/path.name)
        shutil.copy2(ROOT/"MODEL_CARD.md",stage/"variation/MODEL_CARD.md")
        shutil.copy2(ROOT/"LICENSE",stage/"variation/LICENSE")
        files=subprocess.check_output(["git","ls-files","--cached","--others","--exclude-standard"],cwd=ROOT,text=True).splitlines()
        archive=stage/"variation/k-sentinel-0.2.0-source.zip"
        with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED) as z:
            for name in files:
                path=ROOT/name
                if path.is_file():z.write(path,"k-sentinel-0.2.0/"+name.replace("\\","/"))
        print(f"Prepared code archive and three open checkpoints: {archive}",flush=True)
    return stage


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",default="dist/kaggle")
    parser.add_argument("--dataset",action="store_true")
    parser.add_argument("--bundle",action="store_true")
    args=parser.parse_args()
    prepare(args.output,args.dataset,args.bundle)
