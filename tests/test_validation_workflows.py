import ast
import csv
import gzip
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
import numpy as np
from ksentinel.dataset import export_dataset
from ksentinel.model_io import load_model
from support import temporary_directory

ROOT=Path(__file__).resolve().parents[1]


class WorkflowReliability(unittest.TestCase):
    def command(self,*args):
        env=os.environ.copy();env["PYTHONPATH"]=str(ROOT)
        return subprocess.run([sys.executable,*args],cwd=ROOT,env=env,capture_output=True,text=True,timeout=180)

    def test_ks039_external_audit_cli_emits_strict_json(self):
        with temporary_directory() as folder:
            path=Path(folder)/"input.npz"
            np.savez(path,candidate=np.ones((8,8),complex),columns=np.array([4]),audit_values=np.zeros((8,1),complex),calibration_scores=np.ones(99),sigma=1.)
            run=self.command("-m","ksentinel.audit",str(path));self.assertEqual(run.returncode,0,run.stderr)
            result=json.loads(run.stdout,parse_constant=lambda value: self.fail("Nonfinite JSON value"))
            self.assertTrue(result["flagged"]);self.assertEqual(result["p_value"],0.01)

    def test_ks040_cli_does_not_unpickle_object_arrays(self):
        with temporary_directory() as folder:
            path=Path(folder)/"object.npz"
            np.savez(path,candidate=np.array([{"unexpected":"object"}],dtype=object))
            run=self.command("-m","ksentinel.audit",str(path))
            self.assertNotEqual(run.returncode,0)
            self.assertIn("allow_pickle=False",run.stderr)

    def test_ks041_small_benchmark_produces_consistent_portable_records(self):
        with temporary_directory() as folder:
            run=self.command("-m","ksentinel.benchmark","--no-cnn","--seeds","7","--train","2","--calibration","8","--test","2","--epochs","1","--output",folder)
            self.assertEqual(run.returncode,0,run.stderr)
            for path in Path(folder).glob("*.json"):
                json.loads(path.read_text(),parse_constant=lambda value:self.fail("Nonfinite JSON value"))
            summary=json.loads((Path(folder)/"summary.json").read_text())
            with gzip.open(Path(folder)/"subjects.csv.gz","rt") as stream: rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),112);self.assertEqual(len(summary["groups"]),56)
            self.assertTrue(all(int(r["flagged"])==0 for r in rows))

    def test_ks042_invalid_experiment_selectors_fail_before_writing(self):
        with temporary_directory() as folder:
            for arguments in (("--seeds","1","1"),("--train","10001")):
                target=Path(folder)/"never-created"
                run=self.command("-m","ksentinel.benchmark",*arguments,"--output",str(target))
                self.assertNotEqual(run.returncode,0);self.assertFalse(target.exists())

    def test_ks043_frozen_publication_evidence_reaggregates(self):
        run=self.command("scripts/verify_results.py","--check-only")
        self.assertEqual(run.returncode,0,run.stderr)
        report=json.loads(run.stdout)
        self.assertEqual(report["rows_checked"],129024);self.assertEqual(report["groups_checked"],84)

    def test_ks044_public_notebook_is_parseable_offline_and_connected(self):
        spec=importlib.util.spec_from_file_location("kaggle_preparation",ROOT/"scripts/prepare_kaggle.py")
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with temporary_directory() as folder:
            metadata=module.write_notebook(folder)
            self.assertFalse(metadata["is_private"]);self.assertFalse(metadata["enable_internet"])
            self.assertFalse(metadata["enable_gpu"])
            self.assertEqual(metadata["dataset_sources"],["vinsymoyya/k-sentinel-synthetic-mri-3840"])
            self.assertEqual(len(metadata["model_sources"]),1)
            model_path,version=metadata["model_sources"][0].rsplit("/",1)
            self.assertEqual(model_path,"vinsymoyya/k-sentinel/pyTorch/residual-cnn")
            self.assertTrue(version.isdigit() and int(version)>=1)
            notebook=json.loads((Path(folder)/metadata["code_file"]).read_text())
            code=["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"]=="code"]
            for cell in code:ast.parse(cell)
            self.assertTrue(any("scripts/run_validation.py" in cell for cell in code))
            self.assertTrue(any("train_dataset" in cell for cell in code))

    def test_ks045_downloaded_data_training_cli_is_end_to_end(self):
        with temporary_directory() as folder:
            folder=Path(folder);data=folder/"data";output=folder/"training"
            export_dataset(data,seeds=(11,),counts=(8,24,4))
            run=self.command("-m","ksentinel.training","--data",str(data),"--output",str(output),"--seed","11","--epochs","1")
            self.assertEqual(run.returncode,0,run.stderr)
            report=json.loads((output/"training_seed11.json").read_text())
            self.assertEqual(report["test_count"],4)
            self.assertEqual(len(report["training_mse"]),1)
            self.assertTrue(np.isfinite(report["mean_cnn_nrmse"]))
            ids=report["config"]["identities"]
            self.assertFalse(set(ids["train"]) & (set(ids["calibration"])|set(ids["test"])))
            self.assertFalse(load_model(output/"cnn_seed11.pt",report["checkpoint_sha256"]).training)
