import copy
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from ksentinel.dataset import export_dataset, load_partition, make_shard, observed_kspace, validate_shard
from support import temporary_directory

ROOT=Path(__file__).resolve().parents[1]


class DataReliability(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=temporary_directory()
        cls.folder=Path(cls.temp.name)
        cls.manifest=export_dataset(cls.folder,seeds=(11,),counts=(2,2,2))

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def altered_manifest(self, mutate):
        temp=temporary_directory()
        folder=Path(temp.name)
        value=copy.deepcopy(self.manifest);mutate(value)
        (folder/"manifest.json").write_text(json.dumps(value))
        return temp,folder

    def test_ks021_valid_shards_keep_subject_split_membership(self):
        parts=[load_partition(self.folder,s,11) for s in ("train","calibration","test")]
        self.assertEqual([p["identities"].tolist() for p in parts],[[1100000,1100001],[1110000,1110001],[1120000,1120001]])
        self.assertEqual(parts[0]["truth"].shape,(2,64,64))

    def test_ks022_acquired_audit_overlap_is_rejected(self):
        data=make_shard(11,"test",1); data["audit_order"][0,0]=data["initial_columns"][0,0]
        with self.assertRaises(ValueError): validate_shard(data)

    def test_ks023_cross_split_identity_reuse_is_rejected(self):
        def change(m): m["files"][1]["identities"][0]=m["files"][0]["identities"][0]
        temp,folder=self.altered_manifest(change)
        with temp:
            with self.assertRaises(ValueError): load_partition(folder,"train",11)

    def test_ks024_declared_split_must_match_identity_range(self):
        data=make_shard(11,"train",1);data["split"]=np.array("test")
        with self.assertRaises(ValueError): validate_shard(data)

    def test_ks025_changed_shard_bytes_are_rejected(self):
        with temporary_directory() as folder:
            root=Path(folder);manifest=export_dataset(root,seeds=(11,),counts=(1,1,1))
            path=root/manifest["files"][0]["file"]
            path.write_bytes(path.read_bytes()+b"changed")
            with self.assertRaises(ValueError):load_partition(root,"train",11)

    def test_ks026_manifest_count_mismatch_is_rejected(self):
        temp,folder=self.altered_manifest(lambda m:m["files"][0].update(count=3))
        with temp:
            with self.assertRaises(ValueError):load_partition(folder,"train",11)

    def test_ks027_model_input_is_independent_of_truth_and_audit(self):
        data=make_shard(11,"test",1); expected,mask=observed_kspace(data,0)
        data["truth"][:]=np.nan;data["audit_values"][:]=np.nan;data["audit_order"][:]=-1
        actual,actual_mask=observed_kspace(data,0)
        np.testing.assert_array_equal(actual,expected);np.testing.assert_array_equal(actual_mask,mask)

    def test_ks028_export_matches_frozen_published_test_example(self):
        data=make_shard(11,"test",1);measured,mask=observed_kspace(data,0)
        with np.load(ROOT/"results/main/example.npz",allow_pickle=False) as saved:
            # libm/FFT implementations can differ by a few double-precision ulps.
            # Preserve physical values tightly; keep discrete acquisition exact.
            np.testing.assert_allclose(data["truth"][0],saved["truth"],rtol=1e-12,atol=1e-12)
            np.testing.assert_array_equal(mask,saved["initial_mask"])
            np.testing.assert_allclose(measured[:,mask],saved["measured"][:,mask],rtol=1e-12,atol=1e-12)
            np.testing.assert_array_equal(data["audit_order"][0],saved["audit_columns"])
