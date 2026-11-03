import hashlib
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from ksentinel.core import acquisition, noisy_kspace, phantom, fft
from ksentinel.dataset import make_shard
from ksentinel.learned import ResidualCNN
from ksentinel.model_io import load_model,reconstruct
from ksentinel.training import fit
from support import temporary_directory

ROOT=Path(__file__).resolve().parents[1]


class ModelReliability(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.path=ROOT/"results/main/cnn_seed11.pt"
        cls.model=load_model(cls.path)
        cls.data=make_shard(7,"train",8)
        cls.trained,cls.history=fit(cls.data,seed=7,epochs=1)

    def test_ks029_all_open_checkpoints_load_default_safely(self):
        for path in sorted((ROOT/"results/main").glob("cnn_seed*.pt")):
            model=load_model(path)
            self.assertFalse(model.training)
            self.assertTrue(all(p.device.type=="cpu" for p in model.parameters()))

    def test_ks030_checkpoint_digest_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError,"SHA256"):
            load_model(self.path,expected_sha256="0"*64)

    def test_ks031_nonfinite_model_weights_are_rejected(self):
        with temporary_directory() as folder:
            state={k:v.clone() for k,v in self.model.state_dict().items()}
            state[next(iter(state))].view(-1)[0]=float("nan")
            path=Path(folder)/"invalid.pt";torch.save({"state_dict":state},path)
            with self.assertRaisesRegex(ValueError,"nonfinite"):load_model(path)

    def test_ks032_incompatible_architecture_is_rejected(self):
        with temporary_directory() as folder:
            path=Path(folder)/"wrong.pt";torch.save({"state_dict":{"wrong.weight":torch.ones(1)}},path)
            with self.assertRaises(RuntimeError):load_model(path)

    def test_ks033_inference_preserves_acquired_measurements(self):
        rng=np.random.default_rng(71);truth=phantom(71);mask,_=acquisition(rng);y=noisy_kspace(truth,rng)
        output=reconstruct(self.model,y,mask)
        np.testing.assert_allclose(fft(output)[:,mask],y[:,mask],atol=2e-6,rtol=2e-6)

    def test_ks034_unacquired_values_cannot_leak_into_inference(self):
        rng=np.random.default_rng(72);y=noisy_kspace(phantom(72),rng);mask,_=acquisition(rng)
        first=reconstruct(self.model,y,mask);y[:,~mask]=1e9+1e9j
        np.testing.assert_array_equal(reconstruct(self.model,y,mask),first)

    def test_ks035_complex_phase_survives_full_acquisition(self):
        truth=phantom(73)*1j
        image=reconstruct(self.model,fft(truth),np.ones(64,dtype=bool))
        np.testing.assert_allclose(image,truth,atol=2e-6)
        self.assertGreater(np.linalg.norm(image.imag),1.)

    def test_ks036_checkpoint_reload_preserves_prediction(self):
        rng=np.random.default_rng(74);y=noisy_kspace(phantom(74),rng);mask,_=acquisition(rng)
        digest=hashlib.sha256(self.path.read_bytes()).hexdigest()
        loaded=load_model(self.path,expected_sha256=digest)
        np.testing.assert_array_equal(reconstruct(loaded,y,mask),reconstruct(self.model,y,mask))

    def test_ks037_real_training_updates_weights_with_finite_loss(self):
        torch.manual_seed(7);initial=ResidualCNN()
        changes=[not torch.equal(v,self.trained.state_dict()[k]) for k,v in initial.state_dict().items()]
        self.assertTrue(any(changes));self.assertEqual(len(self.history),1)
        self.assertTrue(np.isfinite(self.history).all())

    def test_ks038_fixed_seed_training_is_reproducible(self):
        again,history=fit(self.data,seed=7,epochs=1)
        self.assertEqual(history,self.history)
        for key,value in self.trained.state_dict().items():self.assertTrue(torch.equal(value,again.state_dict()[key]))
