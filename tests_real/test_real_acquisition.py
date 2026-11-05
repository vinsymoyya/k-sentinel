"""Real-acquisition contracts, separate from the frozen 45-case legacy suite."""
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
import h5py
import numpy as np
import torch
from ksentinel.model_io import load_model
from ksentinel.real_acquisition import (audit_score, file_sha256, fftc, fuse, ifftc,
    load_acquisition, observed_input, rank_pvalue, reconstruct_coils, rss, sampling,
    sampling_seed, validate_manifest)

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT/'tmp'/'real-tests'


class RealAcquisitionContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        SCRATCH.mkdir(parents=True, exist_ok=True)
        torch.set_num_threads(2)

    def test_centered_complex_fourier_inverse_and_energy(self):
        rng = np.random.default_rng(20)
        image = rng.normal(size=(4, 15, 18)) + 1j*rng.normal(size=(4, 15, 18))
        np.testing.assert_allclose(ifftc(fftc(image)), image, atol=2e-14)
        self.assertAlmostEqual(np.linalg.norm(image), np.linalg.norm(fftc(image)), places=10)

    def test_analytic_center_and_multicoil_rss(self):
        kspace = np.ones((2, 8, 8), complex)
        kspace[1] *= 2
        expected = np.zeros((8, 8))
        expected[4, 4] = 8*np.sqrt(5)
        np.testing.assert_allclose(rss(ifftc(kspace)), expected, atol=1e-12)

    def test_unavailable_source_columns_never_enter_initial_or_audit(self):
        support = np.zeros(256, bool)
        support[30:225] = True
        for r in (2, 4, 8):
            for pattern in ('random', 'equispaced'):
                mask, audit, acs = sampling(support, r, pattern, 20261105)
                self.assertEqual(mask.sum(), round(195/r))
                self.assertEqual(len(audit), 8)
                self.assertEqual(acs.sum(), 16)
                self.assertFalse(np.any(mask & ~support))
                self.assertTrue(support[audit].all())
                self.assertFalse(mask[audit].any())
                self.assertFalse(np.any(acs & ~mask))

    def test_sampling_reproducible_and_rejects_insufficient_support(self):
        support = np.ones(64, bool)
        seed = sampling_seed('case', 'T2', 4, 'random')
        first = sampling(support, 4, 'random', seed, acs=4)
        second = sampling(support, 4, 'random', seed, acs=4)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)
        with self.assertRaises(ValueError):
            sampling(np.ones(20, bool), 8, 'random', 1)

    def test_initial_normalization_ignores_poisoned_audit_and_reference_values(self):
        rng = np.random.default_rng(6)
        values = rng.normal(size=(4, 32, 64)) + 1j*rng.normal(size=(4, 32, 64))
        mask, _, acs = sampling(np.ones(64, bool), 4, 'random', 9, acs=4)
        poisoned = values.copy()
        poisoned[..., ~mask] = complex(float('nan'), float('nan'))
        a, scale_a = observed_input(values, mask, acs)
        b, scale_b = observed_input(poisoned, mask, acs)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(scale_a, scale_b)

    def test_real_open_model_keeps_measurements_and_ignores_unobserved_values(self):
        rng = np.random.default_rng(10)
        values = rng.normal(size=(4, 64, 64)) + 1j*rng.normal(size=(4, 64, 64))
        mask, _, acs = sampling(np.ones(64, bool), 4, 'random', 10, acs=4)
        model = load_model(ROOT/'results/main/cnn_seed11.pt')
        candidate = reconstruct_coils(model, values, mask, acs)
        poisoned = values.copy()
        poisoned[..., ~mask] = 1e20 + 1e20j
        np.testing.assert_array_equal(candidate, reconstruct_coils(model, poisoned, mask, acs))
        np.testing.assert_allclose(fftc(candidate)[..., mask], values[..., mask], atol=1e-12)

    def test_analytic_opposite_phase_audit_and_unit_invariance(self):
        candidate = np.ones((4, 8, 8), complex)
        columns = np.array([4])
        values = fftc(-candidate)[..., columns]
        self.assertAlmostEqual(audit_score(candidate, columns, values), 4.)
        self.assertAlmostEqual(audit_score(candidate*13, columns, values*13), 4.)
        with self.assertRaises(ValueError):
            audit_score(candidate, columns, np.zeros_like(values))

    def test_rank_calibration_ties_are_conservative_and_subject_counted(self):
        self.assertEqual(rank_pvalue(2, [1]*30), 1/31)
        self.assertEqual(rank_pvalue(1+1e-15, [1]*30), 1.)
        self.assertEqual(rank_pvalue(1, [1]*30), 1.)
        with self.assertRaises(ValueError):
            rank_pvalue(1, [])
        with self.assertRaises(ValueError):
            rank_pvalue(1, np.ones(30, dtype=complex))

    def test_audit_fusion_preserves_initial_samples_and_reduces_coil_reference_error(self):
        rng = np.random.default_rng(77)
        source = rng.normal(size=(4, 16, 64)) + 1j*rng.normal(size=(4, 16, 64))
        mask, audit, acs = sampling(np.ones(64, bool), 4, 'random', 3, acs=4)
        candidate = reconstruct_coils(None, source, mask, acs)
        result = fuse(candidate, audit, source[..., audit])
        np.testing.assert_allclose(fftc(result)[..., mask], source[..., mask], atol=1e-12)
        np.testing.assert_allclose(fftc(result)[..., audit], source[..., audit], atol=1e-12)
        self.assertLess(np.linalg.norm(result-ifftc(source)), np.linalg.norm(candidate-ifftc(source)))

    def test_participants_cannot_cross_roles_but_repetitions_can_share_a_role(self):
        one = {'scan_id':'s', 'role':'test', 'relative_path':'test/a.h5'}
        two = {**one, 'relative_path':'test/b.h5'}
        self.assertEqual(validate_manifest({'files':[one, two]}), {'s':'test'})
        with self.assertRaisesRegex(ValueError, 'leakage'):
            validate_manifest({'files':[one, {**two, 'role':'calibration'}]})
        with self.assertRaises(ValueError):
            validate_manifest({'files':[{**one, 'relative_path':'../escape.h5'}]})

    def test_h5_loader_checks_bytes_support_centered_rss_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as directory:
            path = Path(directory)/'case.h5'
            kspace = np.ones((18,4,16,16), np.complex64)*np.arange(1,5)[None,:,None,None]
            reference = np.zeros((18,16,16), np.float32)
            reference[:,8,8] = 16*np.sqrt(30)
            with h5py.File(path,'w') as h:
                h['kspace'] = kspace
                h['reconstruction_rss'] = reference
            item = {'relative_path':'case.h5', 'sha256':file_sha256(path), 'bytes':path.stat().st_size,
                    'slices':18, 'coils':4, 'frequency_encode':16, 'phase_encode':16, 'acquired_pe_lines':16}
            raw, target, support, error = load_acquisition(directory, item)
            self.assertEqual(raw.shape, (3,4,16,16))
            self.assertTrue(support.all())
            self.assertLess(error, 1e-5)
            with h5py.File(path,'r+') as h:
                h['reconstruction_rss'][8,8,8] = 0
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                load_acquisition(directory,item)
            item.update(sha256=file_sha256(path), bytes=path.stat().st_size)
            with self.assertRaisesRegex(ValueError, 'RSS'):
                load_acquisition(directory,item)

    def test_natural_error_summary_keeps_test_errors_out_of_calibration(self):
        spec = importlib.util.spec_from_file_location('real_run', ROOT/'scripts/run_real_acquisition.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        base = {'sequence':'T2', 'pattern':'random', 'acceleration':4, 'method':'m4raw_finetuned_cnn',
                'motion_type':'none', 'audit_score':1., 'coil_nrmse':.2, 'fused_rss_nrmse':.05}
        rows = [{**base, 'role':'calibration', 'scan_id':f'c{i}', 'repetition':1, 'rss_nrmse':.1} for i in range(30)]
        rows += [{**base, 'role':'test', 'scan_id':f't{i}', 'repetition':rep, 'rss_nrmse':.2*rep} for i in range(25) for rep in (1,2)]
        result = module.summarize(rows)
        self.assertEqual(result['calibrations'][0]['subjects'], 30)
        self.assertEqual(result['calibrations'][0]['high_reference_error_cutoff'], .1)
        primary = next(g for g in result['groups'] if g['repetition']==1)
        self.assertEqual(primary['unflagged_high_reference_error'], 25)
        self.assertIsNone(primary['reference_error_auroc'])
        self.assertEqual(len(result['paired_repeats']), 25)
        self.assertEqual(result['paired_repetition_error_change']['participants'], 25)


if __name__ == '__main__':
    unittest.main()
