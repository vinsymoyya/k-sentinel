import unittest
import numpy as np
from ksentinel.core import (fft, ifft, phantom, acquisition, project, noisy_kspace,
                           null_template, line_error, score, threshold, wilson, paired_auc)


class ScientificContracts(unittest.TestCase):
    def test_unitarity_and_inverse(self):
        rng = np.random.default_rng(4)
        image = rng.normal(size=(64,64))+1j*rng.normal(size=(64,64))
        np.testing.assert_allclose(ifft(fft(image)), image, atol=1e-12)
        self.assertAlmostEqual(np.linalg.norm(fft(image)), np.linalg.norm(image), places=10)

    def test_projection_cannot_see_injected_structure(self):
        rng = np.random.default_rng(10)
        image = phantom(10)
        mask, audit = acquisition(rng)
        y = noisy_kspace(image, rng)
        baseline = project(image, y, mask)
        delta = null_template(image, mask, rng)
        self.assertAlmostEqual(np.linalg.norm(delta)/np.linalg.norm(image), 0.1, places=12)
        self.assertLess(np.linalg.norm(fft(delta)[:,mask]), 1e-12)
        np.testing.assert_allclose(fft(baseline+delta)[:,mask], y[:,mask], atol=1e-12)
        self.assertGreater(np.linalg.norm(fft(delta)[:,audit]), 0)

    def test_frozen_audit_can_be_evaded(self):
        rng = np.random.default_rng(18)
        image = phantom(18)
        mask, audit = acquisition(rng)
        forbidden = mask.copy()
        forbidden[audit] = True
        delta = line_error(image, forbidden, rng)
        self.assertLess(np.linalg.norm(fft(delta)[:,forbidden]), 1e-12)
        y = noisy_kspace(image, rng)
        self.assertAlmostEqual(score(image+delta,y,audit),score(image,y,audit),places=10)

    def test_audit_fraction_and_independence(self):
        rng = np.random.default_rng(5)
        for _ in range(50):
            mask, audit = acquisition(rng)
            self.assertEqual(mask.sum(), 16)
            self.assertEqual(len(set(audit)), 8)
            self.assertFalse(mask[audit].any())

    def test_exact_rank_threshold_and_ties(self):
        self.assertEqual(threshold(range(1,20),0.1),18)
        self.assertEqual(threshold([1],0.05),float("inf"))
        self.assertEqual(threshold([3]*100),3)
        for bad in ([], [float("nan")]):
            with self.assertRaises(ValueError): threshold(bad)

    def test_wilson_known_boundary_and_auc(self):
        self.assertAlmostEqual(wilson(0,100)[1],0.03699349820698568)
        self.assertAlmostEqual(wilson(100,100)[0],0.9630065017930143)
        self.assertEqual(paired_auc([2,3],[0,1]),1)
        self.assertEqual(paired_auc([1,1],[1,1]),0.5)

    def test_measured_fusion_reduces_noiseless_error(self):
        rng = np.random.default_rng(1)
        truth = phantom(20)
        mask, audit = acquisition(rng)
        candidate = truth+null_template(truth,mask,rng)
        combined=mask.copy(); combined[audit]=True
        fused=project(candidate,fft(truth),combined)
        self.assertLess(np.linalg.norm(fused-truth),np.linalg.norm(candidate-truth))


if __name__ == "__main__": unittest.main()
