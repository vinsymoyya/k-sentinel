import json
import unittest
import numpy as np
from ksentinel.audit import audit_reconstruction


class AuditReliability(unittest.TestCase):
    def sample(self, **changes):
        args=dict(candidate=np.ones((8,8),complex),columns=np.array([4]),
                  audit_values=np.zeros((8,1),complex),calibration_scores=np.ones(99),sigma=1.)
        args.update(changes)
        return audit_reconstruction(**args)

    def test_ks011_negative_calibration_rejected(self):
        with self.assertRaises(ValueError): self.sample(calibration_scores=np.array([-1.,0.]))

    def test_ks012_complex_calibration_rejected(self):
        with self.assertRaises(ValueError): self.sample(calibration_scores=np.array([1+2j]*99))

    def test_ks013_empty_spatial_dimension_rejected(self):
        with self.assertRaises(ValueError): self.sample(candidate=np.ones((0,8),complex),audit_values=np.zeros((0,1)))

    def test_ks014_score_overflow_fails_closed(self):
        with np.errstate(over="ignore",invalid="ignore"):
            with self.assertRaises(ValueError): self.sample(candidate=np.full((8,8),1e308+1e308j))

    def test_ks015_invalid_alpha_rejected(self):
        for alpha in (0.,1.,-0.1,float("nan")):
            with self.subTest(alpha=alpha):
                with self.assertRaises(ValueError): self.sample(alpha=alpha)

    def test_ks016_invalid_noise_scale_rejected(self):
        for sigma in (0.,-1.,float("inf"),float("nan"),np.array([1.,2.]),1+1j):
            with self.subTest(sigma=str(sigma)):
                with self.assertRaises(ValueError): self.sample(sigma=sigma)

    def test_ks017_rank_pvalue_handles_ties_conservatively(self):
        result=self.sample(candidate=np.zeros((8,8),complex),calibration_scores=np.zeros(99))
        self.assertEqual(result["p_value"],1.)
        self.assertFalse(result["flagged"])

    def test_ks018_small_calibration_is_json_serializable_and_unflagged(self):
        result=self.sample(calibration_scores=np.ones(1))
        self.assertIsNone(result["threshold"])
        self.assertFalse(result["flagged"])
        json.dumps(result,allow_nan=False)

    def test_ks019_complex_phase_disagreement_is_detected(self):
        # Same magnitude, opposite phase: magnitude-only comparison would miss it.
        expected=np.zeros((8,1),complex); expected[4,0]=-8j
        result=self.sample(candidate=np.full((8,8),1j),audit_values=expected)
        self.assertAlmostEqual(result["score"],32.)
        self.assertTrue(result["flagged"])

    def test_ks020_score_invariant_to_consistent_unit_change(self):
        baseline=self.sample()
        scaled=self.sample(candidate=np.full((8,8),3+0j),sigma=3.)
        self.assertAlmostEqual(baseline["score"],scaled["score"])
        self.assertEqual(baseline["p_value"],scaled["p_value"])
