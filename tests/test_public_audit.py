import unittest
import numpy as np
from ksentinel.audit import audit_reconstruction


class ExternalAudit(unittest.TestCase):
    def test_analytic_signal_rejection(self):
        # A constant image has a single nonzero centered DC coefficient.
        image=np.ones((8,8),dtype=complex)
        result=audit_reconstruction(image,np.array([4]),np.zeros((8,1),complex),np.ones(99),1)
        self.assertAlmostEqual(result["score"],8)
        self.assertTrue(result["flagged"])
        self.assertEqual(result["p_value"],0.01)

    def test_no_information_does_not_imply_safe(self):
        image=np.ones((8,8),dtype=complex)
        result=audit_reconstruction(image,np.array([1]),np.zeros((8,1),complex),np.ones(99),1)
        self.assertFalse(result["flagged"])
        self.assertIn("not an anatomical correctness certificate",result["meaning"])

    def test_invalid_masks_and_values(self):
        for cols in (np.array([1,1]),np.array([-1]),np.array([8]),np.array([1.5])):
            with self.assertRaises(ValueError):
                audit_reconstruction(np.ones((8,8),complex),cols,np.zeros((8,len(cols))),np.ones(99),1)
        with self.assertRaises(ValueError):
            audit_reconstruction(np.ones((8,8),complex),np.array([1]),np.full((8,1),np.nan),np.ones(99),1)


if __name__ == "__main__": unittest.main()
