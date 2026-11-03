"""Default-safe loading and measurement-only inference for the open CNN."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from .core import ifft
from .learned import ResidualCNN, predict


def load_model(path, expected_sha256=None):
    path = Path(path)
    if expected_sha256 is not None:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected_sha256.lower():
            raise ValueError("Checkpoint SHA256 mismatch")
    checkpoint = torch.load(path, weights_only=True, map_location="cpu")
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError("Expected a checkpoint containing state_dict")
    model = ResidualCNN()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("Checkpoint contains nonfinite weights")
    model.eval()
    return model


def reconstruct(model, measured_kspace, initial_mask):
    """Use only initially acquired coefficients; unacquired values are ignored.

    No ground truth, audit values or audit mask are accepted by this interface.
    Input follows core.fft's centered, orthonormal convention.
    """
    measured = np.asarray(measured_kspace)
    mask = np.asarray(initial_mask)
    if measured.ndim != 2 or not all(measured.shape) or not np.iscomplexobj(measured):
        raise ValueError("Measured k-space must be a nonempty 2D complex array")
    if mask.dtype != np.bool_ or mask.shape != (measured.shape[1],) or not mask.any():
        raise ValueError("Initial mask must select columns with a boolean width-vector")
    if not np.isfinite(measured[:, mask]).all():
        raise ValueError("Acquired coefficients must be finite")
    observed = np.where(mask[None, :], measured, 0)
    image = predict(model, ifft(observed), observed, mask)
    if not np.isfinite(image).all():
        raise ValueError("Model produced a nonfinite reconstruction")
    return image
