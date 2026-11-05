"""Native centered multicoil operators for the separate real-MRI extension.

The synthetic v0.2 operators and experiment are unchanged. An all-available
source reconstruction is a noisy reference, never an anatomical certificate.
"""
import hashlib
import json
from pathlib import Path
import numpy as np


SLICES = (7, 8, 9)


def fftc(image):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image, axes=(-2, -1)),
                                    axes=(-2, -1), norm='ortho'), axes=(-2, -1))


def ifftc(kspace):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace, axes=(-2, -1)),
                                     axes=(-2, -1), norm='ortho'), axes=(-2, -1))


def rss(coil_images):
    return np.sqrt(np.sum(np.abs(coil_images) ** 2, axis=-3))


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_manifest(manifest):
    roles, paths = {}, set()
    for item in manifest['files']:
        path = item['relative_path']
        if path in paths or Path(path).is_absolute() or '..' in Path(path).parts:
            raise ValueError('Repeated or unsafe acquisition path')
        paths.add(path)
        subject, role = item['scan_id'], item['role']
        if role not in ('train', 'calibration', 'test', 'motion_risk'):
            raise ValueError('Unknown participant role')
        declared = item.get('participant_split')
        expected = {'train':'train', 'calibration':'validation', 'test':'test'}
        if declared is not None and ((role in expected and declared != expected[role])
                or (role == 'motion_risk' and not declared.startswith('motion_'))):
            raise ValueError('Role disagrees with official source participant split')
        if subject in roles and roles[subject] != role:
            raise ValueError('Participant leakage across roles')
        roles[subject] = role
    if not paths:
        raise ValueError('Empty manifest')
    return roles


def load_acquisition(root, item, slices=SLICES):
    import h5py
    root = Path(root).resolve()
    path = (root / item['relative_path']).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Acquisition path escapes data directory')
    if file_sha256(path) != item['sha256'] or path.stat().st_size != int(item['bytes']):
        raise ValueError('Source acquisition SHA256/size mismatch')
    with h5py.File(path, 'r') as source:
        shape = (int(item['slices']), int(item['coils']), int(item['frequency_encode']), int(item['phase_encode']))
        if source['kspace'].shape != shape or source['reconstruction_rss'].shape != (shape[0], *shape[2:]):
            raise ValueError('Unexpected source array axes/shape')
        if not slices or sorted(set(slices)) != list(slices) or min(slices) < 0 or max(slices) >= shape[0]:
            raise ValueError('Invalid slice selection')
        kspace = source['kspace'][list(slices)]
        reference = source['reconstruction_rss'][list(slices)]
    if not np.iscomplexobj(kspace) or not np.isfinite(kspace).all() or not np.isfinite(reference).all():
        raise ValueError('Source arrays must be finite, with complex raw data')
    support = np.any(kspace != 0, axis=(0, 1, 2))
    if int(support.sum()) != int(item['acquired_pe_lines']):
        raise ValueError('Source PE support differs from manifest')
    relative = float(np.linalg.norm(rss(ifftc(kspace)) - reference) / max(np.linalg.norm(reference), 1e-12))
    if relative > 1e-5:
        raise ValueError('Centered reconstruction disagrees with stored source RSS')
    return kspace, reference, support, relative


def sampling(support, acceleration, pattern, seed, audit=8, acs=16):
    support = np.asarray(support)
    if support.dtype != np.bool_ or support.ndim != 1 or not support.any():
        raise ValueError('Source support must be a nonempty boolean vector')
    if acceleration not in (2, 4, 8) or pattern not in ('random', 'equispaced'):
        raise ValueError('Unsupported acquisition condition')
    if not isinstance(audit, int) or audit < 1 or not isinstance(acs, int) or acs < 1:
        raise ValueError('Positive integer audit/ACS counts required')
    available = np.flatnonzero(support)
    count = int(np.rint(len(available) / acceleration))
    if count < acs or len(available) - count < audit:
        raise ValueError('Insufficient acquired support for initial/ACS/audit budget')
    central = available[np.argsort(abs(available - (len(support) // 2)), kind='stable')[:acs]]
    other = np.setdiff1d(available, central)
    rng = np.random.default_rng(seed)
    if pattern == 'random':
        selected = rng.choice(other, count - acs, replace=False)
    else:
        selected = other[np.linspace(0, len(other) - 1, count - acs, dtype=int)]
    mask = np.zeros(len(support), dtype=bool)
    mask[np.r_[central, selected]] = True
    # Independent RNG stream; no candidate, reference or audit values select it.
    order = np.random.default_rng(seed + 1729).permutation(np.flatnonzero(support & ~mask))[:audit]
    acs_mask = np.zeros_like(mask)
    acs_mask[central] = True
    return mask, order, acs_mask


def sampling_seed(subject, sequence, acceleration, pattern):
    # Omit repetition so paired repeated acquisitions use identical sampling.
    label = f'20261105:{subject}:{sequence}:{acceleration}:{pattern}'
    return int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], 'little')


def observed_input(measurements, mask, acs_mask):
    values = np.asarray(measurements)
    mask, acs_mask = np.asarray(mask), np.asarray(acs_mask)
    if values.ndim != 3 or not np.iscomplexobj(values) or not all(values.shape):
        raise ValueError('Expected nonempty complex [coil,FE,PE] input')
    if mask.dtype != np.bool_ or acs_mask.dtype != np.bool_ or mask.shape != (values.shape[-1],) or acs_mask.shape != mask.shape:
        raise ValueError('Boolean PE masks must match the last dimension')
    if not mask.any() or not acs_mask.any() or np.any(acs_mask & ~mask):
        raise ValueError('ACS must be a nonempty subset of initial measurements')
    if not np.isfinite(values[..., mask]).all():
        raise ValueError('Initial measurements must be finite')
    measured = np.where(mask[None, None, :], values, 0)
    low_resolution = ifftc(np.where(acs_mask[None, None, :], measured, 0))
    scales = np.quantile(abs(low_resolution), .99, axis=(-2, -1))
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError('Initial ACS normalization is degenerate')
    return measured, scales


def reconstruct_coils(model, measurements, mask, acs_mask):
    """No source reference, audit values or audit identities enter inference."""
    import torch
    from .learned import channels
    measured, scales = observed_input(measurements, mask, acs_mask)
    zero_fill = ifftc(measured)
    if model is None:
        return zero_fill
    normalized = zero_fill / scales[:, None, None]
    with torch.inference_mode():
        output = model(channels(normalized)).cpu().numpy()
    image = (output[:, 0] + 1j * output[:, 1]) * scales[:, None, None]
    predicted = fftc(image)
    predicted[..., mask] = measured[..., mask]
    result = ifftc(predicted)
    if not np.isfinite(result).all():
        raise ValueError('Model produced nonfinite coil reconstruction')
    return result


def audit_score(candidate_coils, columns, audit_values):
    candidate = np.asarray(candidate_coils)
    columns = np.asarray(columns)
    values = np.asarray(audit_values)
    if candidate.ndim != 3 or columns.ndim != 1 or not len(columns) or columns.dtype.kind not in 'iu':
        raise ValueError('Invalid multicoil audit dimensions')
    if len(set(columns.tolist())) != len(columns) or columns.min() < 0 or columns.max() >= candidate.shape[-1]:
        raise ValueError('Invalid audit columns')
    if values.shape != (*candidate.shape[:-1], len(columns)) or not np.isfinite(values).all() or not np.isfinite(candidate).all():
        raise ValueError('Invalid audit values')
    denominator = float(np.sum(abs(values) ** 2))
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError('Audit signal norm must be finite and positive')
    score = float(np.sum(abs(fftc(candidate)[..., columns] - values) ** 2) / denominator)
    if not np.isfinite(score):
        raise ValueError('Nonfinite audit score')
    # Canonicalize insignificant FFT noise, including the exactly-1 zero-fill control.
    return float(np.round(score, 12))


def fuse(candidate_coils, columns, audit_values):
    predicted = fftc(candidate_coils)
    predicted[..., columns] = audit_values
    return ifftc(predicted)


def relative_error(candidate, reference):
    return float(np.linalg.norm(candidate - reference) / max(np.linalg.norm(reference), 1e-12))


def rank_pvalue(score, calibration):
    calibration = np.asarray(calibration)
    if np.iscomplexobj(calibration) or not np.issubdtype(calibration.dtype, np.number):
        raise ValueError('Calibration scores must be real numbers')
    if np.ndim(score) != 0 or np.iscomplexobj(score):
        raise ValueError('Score must be a real scalar')
    calibration = calibration.astype(float)
    if calibration.ndim != 1 or not len(calibration) or not np.isfinite(calibration).all() or np.any(calibration < 0):
        raise ValueError('Calibration must contain finite nonnegative subject scores')
    if not np.isfinite(score) or score < 0:
        raise ValueError('Score must be finite and nonnegative')
    # Treat FFT round-off as a tie, conservatively increasing the p-value.
    tied = np.isclose(calibration, score, rtol=1e-12, atol=1e-12)
    return float((1 + np.count_nonzero((calibration >= score) | tied)) / (len(calibration) + 1))


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
