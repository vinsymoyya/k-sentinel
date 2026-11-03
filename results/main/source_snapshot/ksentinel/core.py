"""Numpy-only imaging operators and statistically calibrated audit.

All images remain complex. Magnitude is used only for display.
"""
import math
import numpy as np


def fft(image):
    return np.fft.fftshift(np.fft.fft2(image, norm="ortho"), axes=(-2, -1))


def ifft(kspace):
    return np.fft.ifft2(np.fft.ifftshift(kspace, axes=(-2, -1)), norm="ortho")


def phantom(identity, size=64, shifted=False):
    """One independent digital object; no source patient or copied anatomy."""
    rng = np.random.default_rng(identity)
    yy, xx = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
    angle = rng.uniform(-0.3, 0.3)
    u, v = xx*np.cos(angle)+yy*np.sin(angle), -xx*np.sin(angle)+yy*np.cos(angle)
    outer = (u/0.81)**2 + (v/0.92)**2 < 1
    image = outer.astype(float) * rng.uniform(0.2, 0.4)
    for _ in range(12 if shifted else 7):
        cx, cy = rng.uniform(-0.55, 0.55, 2)
        rx, ry = rng.uniform(0.04 if shifted else 0.09, 0.26, 2)
        a = rng.uniform(-math.pi, math.pi)
        dx, dy = xx-cx, yy-cy
        region = ((dx*np.cos(a)+dy*np.sin(a))/rx)**2 + ((-dx*np.sin(a)+dy*np.cos(a))/ry)**2 < 1
        image += region * outer * rng.uniform(-0.14, 0.38)
    image = np.maximum(image, 0)
    image /= max(image.max(), 1e-12)
    phase = rng.uniform(-0.8, 0.8)*xx + rng.uniform(-0.8, 0.8)*yy
    return image * np.exp(1j*phase)


def acquisition(rng, size=64, initial=16, maximum_audit=8):
    if initial < 6 or initial+maximum_audit >= size:
        raise ValueError("Require six central lines and at least one unobserved line")
    mask = np.zeros(size, dtype=bool)
    mask[size//2-3:size//2+3] = True
    mask[rng.choice(np.flatnonzero(~mask), initial-6, replace=False)] = True
    audit_order = rng.permutation(np.flatnonzero(~mask))[:maximum_audit]
    return mask, audit_order


def noisy_kspace(image, rng, sigma=0.02):
    noise = sigma / np.sqrt(2) * (rng.normal(size=image.shape) + 1j*rng.normal(size=image.shape))
    return fft(image) + noise


def project(candidate, measured, mask):
    k = fft(candidate)
    k[..., mask] = measured[..., mask]
    return ifft(k)


def null_template(image, mask, rng, relative=0.1):
    n = image.shape[-1]
    yy, xx = np.mgrid[-1:1:complex(n), -1:1:complex(n)]
    cx, cy = rng.uniform(-0.45, 0.45, 2)
    radius = rng.uniform(0.035, 0.09)
    template = np.exp(-((xx-cx)**2+(yy-cy)**2)/(2*radius**2))
    k = fft(template.astype(complex))
    k[:, mask] = 0
    template = ifft(k)
    return template * (relative*np.linalg.norm(image)/np.linalg.norm(template))


def line_error(image, forbidden, rng, relative=0.1):
    allowed = np.flatnonzero(~forbidden)
    if not len(allowed):
        raise ValueError("No unobserved column remains")
    k = np.zeros_like(image, dtype=complex)
    line = int(rng.choice(allowed))
    k[:, line] = rng.normal(size=len(image)) + 1j*rng.normal(size=len(image))
    error = ifft(k)
    return error * relative*np.linalg.norm(image)/np.linalg.norm(error)


def score(candidate, measured_audit, columns, sigma=0.02):
    if sigma <= 0 or len(columns) == 0:
        raise ValueError("Positive noise scale and nonempty audit required")
    residual = fft(candidate)[:, columns] - measured_audit[:, columns]
    return float(np.mean(np.abs(residual)**2)/sigma**2)


def threshold(scores, alpha=0.05):
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Calibration requires a finite nonempty score vector")
    if not 0 < alpha < 1:
        raise ValueError("Alpha must be between zero and one")
    rank = math.ceil((len(values)+1)*(1-alpha))
    return float(np.sort(values)[rank-1]) if rank <= len(values) else float("inf")


def nrmse(candidate, truth):
    return float(np.linalg.norm(candidate-truth)/max(np.linalg.norm(truth), 1e-12))


def wilson(successes, total):
    if total <= 0:
        raise ValueError("Positive denominator required")
    z = 1.959963984540054
    p = successes/total
    den = 1+z*z/total
    center = (p+z*z/(2*total))/den
    radius = z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return [max(0., center-radius), min(1., center+radius)]


def paired_auc(positive, negative):
    """Ordinary empirical AUROC, with ties awarded half credit."""
    positive, negative = np.asarray(positive), np.sort(negative)
    less = np.searchsorted(negative, positive, side="left")
    equal = np.searchsorted(negative, positive, side="right") - less
    return float(np.mean((less+equal/2)/len(negative)))
