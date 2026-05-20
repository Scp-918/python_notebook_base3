"""Quantized Gaussian-kernel LMS filter for protocol windows."""

from __future__ import annotations

import numpy as np

from .tap_matrix import build_noncausal_tap_matrix

__all__ = ["noncausal_klms_filter"]


def noncausal_klms_filter(
    u: np.ndarray,
    d: np.ndarray,
    M: int,
    K: int,
    *,
    step_size: float,
    sigma: float,
    epsilon: float,
) -> np.ndarray:
    """Filter ``d`` with quantized Gaussian-kernel LMS using reference ``u``.

    ``epsilon`` follows the MATLAB reference literally: it is compared against
    squared Euclidean distances between tap vectors.
    """

    u_arr = _zscore(np.asarray(u, dtype=float).ravel())
    d_arr = _zscore(np.asarray(d, dtype=float).ravel())
    n = min(u_arr.size, d_arr.size)
    if n == 0:
        return np.asarray([], dtype=float)
    u_arr = u_arr[:n]
    d_arr = d_arr[:n]

    M = max(1, int(M))
    K = max(0, int(K))
    out = np.zeros(n, dtype=float)
    X, valid_indices = build_noncausal_tap_matrix(u_arr, M, K)
    if valid_indices.size == 0:
        return out

    step_size = float(step_size)
    if not np.isfinite(step_size):
        step_size = 0.0
    sigma = float(sigma)
    epsilon = float(epsilon)
    two_sigma2 = 2.0 * sigma * sigma
    centers = np.zeros((X.shape[1], 0), dtype=float)
    weights = np.zeros(0, dtype=float)

    for idx, uvec in zip(valid_indices, X, strict=True):
        if centers.shape[1] == 0:
            err = float(d_arr[idx])
            out[idx] = err
            centers = uvec.reshape(-1, 1)
            weights = np.asarray([step_size * err], dtype=float)
            continue

        diffs = centers - uvec[:, None]
        dists = np.sum(diffs * diffs, axis=0)
        if two_sigma2 > 0.0 and np.isfinite(two_sigma2):
            kappa = np.exp(-dists / two_sigma2)
        else:
            kappa = (dists == 0.0).astype(float)
        y = float(weights @ kappa)
        err = float(d_arr[idx] - y)
        out[idx] = err

        min_idx = int(np.argmin(dists))
        if float(dists[min_idx]) <= epsilon:
            weights[min_idx] += step_size * err
        else:
            centers = np.concatenate([centers, uvec.reshape(-1, 1)], axis=1)
            weights = np.concatenate([weights, np.asarray([step_size * err])])

    out[~np.isfinite(out)] = 0.0
    return out


def _zscore(x: np.ndarray) -> np.ndarray:
    """Return a finite z-scored copy of ``x``."""

    arr = np.asarray(x, dtype=float).copy()
    arr[~np.isfinite(arr)] = 0.0
    sd = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    if not np.isfinite(sd) or sd <= 1e-12:
        return arr - float(np.mean(arr))
    return (arr - float(np.mean(arr))) / sd
