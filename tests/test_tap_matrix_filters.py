"""Consistency tests for vectorized non-causal tap-matrix filters."""

from __future__ import annotations

import numpy as np

from ppg_hr.experimental.noncausal_lms import noncausal_lms_filter
from ppg_hr.experimental.rff_lms import get_rff_weights, noncausal_rff_lms_filter
from ppg_hr.experimental.tap_matrix import build_noncausal_tap_matrix
from ppg_hr.experimental.volterra import (
    _nearest_tap_indices,
    _quadratic_features,
    noncausal_volterra_filter,
)


def _zscore(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float).copy()
    arr[~np.isfinite(arr)] = 0.0
    sd = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    if not np.isfinite(sd) or sd <= 1e-12:
        return arr - float(np.mean(arr))
    return (arr - float(np.mean(arr))) / sd


def _ref_lms(u: np.ndarray, d: np.ndarray, M: int, K: int, mu: float) -> np.ndarray:
    u_arr = _zscore(u)
    d_arr = _zscore(d)
    n = min(u_arr.size, d_arr.size)
    u_arr = u_arr[:n]
    d_arr = d_arr[:n]
    M = max(1, int(M))
    K = max(0, int(K))
    out = d_arr.copy()
    if n - K < M:
        return out
    w = np.zeros(M + K, dtype=float)
    for idx in range(M - 1, n - K):
        x = u_arr[idx - M + 1 : idx + K + 1][::-1]
        y = float(np.dot(w, x))
        err = float(d_arr[idx] - y)
        out[idx] = err
        w += (float(mu) / (float(np.dot(x, x)) + 1e-9)) * x * err
    return out


def _ref_volterra(
    u: np.ndarray,
    d: np.ndarray,
    M: int,
    K: int,
    mu1: float,
    alpha_u: float,
    M2: int,
    mu_min: float = 1e-5,
) -> np.ndarray:
    u_arr = _zscore(u)
    d_arr = _zscore(d)
    n = min(u_arr.size, d_arr.size)
    u_arr = u_arr[:n]
    d_arr = d_arr[:n]
    M = max(1, int(M))
    K = max(0, int(K))
    out = d_arr.copy()
    if n - K < M:
        return out
    mu1 = max(float(mu_min), float(mu1))
    mu2 = max(float(mu_min) * max(float(alpha_u), 1e-6), float(alpha_u) * mu1)
    near_idx = _nearest_tap_indices(M, K, M2)
    q_count = int(len(near_idx) * (len(near_idx) + 1) / 2)
    w1 = np.zeros(M + K, dtype=float)
    w2 = np.zeros(q_count, dtype=float)
    for idx in range(M - 1, n - K):
        x = u_arr[idx - M + 1 : idx + K + 1][::-1]
        q = _quadratic_features(x[near_idx])
        y = float(np.dot(w1, x) + np.dot(w2, q))
        err = float(d_arr[idx] - y)
        out[idx] = err
        w1 += (mu1 / (float(np.dot(x, x)) + 1e-9)) * x * err
        if q.size:
            w2 += (mu2 / (float(np.dot(q, q)) + 1e-9)) * q * err
    out[~np.isfinite(out)] = 0.0
    return out


def _ref_rff(
    u: np.ndarray,
    d: np.ndarray,
    M: int,
    K: int,
    mu: float,
    D: int,
    sigma: float,
    rff_seed: int,
    mu_min: float = 1e-5,
) -> np.ndarray:
    u_arr = _zscore(u)
    d_arr = _zscore(d)
    n = min(u_arr.size, d_arr.size)
    u_arr = u_arr[:n]
    d_arr = d_arr[:n]
    M = max(1, int(M))
    K = max(0, int(K))
    out = d_arr.copy()
    if n - K < M:
        return out
    D = max(1, int(D))
    W, b = get_rff_weights(D, M + K, max(float(sigma), 1e-6), int(rff_seed))
    theta = np.zeros(D, dtype=float)
    scale = float(np.sqrt(2.0 / D))
    mu = max(float(mu_min), float(mu) if np.isfinite(mu) else float(mu_min))
    for idx in range(M - 1, n - K):
        x = u_arr[idx - M + 1 : idx + K + 1][::-1]
        z = scale * np.cos(W @ x + b)
        y = float(theta @ z)
        err = float(d_arr[idx] - y)
        out[idx] = err
        theta += mu * err * z
    out[~np.isfinite(out)] = 0.0
    return out


def test_tap_matrix_order_and_indices() -> None:
    u = np.arange(20, dtype=float)
    for M, K in [(1, 0), (2, 0), (2, 3), (8, 0), (8, 4)]:
        X, valid = build_noncausal_tap_matrix(u, M, K)
        assert X.shape == (len(range(M - 1, len(u) - K)), M + K)
        for row, idx in enumerate(valid):
            np.testing.assert_array_equal(X[row], u[idx - M + 1 : idx + K + 1][::-1])


def test_vectorized_filters_match_reference_loops() -> None:
    rng = np.random.default_rng(1234)
    u = rng.normal(size=128)
    d = rng.normal(size=128)
    for M, K in [(1, 0), (2, 0), (2, 3), (8, 0), (8, 4)]:
        np.testing.assert_allclose(noncausal_lms_filter(u, d, M, K, 0.01), _ref_lms(u, d, M, K, 0.01), atol=1e-12)
        np.testing.assert_allclose(
            noncausal_volterra_filter(u, d, M, K, 0.01, alpha_u=0.1, M2=3),
            _ref_volterra(u, d, M, K, 0.01, alpha_u=0.1, M2=3),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            noncausal_rff_lms_filter(u, d, M, K, 0.008, D=32, sigma=1.5, rff_seed=77),
            _ref_rff(u, d, M, K, 0.008, D=32, sigma=1.5, rff_seed=77),
            atol=1e-12,
        )
