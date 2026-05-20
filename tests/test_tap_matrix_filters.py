"""Consistency tests for vectorized non-causal tap-matrix filters."""

from __future__ import annotations

import numpy as np

from ppg_hr.experimental.klms import noncausal_klms_filter
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


def _ref_klms(
    u: np.ndarray,
    d: np.ndarray,
    M: int,
    K: int,
    mu: float,
    sigma: float,
    epsilon: float,
) -> np.ndarray:
    u_arr = _zscore(u)
    d_arr = _zscore(d)
    n = min(u_arr.size, d_arr.size)
    u_arr = u_arr[:n]
    d_arr = d_arr[:n]
    M = max(1, int(M))
    K = max(0, int(K))
    out = np.zeros(n, dtype=float)
    if n - K < M:
        return out
    centers = np.zeros((M + K, 0), dtype=float)
    weights = np.zeros(0, dtype=float)
    two_sigma2 = 2.0 * max(float(sigma), 1e-12) ** 2
    for idx in range(M - 1, n - K):
        x = u_arr[idx - M + 1 : idx + K + 1][::-1]
        if centers.shape[1] == 0:
            err = float(d_arr[idx])
            out[idx] = err
            centers = x.reshape(-1, 1)
            weights = np.asarray([float(mu) * err], dtype=float)
            continue

        diffs = centers - x[:, None]
        dists = np.sum(diffs * diffs, axis=0)
        kappa = np.exp(-dists / two_sigma2)
        y = float(weights @ kappa)
        err = float(d_arr[idx] - y)
        out[idx] = err

        min_idx = int(np.argmin(dists))
        if float(dists[min_idx]) <= float(epsilon):
            weights[min_idx] += float(mu) * err
        else:
            centers = np.concatenate([centers, x.reshape(-1, 1)], axis=1)
            weights = np.concatenate([weights, np.asarray([float(mu) * err])])
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
            noncausal_rff_lms_filter(u, d, M, K, 0.008, D=32, sigma=1.5, rff_seed=77, update_mode="lms"),
            _ref_rff(u, d, M, K, 0.008, D=32, sigma=1.5, rff_seed=77),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            noncausal_klms_filter(
                u,
                d,
                M,
                K,
                step_size=0.05,
                sigma=1.5,
                epsilon=0.1,
                max_dictionary_size=10_000,
                distance_mode="absolute_squared",
                normalized_update=False,
            ),
            _ref_klms(u, d, M, K, mu=0.05, sigma=1.5, epsilon=0.1),
            atol=1e-12,
        )


def test_rff_lms_default_nlms_stays_finite_and_reports_diagnostics() -> None:
    rng = np.random.default_rng(2026)
    u = rng.normal(size=256)
    d = rng.normal(size=256)

    out, diagnostics = noncausal_rff_lms_filter(
        u,
        d,
        M=8,
        K=4,
        mu=5.0,
        D=64,
        sigma=1.0,
        rff_seed=13,
        err_clip=5.0,
        theta_norm_guard=25.0,
        return_diagnostics=True,
    )

    assert np.all(np.isfinite(out))
    assert "theta_norm_t" in diagnostics
    assert "max_abs_theta_t" in diagnostics
    assert diagnostics["theta_norm_t"].shape[0] > 0
    assert diagnostics["guard_triggered_count"] >= 0
    assert diagnostics["err_clip_count"] >= 0


def test_klms_default_limits_dictionary_and_reports_diagnostics() -> None:
    rng = np.random.default_rng(2027)
    u = rng.normal(size=192)
    d = rng.normal(size=192)

    out, diagnostics = noncausal_klms_filter(
        u,
        d,
        M=6,
        K=2,
        step_size=1.0,
        sigma=1.0,
        epsilon=0.005,
        max_dictionary_size=12,
        center_prune_policy="freeze_new_centers",
        return_diagnostics=True,
    )

    assert np.all(np.isfinite(out))
    assert int(np.max(diagnostics["dictionary_size_t"])) <= 12
    assert diagnostics["max_dictionary_reached_count"] > 0
    assert diagnostics["prune_count"] == 0


def test_klms_normalized_distance_differs_from_absolute_squared_threshold() -> None:
    rng = np.random.default_rng(2028)
    u = rng.normal(size=96)
    d = rng.normal(size=96)

    _, normalized = noncausal_klms_filter(
        u,
        d,
        M=8,
        K=4,
        step_size=0.05,
        sigma=1.0,
        epsilon=0.1,
        distance_mode="normalized",
        normalized_update=True,
        return_diagnostics=True,
    )
    _, absolute = noncausal_klms_filter(
        u,
        d,
        M=8,
        K=4,
        step_size=0.05,
        sigma=1.0,
        epsilon=0.1,
        distance_mode="absolute_squared",
        normalized_update=True,
        return_diagnostics=True,
    )

    assert int(normalized["dictionary_size_t"][-1]) <= int(absolute["dictionary_size_t"][-1])
