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
    max_dictionary_size: int = 300,
    center_prune_policy: str = "freeze_new_centers",
    distance_mode: str = "normalized",
    normalized_update: bool = True,
    nlms_eps: float = 1e-6,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray | int]]:
    """Filter ``d`` with quantized Gaussian-kernel LMS using reference ``u``.

    中文说明：
    ``epsilon`` 字段名保持不变以兼容旧 JSON/CSV，但默认语义已从绝对平方距离
    改为 ``squared_distance / tap_dim`` 的归一化距离阈值，避免 tap 维度变化时字典
    无限制膨胀。若需要复现实验旧逻辑，可传入
    ``distance_mode="absolute_squared", normalized_update=False``。
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
        if return_diagnostics:
            return out, _empty_diagnostics()
        return out

    step_size = float(step_size)
    if not np.isfinite(step_size):
        step_size = 0.0
    sigma = max(float(sigma), 1e-12)
    epsilon = float(epsilon)
    if not np.isfinite(epsilon):
        epsilon = 0.0
    max_dictionary_size = max(1, int(max_dictionary_size))
    prune_policy = str(center_prune_policy or "freeze_new_centers").lower()
    if prune_policy not in {"freeze_new_centers", "prune_oldest", "prune_smallest_weight"}:
        raise ValueError("center_prune_policy must be freeze_new_centers, prune_oldest, or prune_smallest_weight")
    mode = str(distance_mode or "normalized").lower()
    if mode not in {"normalized", "absolute_squared"}:
        raise ValueError("distance_mode must be 'normalized' or 'absolute_squared'")
    eps = float(nlms_eps) if np.isfinite(float(nlms_eps)) and float(nlms_eps) > 0.0 else 1e-6
    two_sigma2 = 2.0 * sigma * sigma
    centers = np.zeros((X.shape[1], 0), dtype=float)
    weights = np.zeros(0, dtype=float)
    tap_dim = max(1, int(X.shape[1]))
    dictionary_size_t: list[float] = []
    weight_norm_t: list[float] = []
    max_abs_weight_t: list[float] = []
    prune_count = 0
    max_dictionary_reached_count = 0

    for idx, uvec in zip(valid_indices, X, strict=True):
        if centers.shape[1] == 0:
            err = float(d_arr[idx])
            out[idx] = err
            centers = uvec.reshape(-1, 1)
            first_den = 1.0 + eps if normalized_update else 1.0
            weights = np.asarray([step_size * err / first_den], dtype=float)
            dictionary_size_t.append(float(centers.shape[1]))
            weight_norm_t.append(float(np.linalg.norm(weights)))
            max_abs_weight_t.append(float(np.max(np.abs(weights))) if weights.size else 0.0)
            continue

        diffs = centers - uvec[:, None]
        dists = np.sum(diffs * diffs, axis=0)
        kappa = np.exp(-dists / two_sigma2)
        y = float(weights @ kappa)
        err = float(d_arr[idx] - y)
        out[idx] = err

        min_idx = int(np.argmin(dists))
        nearest_dist = float(dists[min_idx])
        compare_dist = nearest_dist / float(tap_dim) if mode == "normalized" else nearest_dist
        update_den = float(kappa @ kappa) + eps if normalized_update else 1.0
        update_value = step_size * err / update_den
        if compare_dist <= epsilon:
            weights[min_idx] += update_value
        else:
            if centers.shape[1] < max_dictionary_size:
                centers = np.concatenate([centers, uvec.reshape(-1, 1)], axis=1)
                weights = np.concatenate([weights, np.asarray([update_value])])
            else:
                max_dictionary_reached_count += 1
                if prune_policy == "freeze_new_centers":
                    weights[min_idx] += update_value
                else:
                    drop_idx = 0 if prune_policy == "prune_oldest" else int(np.argmin(np.abs(weights)))
                    centers = np.delete(centers, drop_idx, axis=1)
                    weights = np.delete(weights, drop_idx)
                    centers = np.concatenate([centers, uvec.reshape(-1, 1)], axis=1)
                    weights = np.concatenate([weights, np.asarray([update_value])])
                    prune_count += 1
        if not np.all(np.isfinite(weights)):
            # 中文说明：KLMS 字典权重一旦溢出，后续核预测会持续传播非有限值；
            # 这里直接回到有限状态，输出端仍会在函数结尾做最终 finite 清理。
            weights[~np.isfinite(weights)] = 0.0
        dictionary_size_t.append(float(centers.shape[1]))
        weight_norm_t.append(float(np.linalg.norm(weights)))
        max_abs_weight_t.append(float(np.max(np.abs(weights))) if weights.size else 0.0)

    out[~np.isfinite(out)] = 0.0
    if return_diagnostics:
        diagnostics = {
            "dictionary_size_t": np.asarray(dictionary_size_t, dtype=float),
            "weight_norm_t": np.asarray(weight_norm_t, dtype=float),
            "max_abs_weight_t": np.asarray(max_abs_weight_t, dtype=float),
            "prune_count": int(prune_count),
            "max_dictionary_reached_count": int(max_dictionary_reached_count),
        }
        diagnostics["weight_norm_t"][~np.isfinite(diagnostics["weight_norm_t"])] = 0.0
        diagnostics["max_abs_weight_t"][~np.isfinite(diagnostics["max_abs_weight_t"])] = 0.0
        return out, diagnostics
    return out


def _empty_diagnostics() -> dict[str, np.ndarray | int]:
    """Return empty KLMS diagnostics for degenerate windows."""

    return {
        "dictionary_size_t": np.asarray([], dtype=float),
        "weight_norm_t": np.asarray([], dtype=float),
        "max_abs_weight_t": np.asarray([], dtype=float),
        "prune_count": 0,
        "max_dictionary_reached_count": 0,
    }


def _zscore(x: np.ndarray) -> np.ndarray:
    """Return a finite z-scored copy of ``x``."""

    arr = np.asarray(x, dtype=float).copy()
    arr[~np.isfinite(arr)] = 0.0
    sd = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    if not np.isfinite(sd) or sd <= 1e-12:
        return arr - float(np.mean(arr))
    return (arr - float(np.mean(arr))) / sd
