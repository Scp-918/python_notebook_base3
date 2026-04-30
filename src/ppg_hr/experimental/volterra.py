"""Second-order non-causal Volterra adaptive filter for protocol windows.

中文说明：Volterra 滤波在普通 LMS 线性项外增加二阶交叉项。为避免维度爆炸，
二阶项只从距离当前样本最近的 M2 个 tap 构造上三角组合。
"""

from __future__ import annotations

import numpy as np

from .tap_matrix import build_noncausal_tap_matrix

__all__ = ["noncausal_volterra_filter"]


def noncausal_volterra_filter(
    u: np.ndarray,
    d: np.ndarray,
    M: int,
    K: int,
    mu1: float,
    alpha_u: float,
    M2: int,
    mu_min: float = 1e-5,
) -> np.ndarray:
    """Filter ``d`` with a linear + second-order Volterra adaptive model.

    中文说明：
    - 输入 ``u`` 是补偿参考，``d`` 是当前 PPG 残差信号；
    - 线性项使用完整 ``M + K`` 非因果向量；
    - 二次项只取最近 ``M2`` 个 tap，并构造 ``i <= j`` 的上三角组合；
    - ``mu2 = alpha_u * mu1``，两个步长都做 NaN/inf 安全处理。
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
    out = d_arr.copy()
    X, valid_indices = build_noncausal_tap_matrix(u_arr, M, K)
    if valid_indices.size == 0:
        return out

    mu1 = _safe_positive(mu1, mu_min)
    mu2 = _safe_positive(float(alpha_u) * mu1, mu_min * max(float(alpha_u), 1e-6))
    near_idx = _nearest_tap_indices(M, K, M2)
    q_count = int(len(near_idx) * (len(near_idx) + 1) / 2)
    if q_count:
        near = X[:, near_idx]
        tri_i, tri_j = np.triu_indices(len(near_idx))
        Q = near[:, tri_i] * near[:, tri_j]
    else:
        Q = np.empty((valid_indices.size, 0), dtype=float)

    span = X.shape[1]
    w1 = np.zeros(span, dtype=float)
    w2 = np.zeros(q_count, dtype=float)
    eps = 1e-9
    # 中文注释：二阶 Q 只在当前窗口/级联级内预计算，用完即释放，不跨 trial 缓存。
    for row, idx in enumerate(valid_indices):
        x = X[row]
        q = Q[row]
        y = float(np.dot(w1, x) + np.dot(w2, q))
        err = float(d_arr[idx] - y)
        out[idx] = err
        w1 += (mu1 / (float(np.dot(x, x)) + eps)) * x * err
        if q.size:
            w2 += (mu2 / (float(np.dot(q, q)) + eps)) * q * err
    del Q
    out[~np.isfinite(out)] = 0.0
    return out


def _nearest_tap_indices(M: int, K: int, M2: int) -> np.ndarray:
    """Return tap indices nearest to current sample n.

    中文说明：非因果向量顺序是 ``u[n+K], ..., u[n], ..., u[n-M+1]``，所以每个
    位置都有相对 offset。排序时先按 ``abs(offset)``，同距离优先当前/过去项，再
    选未来项。
    """

    offsets = list(range(K, -M, -1))
    indexed = list(enumerate(offsets))
    indexed.sort(key=lambda item: (abs(item[1]), 0 if item[1] <= 0 else 1, abs(item[1])))
    keep = indexed[: max(1, min(int(M2), len(indexed)))]
    keep.sort(key=lambda item: item[0])
    return np.asarray([idx for idx, _ in keep], dtype=int)


def _quadratic_features(x: np.ndarray) -> np.ndarray:
    """Build upper-triangular second-order products from one tap vector."""

    arr = np.asarray(x, dtype=float)
    feats: list[float] = []
    for i in range(arr.size):
        for j in range(i, arr.size):
            feats.append(float(arr[i] * arr[j]))
    return np.asarray(feats, dtype=float)


def _safe_positive(value: float, floor: float) -> float:
    if not np.isfinite(value):
        return float(floor)
    return max(float(floor), float(value))


def _zscore(x: np.ndarray) -> np.ndarray:
    """Return a finite z-scored copy of ``x``."""

    arr = np.asarray(x, dtype=float).copy()
    arr[~np.isfinite(arr)] = 0.0
    sd = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    if not np.isfinite(sd) or sd <= 1e-12:
        return arr - float(np.mean(arr))
    return (arr - float(np.mean(arr))) / sd
