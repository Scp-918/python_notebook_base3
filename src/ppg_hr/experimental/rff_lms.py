"""Random Fourier Feature LMS filter for non-causal protocol windows.

中文说明：RFF-LMS 用固定随机特征近似 RBF 非线性映射，然后在特征空间做 LMS。
随机特征由 ``rff_seed`` 固定；同一 trial、mode、repeat 和 random_state 会得到
完全一致的特征矩阵。
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .tap_matrix import build_noncausal_tap_matrix

__all__ = ["get_rff_weights", "noncausal_rff_lms_filter"]


@lru_cache(maxsize=64)
def get_rff_weights(D: int, span: int, sigma: float, rff_seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return cached Random Fourier Feature weights for one RFF layout.

    中文说明：``W`` 和 ``b`` 只由特征数 D、tap 长度 span、核宽 sigma 与随机种子决定。
    缓存这些只读数组可以避免每个窗口重复初始化随机数；key 中包含 span，防止
    不同 M/K 组合误用同一组随机特征。
    """

    D = max(1, int(D))
    span = max(1, int(span))
    sigma = max(float(sigma), 1e-6)
    rng = np.random.default_rng(int(rff_seed) % (2**32))
    W = rng.normal(loc=0.0, scale=1.0 / sigma, size=(D, span))
    b = rng.uniform(0.0, 2.0 * np.pi, size=D)
    W.setflags(write=False)
    b.setflags(write=False)
    return W, b


def noncausal_rff_lms_filter(
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
    """Filter ``d`` with fixed Random Fourier Features and LMS updates.

    中文说明：
    - 输入向量仍是完整 ``M + K`` 非因果向量；
    - ``W ~ Normal(0, 1/sigma^2)``，``b ~ Uniform(0, 2*pi)``；
    - 特征 ``z(x)=sqrt(2/D)*cos(W@x+b)``；
    - 更新 ``theta = theta + mu * e * z``；
    - 输出长度与输入窗口一致，边界样本保留归一化后的 ``d``。
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
    D = max(1, int(D))
    sigma = max(float(sigma), 1e-6)
    out = d_arr.copy()
    X, valid_indices = build_noncausal_tap_matrix(u_arr, M, K)
    if valid_indices.size == 0:
        return out

    W, b = get_rff_weights(D, X.shape[1], sigma, int(rff_seed))
    scale = float(np.sqrt(2.0 / D))
    # 中文注释：RFF 特征矩阵只在单窗口单级内批量计算，随后按时间顺序递推 theta。
    Z = scale * np.cos(X @ W.T + b)
    theta = np.zeros(D, dtype=float)
    mu = max(float(mu_min), float(mu) if np.isfinite(mu) else float(mu_min))

    for idx, z in zip(valid_indices, Z, strict=True):
        y = float(theta @ z)
        err = float(d_arr[idx] - y)
        out[idx] = err
        theta += mu * err * z
    del Z
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
