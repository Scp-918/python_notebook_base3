"""Random Fourier Feature LMS filter for non-causal protocol windows.

中文说明：RFF-LMS 用固定随机特征近似 RBF 非线性映射，然后在特征空间做 LMS。
随机特征由 ``rff_seed`` 固定；同一 trial、mode、repeat 和 random_state 会得到
完全一致的特征矩阵。
"""

from __future__ import annotations

import numpy as np

__all__ = ["noncausal_rff_lms_filter"]


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
    P = M + K
    D = max(1, int(D))
    sigma = max(float(sigma), 1e-6)
    out = d_arr.copy()
    if n - K < M:
        return out

    rng = np.random.default_rng(int(rff_seed) % (2**32))
    W = rng.normal(loc=0.0, scale=1.0 / sigma, size=(D, P))
    b = rng.uniform(0.0, 2.0 * np.pi, size=D)
    scale = float(np.sqrt(2.0 / D))
    theta = np.zeros(D, dtype=float)
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


def _zscore(x: np.ndarray) -> np.ndarray:
    """Return a finite z-scored copy of ``x``."""

    arr = np.asarray(x, dtype=float).copy()
    arr[~np.isfinite(arr)] = 0.0
    sd = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    if not np.isfinite(sd) or sd <= 1e-12:
        return arr - float(np.mean(arr))
    return (arr - float(np.mean(arr))) / sd
