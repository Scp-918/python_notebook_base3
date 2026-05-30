"""Random Fourier Feature LMS filter for non-causal protocol windows.

中文说明：RFF-LMS 用固定随机特征近似 RBF 非线性映射，然后在特征空间做 LMS。
随机特征由 ``rff_seed`` 固定；同一 trial、mode、repeat 和 random_state 会得到
完全一致的特征矩阵。
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .tap_matrix import build_noncausal_tap_matrix

__all__ = ["estimate_rff_sigma_base", "get_rff_weights", "noncausal_rff_lms_filter"]


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
    sigma_scale: float | None = None,
    min_sigma: float = 1e-6,
    mu_min: float = 1e-5,
    update_mode: str = "nlms",
    nlms_eps: float = 1e-9,
    leakage: float = 0.0,
    err_clip: float | None = None,
    theta_norm_guard: float | None = None,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray | int]]:
    """Filter ``d`` with fixed Random Fourier Features and LMS updates.

    中文说明：
    - 输入向量仍是完整 ``M + K`` 非因果向量；
    - ``W ~ Normal(0, 1/sigma^2)``，``b ~ Uniform(0, 2*pi)``；
    - 特征 ``z(x)=sqrt(2/D)*cos(W@x+b)``；
    - 默认更新 ``theta = (1-leakage)*theta + mu*e*z/(z@z+nlms_eps)``；
    - 新训练可传入 ``sigma_scale``，核宽会按当前 tap matrix 的 robust 距离自适应；
    - 如需复现实验旧逻辑，可显式传入 ``update_mode="lms"``，使用普通 LMS 更新；
    - ``err_clip`` 和 ``theta_norm_guard`` 是工程保护项，只在非 None 时生效；
    - ``return_diagnostics=True`` 时返回短诊断序列，默认仍只返回 ``out``。
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
    sigma = max(float(sigma), float(min_sigma), 1e-12)
    out = d_arr.copy()
    X, valid_indices = build_noncausal_tap_matrix(u_arr, M, K)
    if valid_indices.size == 0:
        if return_diagnostics:
            return out, _empty_diagnostics()
        return out

    sigma_base = estimate_rff_sigma_base(X, min_sigma=min_sigma)
    if sigma_scale is None:
        sigma_eff = sigma
        sigma_mode = "fixed_deprecated_rff_sigma"
    else:
        scale_value = float(sigma_scale) if np.isfinite(float(sigma_scale)) else 1.0
        sigma_eff = max(float(min_sigma), abs(scale_value) * float(sigma_base))
        sigma_mode = "adaptive"

    W, b = get_rff_weights(D, X.shape[1], sigma_eff, int(rff_seed))
    scale = float(np.sqrt(2.0 / D))
    # 中文注释：RFF 特征矩阵只在单窗口单级内批量计算，随后按时间顺序递推 theta。
    Z = scale * np.cos(X @ W.T + b)
    theta = np.zeros(D, dtype=float)
    mu = max(float(mu_min), float(mu) if np.isfinite(mu) else float(mu_min))
    mode = str(update_mode or "nlms").lower()
    if mode not in {"nlms", "lms"}:
        raise ValueError("update_mode must be 'nlms' or 'lms'")
    eps = float(nlms_eps) if np.isfinite(float(nlms_eps)) and float(nlms_eps) > 0.0 else 1e-9
    leak = float(leakage) if np.isfinite(float(leakage)) else 0.0
    leak = min(max(leak, 0.0), 1.0)
    clip_value = None if err_clip is None else abs(float(err_clip))
    if clip_value is not None and (not np.isfinite(clip_value) or clip_value <= 0.0):
        clip_value = None
    guard_value = None if theta_norm_guard is None else abs(float(theta_norm_guard))
    if guard_value is not None and (not np.isfinite(guard_value) or guard_value <= 0.0):
        guard_value = None
    theta_norm_t: list[float] = []
    max_abs_theta_t: list[float] = []
    guard_triggered_count = 0
    err_clip_count = 0

    for idx, z in zip(valid_indices, Z, strict=True):
        y = float(theta @ z)
        err = float(d_arr[idx] - y)
        if clip_value is not None:
            clipped = float(np.clip(err, -clip_value, clip_value))
            if clipped != err:
                err_clip_count += 1
            err = clipped
        out[idx] = err
        if mode == "nlms":
            z_energy = float(z @ z)
            theta = (1.0 - leak) * theta + (mu * err * z) / (z_energy + eps)
        else:
            theta = (1.0 - leak) * theta + mu * err * z
        if guard_value is not None:
            theta_norm = float(np.linalg.norm(theta))
            if np.isfinite(theta_norm) and theta_norm > guard_value:
                theta *= guard_value / (theta_norm + eps)
                guard_triggered_count += 1
            elif not np.isfinite(theta_norm):
                theta[:] = 0.0
                guard_triggered_count += 1
        elif not np.all(np.isfinite(theta)):
            # 中文说明：即使未启用显式范数阈值，也不能让非有限 theta 污染后续样本。
            theta[:] = 0.0
            guard_triggered_count += 1
        theta_norm_t.append(float(np.linalg.norm(theta)))
        max_abs_theta_t.append(float(np.max(np.abs(theta))) if theta.size else 0.0)
    del Z
    out[~np.isfinite(out)] = 0.0
    if return_diagnostics:
        diagnostics = {
            "theta_norm_t": np.asarray(theta_norm_t, dtype=float),
            "max_abs_theta_t": np.asarray(max_abs_theta_t, dtype=float),
            "guard_triggered_count": int(guard_triggered_count),
            "err_clip_count": int(err_clip_count),
            "sigma_base": float(sigma_base),
            "sigma_eff": float(sigma_eff),
            "sigma_mode": sigma_mode,
        }
        diagnostics["theta_norm_t"][~np.isfinite(diagnostics["theta_norm_t"])] = 0.0
        diagnostics["max_abs_theta_t"][~np.isfinite(diagnostics["max_abs_theta_t"])] = 0.0
        return out, diagnostics
    return out


def estimate_rff_sigma_base(X: np.ndarray, *, min_sigma: float = 1e-6, max_pairs: int = 2048) -> float:
    """Estimate a robust tap-space distance scale for adaptive RFF sigma."""

    mat = np.asarray(X, dtype=float)
    if mat.ndim != 2 or mat.shape[0] == 0:
        return float(min_sigma)
    finite_rows = np.isfinite(mat).all(axis=1)
    mat = mat[finite_rows]
    if mat.shape[0] == 0:
        return float(min_sigma)
    if mat.shape[0] >= 2:
        adjacent = np.linalg.norm(np.diff(mat, axis=0), axis=1)
        adjacent = adjacent[np.isfinite(adjacent) & (adjacent > 0.0)]
        if adjacent.size:
            return max(float(min_sigma), float(np.median(adjacent)))
    if mat.shape[0] >= 2:
        step = max(1, int(np.ceil(mat.shape[0] / max(2, min(int(max_pairs), mat.shape[0])))))
        sample = mat[::step]
        if sample.shape[0] > 1:
            diffs = sample[:, None, :] - sample[None, :, :]
            tri = np.triu_indices(sample.shape[0], k=1)
            dists = np.linalg.norm(diffs[tri], axis=1)
            dists = dists[np.isfinite(dists) & (dists > 0.0)]
            if dists.size:
                return max(float(min_sigma), float(np.median(dists)))
    spread = float(np.linalg.norm(np.nanstd(mat, axis=0)))
    if np.isfinite(spread) and spread > 0.0:
        return max(float(min_sigma), spread)
    return float(min_sigma)


def _empty_diagnostics() -> dict[str, np.ndarray | int]:
    """Return empty RFF diagnostics for degenerate windows."""

    return {
        "theta_norm_t": np.asarray([], dtype=float),
        "max_abs_theta_t": np.asarray([], dtype=float),
        "guard_triggered_count": 0,
        "err_clip_count": 0,
        "sigma_base": float("nan"),
        "sigma_eff": float("nan"),
        "sigma_mode": "empty",
    }


def _zscore(x: np.ndarray) -> np.ndarray:
    """Return a finite z-scored copy of ``x``."""

    arr = np.asarray(x, dtype=float).copy()
    arr[~np.isfinite(arr)] = 0.0
    sd = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    if not np.isfinite(sd) or sd <= 1e-12:
        return arr - float(np.mean(arr))
    return (arr - float(np.mean(arr))) / sd
