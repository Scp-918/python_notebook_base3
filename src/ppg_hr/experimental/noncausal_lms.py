"""Causal/non-causal NLMS primitives for protocol cascade filtering.

中文说明：本模块把包络时延 D 映射为 LMS 阶数 M 与前向抽头 K，并实现输出长度
不变的非因果 NLMS。D > 0 表示补偿信号超前 PPG；D < 0 表示补偿信号滞后，
允许使用未来 K 个样本补偿。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .envelope_delay import ChannelDelay
from .tap_matrix import build_noncausal_tap_matrix

__all__ = ["LmsDesign", "map_delay_to_lms_params", "noncausal_lms_filter"]

LMS_MU_BASE = 0.01
LMS_MU_MIN = 1e-6


@dataclass(frozen=True)
class LmsDesign:
    """Mapped LMS order/forward-tap design for one sensor type."""

    M: int
    K: int
    u: float
    curr_corr: float
    mode: str
    sensor_type: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""

        return asdict(self)


def map_delay_to_lms_params(
    delay: ChannelDelay | int,
    sensor_type: str,
    search_params: Any,
    fs: int,
) -> LmsDesign:
    """Map an envelope delay in samples to LMS ``M``/``K`` parameters.

    中文说明：相关性越强，说明参考信号可能包含更强运动伪影，因此按
    ``mu = max(mu_min, LMS_Mu_Base - abs_corr / 100)`` 收缩步长。
    """

    del fs
    if isinstance(delay, ChannelDelay):
        D = int(delay.D_opt_samples)
        curr_corr = abs(float(delay.R_max))
    else:
        D = int(delay)
        curr_corr = 0.0

    max_order = int(_get_param(search_params, "max_order"))
    M_base = int(_get_param(search_params, "M_base"))
    C_scale = float(_get_param(search_params, "C_scale"))
    K_max = int(_get_param(search_params, "K_max"))

    if D > 0:
        M = min(max(1, int(np.floor(abs(D) * C_scale))), max_order)
        K = 0
        mode = "causal"
    elif D < 0:
        M = max(1, M_base)
        K = min(K_max, int(np.floor(abs(D) * C_scale)))
        mode = "noncausal"
    else:
        M = max(1, M_base)
        K = 0
        mode = "zero_delay"
    mu_base = float(_get_param(search_params, "LMS_Mu_Base", LMS_MU_BASE))
    mu_min = float(_get_param(search_params, "LMS_Mu_Min", LMS_MU_MIN))
    mu = max(mu_min, mu_base - curr_corr / 100.0)
    return LmsDesign(
        M=int(M),
        K=int(K),
        u=float(mu),
        curr_corr=float(curr_corr),
        mode=mode,
        sensor_type=str(sensor_type),
    )


def noncausal_lms_filter(
    u: np.ndarray,
    d: np.ndarray,
    M: int,
    K: int,
    mu: float,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
    """Filter ``d`` using reference ``u`` with ``K`` future taps.

    中文说明：输入向量从 ``u[n+K]`` 取到 ``u[n-M+1]``。边界处无法构造完整向量的
    样本保留归一化后的 ``d`` 值，保证输出长度与窗口完全一致。
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
        if return_diagnostics:
            return out, {"weight_norm_t": np.asarray([], dtype=float), "max_abs_weight_t": np.asarray([], dtype=float)}
        return out

    span = X.shape[1]
    w = np.zeros(span, dtype=float)
    mu = float(mu)
    eps = 1e-9
    weight_norm_t: list[float] = []
    max_abs_weight_t: list[float] = []
    # 中文注释：tap 矩阵已一次性构造；权重更新仍按时间递推，保持 NLMS 语义不变。
    for idx, uvec in zip(valid_indices, X, strict=True):
        y = float(np.dot(w, uvec))
        err = float(d_arr[idx] - y)
        out[idx] = err
        denom = float(np.dot(uvec, uvec) + eps)
        w += (mu / denom) * uvec * err
        weight_norm_t.append(float(np.linalg.norm(w)))
        max_abs_weight_t.append(float(np.max(np.abs(w))) if w.size else 0.0)
    if return_diagnostics:
        return out, {
            "weight_norm_t": np.asarray(weight_norm_t, dtype=float),
            "max_abs_weight_t": np.asarray(max_abs_weight_t, dtype=float),
        }
    return out


def _get_param(params: Any, name: str, default: Any = None) -> Any:
    if isinstance(params, dict):
        return params.get(name, default)
    return getattr(params, name, default)


def _zscore(x: np.ndarray) -> np.ndarray:
    """Return a finite z-scored copy of ``x``."""

    arr = np.asarray(x, dtype=float)
    arr = arr.copy()
    arr[~np.isfinite(arr)] = 0.0
    sd = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    if not np.isfinite(sd) or sd <= 1e-12:
        return arr - float(np.mean(arr))
    return (arr - float(np.mean(arr))) / sd
