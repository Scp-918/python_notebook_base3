"""Causal/non-causal NLMS primitives for protocol cascade filtering.

中文说明：
本模块把包络时延 D 映射为 LMS 阶数 M 与前向抽头 K，并实现输出长度不变的
因果/非因果 NLMS。D>0 表示补偿信号超前，K=0；D<0 表示补偿信号滞后，
允许使用未来 K 个样本。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .envelope_delay import ChannelDelay

__all__ = ["LmsDesign", "map_delay_to_lms_params", "noncausal_lms_filter"]

LMS_MU_BASE = 0.01
LMS_MU_MIN = 1e-5


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
    """Map an envelope delay in samples to LMS ``M``/``K`` parameters."""

    del fs  # Delay is already in samples; keep ``fs`` for the public signature.
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
        # 中文注释：补偿信号超前 PPG，使用因果抽头，阶数随样本时延放大。
        M = min(max(1, int(np.floor(abs(D) * C_scale))), max_order)
        K = 0
        mode = "causal"
    elif D < 0:
        # 中文注释：补偿信号滞后 PPG，用 K 个未来样本补偿非因果前向信息。
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
) -> np.ndarray:
    """Filter ``d`` using reference ``u`` with ``K`` future taps.

    ``K=0`` uses only current/past reference samples. For ``K>0`` the feature
    vector spans ``u[n+K]`` down to ``u[n-M+1]``. Positions where that vector
    cannot be formed retain the normalised input ``d`` value, keeping output
    length identical to the input window.
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
    span = M + K
    out = d_arr.copy()
    if n - K < M:
        return out

    w = np.zeros(span, dtype=float)
    mu = float(mu)
    eps = 1e-9
    for idx in range(M - 1, n - K):
        # 中文注释：uvec 从未来 K 点一路取到过去 M-1 点；K=0 时自然退化为因果 LMS。
        uvec = u_arr[idx - M + 1 : idx + K + 1][::-1]
        y = float(np.dot(w, uvec))
        err = float(d_arr[idx] - y)
        out[idx] = err
        denom = float(np.dot(uvec, uvec) + eps)
        w += (mu / denom) * uvec * err
    return out


def _get_param(params: Any, name: str, default: Any = None) -> Any:
    if isinstance(params, dict):
        return params.get(name, default)
    return getattr(params, name, default)


def _zscore(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    arr = arr.copy()
    arr[~np.isfinite(arr)] = 0.0
    sd = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    if not np.isfinite(sd) or sd <= 1e-12:
        return arr - float(np.mean(arr))
    return (arr - float(np.mean(arr))) / sd
