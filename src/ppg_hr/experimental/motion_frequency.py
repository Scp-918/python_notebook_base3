"""Motion dominant-frequency estimation.

中文说明：
本模块在检测出的运动段内，从三轴 ACC 中选能量最大的一路，用
Hamming + FFT 估计运动干扰主频 Fmove。若运动段无效或谱峰不可用，
会给出 warning 并返回默认值，保证优化流程不中断。
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.signal.windows import hamming

from .segmentation import SegmentInfo

__all__ = ["estimate_motion_frequency"]


def estimate_motion_frequency(
    accx: np.ndarray,
    accy: np.ndarray,
    accz: np.ndarray,
    motion_segment: SegmentInfo,
    fs: int,
    *,
    default_hz: float = 1.0,
    min_hz: float = 0.2,
    max_hz: float = 5.0,
) -> float:
    """Return the dominant ACC frequency within the detected motion segment."""

    if not motion_segment.is_valid:
        warnings.warn(
            f"motion frequency fallback: {motion_segment.reason}",
            RuntimeWarning,
            stacklevel=2,
        )
        return float(default_hz)

    fs = int(fs)
    start = max(0, int(round(motion_segment.motion_start_s * fs)))
    end = min(len(accx), int(round(motion_segment.motion_end_s * fs)))
    if end - start < max(8, fs):
        warnings.warn("motion frequency fallback: motion segment too short", RuntimeWarning, stacklevel=2)
        return float(default_hz)

    axes = [
        np.asarray(accx, dtype=float)[start:end],
        np.asarray(accy, dtype=float)[start:end],
        np.asarray(accz, dtype=float)[start:end],
    ]
    # 中文注释：运动伪影通常在能量最大的一轴最明显，因此只用该轴估计主频。
    energies = [float(np.nansum((axis - np.nanmean(axis)) ** 2)) for axis in axes]
    sig = axes[int(np.argmax(energies))]
    sig = sig - np.nanmean(sig)
    sig = sig * hamming(sig.size, sym=False)
    freq = np.fft.rfftfreq(sig.size, d=1.0 / fs)
    amp = np.abs(np.fft.rfft(sig))
    mask = (freq >= min_hz) & (freq <= max_hz)
    if not mask.any() or not np.isfinite(amp[mask]).any():
        warnings.warn("motion frequency fallback: no spectral peak", RuntimeWarning, stacklevel=2)
        return float(default_hz)
    idx = np.flatnonzero(mask)[int(np.nanargmax(amp[mask]))]
    value = float(freq[idx])
    if not np.isfinite(value) or value <= 0:
        warnings.warn("motion frequency fallback: invalid peak", RuntimeWarning, stacklevel=2)
        return float(default_hz)
    return value
