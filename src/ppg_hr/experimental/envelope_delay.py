"""Envelope cross-correlation delay estimation for LMS design."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.signal import butter, filtfilt, hilbert

__all__ = ["ChannelDelay", "DelayEstimate", "estimate_envelope_delays"]

# 中文说明：本文件按 HF/CF/ACC 三类补偿信号分别估计相对 PPG 的包络时延。
# D_opt_samples > 0 表示补偿信号超前 PPG，可用因果 LMS；D_opt_samples < 0
# 表示补偿信号滞后，需要非因果前向抽头。

_GROUPS = {
    "HF": ("hf1", "hf2"),
    "CF": ("cf1", "cf2"),
    "ACC": ("accx", "accy", "accz"),
}


@dataclass(frozen=True)
class ChannelDelay:
    """Delay and envelope-correlation result for one compensation channel."""

    channel: str
    sensor_type: str
    D_opt_samples: int
    D_opt_seconds: float
    R_max: float
    abs_corr: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""

        return asdict(self)


@dataclass(frozen=True)
class DelayEstimate:
    """Per-window channel ranking by envelope delay/correlation."""

    by_channel: dict[str, ChannelDelay]
    order_by_type: dict[str, list[str]]
    primary_by_type: dict[str, str | None]
    mode: str = "envelope"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""

        return {
            "by_channel": {k: v.to_dict() for k, v in self.by_channel.items()},
            "order_by_type": self.order_by_type,
            "primary_by_type": self.primary_by_type,
            "mode": self.mode,
        }


def estimate_envelope_delays(
    window_signals: dict[str, np.ndarray],
    Fmove: float,
    Kstop: float,
    fs: int,
    mode: str = "envelope",
) -> DelayEstimate:
    """Estimate channel delays by envelope or direct correlation against PPG.

    Positive ``D_opt_samples`` means the compensation channel leads PPG and can
    be used causally. Negative values imply a non-causal forward-tap design.

    中文说明：``mode='envelope'`` 完全保留旧逻辑：Hilbert 包络、低通后做 Pearson
    相关搜索；``mode='direct'`` 直接对窗口内已带通/归一化的原始波形做 z-score 和
    Pearson 相关搜索，不计算包络，也不额外低通。
    """

    if "ppg_green" not in window_signals:
        raise KeyError("window_signals must include ppg_green")
    fs = int(fs)
    mode = str(mode).lower()
    if mode not in {"envelope", "direct"}:
        raise ValueError("delay_estimation mode must be 'envelope' or 'direct'")
    cutoff = float(np.clip(float(Kstop) * float(Fmove), 0.05, 0.45 * fs))
    if mode == "envelope":
        # 中文注释：包络低通截止频率由 Kstop * Fmove 决定，并限制在滤波器稳定范围内。
        ppg_ref = _envelope(window_signals["ppg_green"], fs, cutoff)
    else:
        # 中文注释：direct 模式直接比较原始波形形状，只做有限值修复和 z-score。
        ppg_ref = _direct_signal(window_signals["ppg_green"])
    max_lag = int(round(0.5 * fs))

    by_channel: dict[str, ChannelDelay] = {}
    order_by_type: dict[str, list[str]] = {}
    primary_by_type: dict[str, str | None] = {}

    for sensor_type, channels in _GROUPS.items():
        ranked: list[ChannelDelay] = []
        for channel in channels:
            if channel not in window_signals:
                continue
            # 中文注释：每一路独立搜索最优延迟，最后按相关性强弱排序。
            comp_ref = (
                _envelope(window_signals[channel], fs, cutoff)
                if mode == "envelope"
                else _direct_signal(window_signals[channel])
            )
            delay, corr = _best_delay(ppg_ref, comp_ref, max_lag)
            item = ChannelDelay(
                channel=channel,
                sensor_type=sensor_type,
                D_opt_samples=int(delay),
                D_opt_seconds=float(delay / fs),
                R_max=float(corr),
                abs_corr=float(abs(corr)),
            )
            by_channel[channel] = item
            ranked.append(item)
        ranked.sort(key=lambda x: abs(x.R_max), reverse=True)
        order_by_type[sensor_type] = [x.channel for x in ranked]
        primary_by_type[sensor_type] = ranked[0].channel if ranked else None

    return DelayEstimate(
        by_channel=by_channel,
        order_by_type=order_by_type,
        primary_by_type=primary_by_type,
        mode=mode,
    )


def _envelope(x: np.ndarray, fs: int, cutoff: float) -> np.ndarray:
    sig = np.asarray(x, dtype=float)
    if sig.size == 0:
        return sig.copy()
    sig = sig - np.nanmean(sig)
    sig[~np.isfinite(sig)] = 0.0
    env = np.abs(hilbert(sig)) if sig.size >= 4 else np.abs(sig)
    nyq = fs / 2.0
    cutoff = min(max(float(cutoff), 1e-3), 0.95 * nyq)
    try:
        b, a = butter(2, cutoff / nyq, btype="lowpass")
        if env.size > 3 * max(len(a), len(b)):
            env = filtfilt(b, a, env)
    except ValueError:
        pass
    return _zscore(env)


def _direct_signal(x: np.ndarray) -> np.ndarray:
    """Return the direct-correlation signal used by ``mode='direct'``.

    中文说明：调用方传入的窗口已经按协议完成带通和 min-max 归一化，这里只负责
    修复 NaN/inf 并统一成 z-score，保证与包络模式使用同一 Pearson 搜索函数。
    """

    sig = np.asarray(x, dtype=float).copy()
    if sig.size == 0:
        return sig
    sig[~np.isfinite(sig)] = 0.0
    return _zscore(sig)


def _best_delay(ppg_env: np.ndarray, comp_env: np.ndarray, max_lag: int) -> tuple[int, float]:
    best_delay = 0
    best_corr = 0.0
    for delay in range(-max_lag, max_lag + 1):
        a, b = _aligned_for_delay(ppg_env, comp_env, delay)
        if a.size < 3:
            continue
        corr = _pearson(a, b)
        if abs(corr) > abs(best_corr):
            best_delay = delay
            best_corr = corr
    return best_delay, best_corr


def _aligned_for_delay(
    ppg_env: np.ndarray,
    comp_env: np.ndarray,
    delay: int,
) -> tuple[np.ndarray, np.ndarray]:
    if delay > 0:
        return ppg_env[delay:], comp_env[:-delay]
    if delay < 0:
        k = -delay
        return ppg_env[:-k], comp_env[k:]
    return ppg_env, comp_env


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size != b.size or a.size < 3:
        return 0.0
    aa = _zscore(a)
    bb = _zscore(b)
    denom = float(np.sqrt(np.dot(aa, aa)) * np.sqrt(np.dot(bb, bb)))
    if denom <= 1e-12 or not np.isfinite(denom):
        return 0.0
    return float(np.dot(aa, bb) / denom)


def _zscore(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    mu = float(np.nanmean(arr)) if arr.size else 0.0
    sd = float(np.nanstd(arr))
    if not np.isfinite(sd) or sd <= 1e-12:
        return arr - mu
    return (arr - mu) / sd

