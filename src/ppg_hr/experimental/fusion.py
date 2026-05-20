"""Unsupervised deployment-style final HR fusion for protocol windows.

中文说明：本模块只使用 baseline/adaptive HR、分段标签、窗口时间和 QC 状态等
无监督信号，明确不读取参考 HR。参考 HR 只允许在外层指标计算时使用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["FinalFusionResult", "fuse_final_hr"]


@dataclass(frozen=True)
class FinalFusionResult:
    """Window-level final HR fusion arrays."""

    final_hr_bpm: np.ndarray
    final_source: np.ndarray
    fusion_reason: np.ndarray


def fuse_final_hr(
    *,
    time_s: np.ndarray,
    baseline_hr_bpm: np.ndarray,
    adaptive_hr_bpm: np.ndarray,
    segment_label: np.ndarray,
    qc_status: np.ndarray,
    adaptive_filter: str,
    motion_end_s: float,
    params: Any,
) -> FinalFusionResult:
    """Fuse baseline/adaptive HR without using reference HR.

    中文说明：
    - 静息段用 baseline FFT；
    - 运动段用 adaptive；
    - recovery 段先给 adaptive 一个 grace period，但若它和 baseline 分歧过大，
      或 grace 结束后仍未自然贴近 baseline，则回退 baseline；
    - QC 坏窗按窗口状态优先处理。
    """

    time = np.asarray(time_s, dtype=float)
    baseline = np.asarray(baseline_hr_bpm, dtype=float)
    adaptive = np.asarray(adaptive_hr_bpm, dtype=float)
    labels = np.asarray(segment_label, dtype=object).astype(str)
    qc = np.asarray(qc_status, dtype=object).astype(str)
    n = min(time.size, baseline.size, adaptive.size, labels.size, qc.size)
    final = np.full(n, np.nan, dtype=float)
    source = np.full(n, "", dtype=object)
    reason = np.full(n, "", dtype=object)
    adaptive_source = _adaptive_source(adaptive_filter)
    grace_s = float(getattr(params, "Recovery_Grace_S", 10.0))
    diff_bpm = float(getattr(params, "Recovery_Diff_Bpm", 20.0))
    cross_bpm = float(getattr(params, "Recovery_Cross_Diff_Bpm", 8.0))
    motion_end = float(motion_end_s)

    for i in range(n):
        base_hr = float(baseline[i])
        adapt_hr = float(adaptive[i])
        label = labels[i]
        qc_i = qc[i]
        if qc_i == "dropped":
            source[i] = "dropped"
            reason[i] = "qc_policy=drop"
            continue
        if qc_i == "interpolated":
            final[i] = adapt_hr if np.isfinite(adapt_hr) else base_hr
            source[i] = "interpolated"
            reason[i] = "qc_policy=interpolate"
            continue
        if qc_i in {"fallback_baseline", "qc_fallback_baseline"}:
            final[i] = base_hr
            source[i] = "qc_fallback_baseline"
            reason[i] = f"qc_status={qc_i}"
            continue
        if label == "rest":
            final[i] = base_hr
            source[i] = "baseline_fft"
            reason[i] = "rest_segment_baseline"
            continue
        if label == "motion":
            if np.isfinite(adapt_hr):
                final[i] = adapt_hr
                source[i] = adaptive_source
                reason[i] = "motion_segment_adaptive"
            else:
                final[i] = base_hr
                source[i] = "baseline_fft"
                reason[i] = "adaptive_not_finite"
            continue
        if label == "recovery":
            diff = abs(adapt_hr - base_hr) if np.isfinite(adapt_hr) and np.isfinite(base_hr) else float("inf")
            since_end = float(time[i] - motion_end) if np.isfinite(motion_end) else float("inf")
            if np.isfinite(adapt_hr) and since_end <= grace_s and diff <= diff_bpm:
                final[i] = adapt_hr
                source[i] = adaptive_source
                reason[i] = f"recovery_grace_s={grace_s:g}, diff_bpm={diff:.3f}"
            elif np.isfinite(adapt_hr) and diff <= cross_bpm:
                final[i] = adapt_hr
                source[i] = adaptive_source
                reason[i] = f"recovery_natural_cross_diff_bpm={diff:.3f}"
            else:
                final[i] = base_hr
                source[i] = "recovery_fallback"
                reason[i] = f"recovery_diff_bpm={diff:.3f}, grace_s={grace_s:g}"
            continue
        final[i] = adapt_hr if np.isfinite(adapt_hr) else base_hr
        source[i] = adaptive_source if np.isfinite(adapt_hr) else "baseline_fft"
        reason[i] = f"unknown_segment={label}"

    return FinalFusionResult(final, source, reason)


def _adaptive_source(adaptive_filter: str) -> str:
    name = str(adaptive_filter).lower()
    if name == "volterra":
        return "adaptive_volterra"
    if name == "rff_lms":
        return "adaptive_rff_lms"
    if name == "klms":
        return "adaptive_klms"
    return "adaptive_lms"
