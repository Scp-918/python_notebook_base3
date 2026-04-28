"""Trial execution for protocol cascade filtering and HR extraction.

中文说明：本模块执行“一组协议参数”的完整 trial：重采样、分段、对齐、估计
Fmove、逐窗归一化、包络时延、级联自适应滤波、频谱惩罚提取 HR，最后计算
目标段 AAE 和 accuracy。Optuna 每个 trial 都会调用这里。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.signal.windows import hamming

from ..core.find_near_biggest import find_near_biggest
from ..params import CascadeScheme, TargetScope
from ..preprocess.utils import smoothdata_movmedian
from .alignment import AlignedDataset, AlignmentInfo, align_ppg_to_ref_hr
from .envelope_delay import DelayEstimate, estimate_envelope_delays
from .motion_frequency import estimate_motion_frequency
from .noncausal_lms import map_delay_to_lms_params, noncausal_lms_filter
from .preprocess_protocol import PROTOCOL_CHANNELS, ProtocolDataset, resample_protocol_dataset
from .protocol_search_space import ProtocolTrialParams
from .rff_lms import noncausal_rff_lms_filter
from .segmentation import SegmentInfo, detect_activity_segments
from .volterra import noncausal_volterra_filter

__all__ = [
    "ProtocolRunResult",
    "extract_hr_with_penalty",
    "extract_plain_fft_hr",
    "run_protocol_trial",
]


@dataclass
class ProtocolRunResult:
    """Full result for one cascade scheme, target scope, and adaptive filter."""

    success: bool
    reason: str
    target_scope: TargetScope
    cascade_scheme: CascadeScheme
    adaptive_filter: str
    params: ProtocolTrialParams
    objective_aae_bpm: float
    baseline_aae_bpm: float
    adaptive_aae_bpm: float
    baseline_acc_pct: float
    adaptive_acc_pct: float
    frame: pd.DataFrame
    segment_info: SegmentInfo | None
    alignment_info: AlignmentInfo | None
    motion_frequency: float | None


@dataclass
class _TrialBase:
    """Cached trial base that depends only on Fs_Target and TW."""

    dataset: ProtocolDataset
    fs: int
    segment_info: SegmentInfo
    aligned: AlignedDataset | None
    motion_frequency: float | None
    failure_reason: str = ""


def run_protocol_trial(
    dataset: ProtocolDataset,
    cascade_scheme: CascadeScheme | str,
    target_scope: TargetScope | str,
    params: ProtocolTrialParams,
) -> ProtocolRunResult:
    """Run one concrete protocol trial and return AAE/accuracy metrics.

    中文说明：自适应滤波权重只在单个窗口内在线更新，不跨窗口、不跨文件保存。
    因此这里的“训练”本质是评估一组超参数，而不是拟合持久模型。
    """

    scheme = CascadeScheme(cascade_scheme)
    scope = TargetScope(target_scope)
    adaptive_filter = str(getattr(params, "adaptive_filter", "lms"))
    empty = pd.DataFrame()

    try:
        base = _get_trial_base(dataset, params)
        if base.failure_reason:
            return _failed(scope, scheme, params, base.failure_reason, base.segment_info)
        if base.aligned is None or base.motion_frequency is None:
            return _failed(scope, scheme, params, "cached trial base is incomplete", base.segment_info)

        frame = _run_windows(base.aligned, scheme, scope, params, base.motion_frequency)
        if frame.empty:
            return _failed(
                scope,
                scheme,
                params,
                "no valid windows after alignment",
                base.segment_info,
                base.aligned.alignment_info,
            )

        filtered_mask = _filtered_segment_mask(
            frame["segment_label"].to_numpy(dtype=str),
            scope,
            frame["time_s"].to_numpy(dtype=float),
            base.aligned.segment_info,
        )
        if not filtered_mask.any():
            return _failed(
                scope,
                scheme,
                params,
                "target scope contains no windows",
                base.segment_info,
                base.aligned.alignment_info,
            )

        frame["is_filtered_segment"] = filtered_mask
        adaptive = frame["adaptive_hr_bpm"].to_numpy(dtype=float)
        adaptive[filtered_mask] = smoothdata_movmedian(
            adaptive[filtered_mask],
            int(params.smooth_win_len),
        )
        frame["adaptive_hr_bpm"] = adaptive
        frame["baseline_abs_err_bpm"] = np.abs(
            frame["baseline_ppg_hr_bpm"].to_numpy(dtype=float)
            - frame["ref_hr_bpm"].to_numpy(dtype=float)
        )
        frame["adaptive_abs_err_bpm"] = np.abs(
            frame["adaptive_hr_bpm"].to_numpy(dtype=float)
            - frame["ref_hr_bpm"].to_numpy(dtype=float)
        )

        baseline_aae = _mean(frame.loc[filtered_mask, "baseline_abs_err_bpm"].to_numpy(dtype=float))
        adaptive_aae = _mean(frame.loc[filtered_mask, "adaptive_abs_err_bpm"].to_numpy(dtype=float))
        baseline_acc = _accuracy(frame.loc[filtered_mask, "baseline_abs_err_bpm"].to_numpy(dtype=float))
        adaptive_acc = _accuracy(frame.loc[filtered_mask, "adaptive_abs_err_bpm"].to_numpy(dtype=float))
        frame["baseline_aae_bpm"] = baseline_aae
        frame["baseline_acc_pct"] = baseline_acc
        frame["adaptive_aae_bpm"] = adaptive_aae
        frame["adaptive_acc_pct"] = adaptive_acc

        return ProtocolRunResult(
            success=True,
            reason="",
            target_scope=scope,
            cascade_scheme=scheme,
            adaptive_filter=adaptive_filter,
            params=params,
            objective_aae_bpm=adaptive_aae,
            baseline_aae_bpm=baseline_aae,
            adaptive_aae_bpm=adaptive_aae,
            baseline_acc_pct=baseline_acc,
            adaptive_acc_pct=adaptive_acc,
            frame=frame,
            segment_info=base.aligned.segment_info,
            alignment_info=base.aligned.alignment_info,
            motion_frequency=base.motion_frequency,
        )
    except Exception as exc:
        return ProtocolRunResult(
            success=False,
            reason=str(exc),
            target_scope=scope,
            cascade_scheme=scheme,
            adaptive_filter=adaptive_filter,
            params=params,
            objective_aae_bpm=float("inf"),
            baseline_aae_bpm=float("nan"),
            adaptive_aae_bpm=float("nan"),
            baseline_acc_pct=float("nan"),
            adaptive_acc_pct=float("nan"),
            frame=empty,
            segment_info=None,
            alignment_info=None,
            motion_frequency=None,
        )


def _get_trial_base(dataset: ProtocolDataset, params: ProtocolTrialParams) -> _TrialBase:
    """Return resampling/segmentation/alignment/Fmove cached by Fs_Target and TW.

    中文说明：重采样、分段、对齐和 Fmove 只依赖样本、Fs_Target 与 TW，不依赖级联
    方案或自适应滤波器，因此可复用缓存，减少每个 mode/trial 的固定开销。
    """

    fs_target = int(params.Fs_Target)
    tw = float(params.TW)
    key = (fs_target, tw, "signed_tdelay_edgepad_v1")
    cache = getattr(dataset, "_trial_base_cache", None)
    if cache is None:
        cache = {}
        setattr(dataset, "_trial_base_cache", cache)
    if key in cache:
        return cache[key]

    ds = resample_protocol_dataset(dataset, fs_target)
    fs = int(ds.fs)
    segment = detect_activity_segments(ds.accx, ds.accy, ds.accz, fs, params.TW)
    if not segment.is_valid:
        base = _TrialBase(
            dataset=ds,
            fs=fs,
            segment_info=segment,
            aligned=None,
            motion_frequency=None,
            failure_reason=f"segmentation failed: {segment.reason}",
        )
        cache[key] = base
        return base

    try:
        aligned = align_ppg_to_ref_hr(ds, segment, params.TW, fs)
    except Exception as exc:
        base = _TrialBase(
            dataset=ds,
            fs=fs,
            segment_info=segment,
            aligned=None,
            motion_frequency=None,
            failure_reason=f"alignment failed: {exc}",
        )
        cache[key] = base
        return base

    if aligned.ref_hr_bpm.size == 0:
        base = _TrialBase(
            dataset=ds,
            fs=fs,
            segment_info=segment,
            aligned=aligned,
            motion_frequency=None,
            failure_reason="alignment produced zero windows",
        )
        cache[key] = base
        return base

    fmove = estimate_motion_frequency(ds.accx, ds.accy, ds.accz, aligned.segment_info, fs)
    base = _TrialBase(
        dataset=ds,
        fs=fs,
        segment_info=aligned.segment_info,
        aligned=aligned,
        motion_frequency=fmove,
    )
    cache[key] = base
    return base


def extract_hr_with_penalty(
    filtered_ppg: np.ndarray,
    penalty_ref: np.ndarray,
    previous_hr: float | None,
    params: ProtocolTrialParams,
    fs: int,
) -> float:
    """Extract HR in BPM with motion-frequency spectral penalties and slew limits."""

    return _extract_hr(
        filtered_ppg,
        fs,
        previous_hr,
        params,
        enable_penalty=True,
        penalty_ref=penalty_ref,
    )


def extract_plain_fft_hr(
    ppg: np.ndarray,
    previous_hr: float | None,
    params: ProtocolTrialParams,
    fs: int,
) -> float:
    """Extract baseline HR from PPG Green without adaptive filtering or penalty."""

    return _extract_hr(ppg, fs, previous_hr, params, enable_penalty=False, penalty_ref=None)


def _run_windows(
    aligned: AlignedDataset,
    scheme: CascadeScheme,
    scope: TargetScope,
    params: ProtocolTrialParams,
    fmove: float,
) -> pd.DataFrame:
    """Run baseline and adaptive HR extraction for every aligned window."""

    ds = aligned.dataset
    fs = int(ds.fs)
    win_len = int(round(float(params.TW) * fs))
    channels = ds.channels()

    rows: list[dict[str, Any]] = []
    prev_baseline: float | None = None
    prev_adaptive: float | None = None
    for window_idx, (start_s, center_s, label, ref_hr) in enumerate(
        zip(
            aligned.window_starts_s,
            aligned.window_centers_s,
            aligned.segment_labels,
            aligned.ref_hr_bpm,
            strict=True,
        )
    ):
        start = int(round(float(start_s) * fs))
        end = start + win_len
        if end > len(ds.time_s):
            break
        window = {name: values[start:end] for name, values in channels.items()}
        norm = _normalise_window(window)
        baseline_hr = extract_plain_fft_hr(norm["ppg_green"], prev_baseline, params, fs)
        prev_baseline = baseline_hr if np.isfinite(baseline_hr) else prev_baseline

        if _window_in_scope(str(label), scope, float(center_s), aligned.segment_info):
            filtered, penalty_ref, adaptive_stages = _cascade_filter_window(norm, scheme, params, fmove, fs)
            adaptive_hr = extract_hr_with_penalty(filtered, penalty_ref, prev_adaptive, params, fs)
        else:
            adaptive_hr = baseline_hr
            adaptive_stages = []

        if np.isfinite(adaptive_hr):
            prev_adaptive = adaptive_hr
        stages_json = json.dumps(adaptive_stages, ensure_ascii=False)
        rows.append(
            {
                "sample": ds.sample_stem,
                "group_id": ds.sample_stem.removeprefix("multi_"),
                "target_scope": scope.value,
                "cascade_scheme": scheme.value,
                "adaptive_filter": str(getattr(params, "adaptive_filter", "lms")),
                "window_idx": window_idx,
                "time_s": float(center_s),
                "segment_label": str(label),
                "ref_hr_bpm": float(ref_hr),
                "baseline_ppg_hr_bpm": float(baseline_hr),
                "adaptive_hr_bpm": float(adaptive_hr),
                "adaptive_stages_json": stages_json,
                "lms_stages_json": stages_json,
            }
        )
    return pd.DataFrame(rows)


def _cascade_filter_window(
    window: dict[str, np.ndarray],
    scheme: CascadeScheme,
    params: ProtocolTrialParams,
    fmove: float,
    fs: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Run the configured adaptive-filter cascade on one normalized window."""

    delay_est = estimate_envelope_delays(window, fmove, params.Kstop, fs)
    current = np.asarray(window["ppg_green"], dtype=float)
    stages: list[dict[str, Any]] = []
    for sensor_type, max_count in _scheme_plan(scheme):
        for channel in delay_est.order_by_type.get(sensor_type, [])[:max_count]:
            delay = delay_est.by_channel[channel]
            design = map_delay_to_lms_params(delay, sensor_type, params, fs)
            filter_type = str(getattr(params, "adaptive_filter", "lms"))
            stage_extra: dict[str, Any] = {}
            if filter_type == "lms":
                current = noncausal_lms_filter(
                    window[channel],
                    current,
                    M=design.M,
                    K=design.K,
                    mu=design.u,
                )
            elif filter_type == "volterra":
                alpha_u = float(getattr(params, "alpha_u", 0.1))
                M2 = int(getattr(params, "M2", 3))
                current = noncausal_volterra_filter(
                    window[channel],
                    current,
                    M=design.M,
                    K=design.K,
                    mu1=design.u,
                    alpha_u=alpha_u,
                    M2=M2,
                    mu_min=float(getattr(params, "LMS_Mu_Min", 1e-6)),
                )
                stage_extra.update(
                    {
                        "alpha_u": alpha_u,
                        "M2": M2,
                        "mu1": float(design.u),
                        "mu2": float(alpha_u * design.u),
                    }
                )
            elif filter_type == "rff_lms":
                rff_D = int(getattr(params, "rff_D", 100))
                rff_sigma = float(getattr(params, "rff_sigma", 1.0))
                rff_seed = int(getattr(params, "rff_seed", 0))
                current = noncausal_rff_lms_filter(
                    window[channel],
                    current,
                    M=design.M,
                    K=design.K,
                    mu=design.u,
                    D=rff_D,
                    sigma=rff_sigma,
                    rff_seed=rff_seed,
                    mu_min=float(getattr(params, "LMS_Mu_Min", 1e-6)),
                )
                stage_extra.update({"D": rff_D, "sigma": rff_sigma, "rff_seed": rff_seed})
            else:
                raise ValueError(f"Unsupported adaptive_filter: {filter_type}")

            stages.append(
                {
                    "sensor_type": sensor_type,
                    "channel": channel,
                    "selected_channel": channel,
                    "D_opt_samples": int(delay.D_opt_samples),
                    "D_opt_seconds": float(delay.D_opt_seconds),
                    "R_max": float(delay.R_max),
                    "abs_corr": float(abs(delay.R_max)),
                    "curr_corr": float(design.curr_corr),
                    "M": int(design.M),
                    "K": int(design.K),
                    "mu": float(design.u),
                    "filter_type": filter_type,
                    "mode": design.mode,
                    **stage_extra,
                }
            )
    return current, _penalty_reference(window, delay_est, scheme), stages


def _scheme_plan(scheme: CascadeScheme) -> list[tuple[str, int]]:
    """Return the ordered sensor groups for a cascade scheme."""

    return {
        CascadeScheme.ACC3: [("ACC", 3)],
        CascadeScheme.HF2: [("HF", 2)],
        CascadeScheme.CF2: [("CF", 2)],
        CascadeScheme.HF2_CF2: [("HF", 2), ("CF", 2)],
        CascadeScheme.CF2_HF2: [("CF", 2), ("HF", 2)],
        CascadeScheme.ACC3_HF2: [("ACC", 3), ("HF", 2)],
        CascadeScheme.HF2_ACC3: [("HF", 2), ("ACC", 3)],
    }[scheme]


def _penalty_reference(
    window: dict[str, np.ndarray],
    delay_est: DelayEstimate,
    scheme: CascadeScheme,
) -> np.ndarray:
    """Choose the reference signal used for spectral motion penalties."""

    if scheme in {CascadeScheme.ACC3, CascadeScheme.ACC3_HF2, CascadeScheme.HF2_ACC3}:
        return window[_best_energy_channel(window, ("accx", "accy", "accz"))]
    if scheme in {CascadeScheme.HF2, CascadeScheme.HF2_CF2, CascadeScheme.CF2_HF2}:
        channel = delay_est.primary_by_type.get("HF") or "hf1"
        return window[channel]
    if scheme == CascadeScheme.CF2:
        channel = delay_est.primary_by_type.get("CF") or "cf1"
        return window[channel]
    return window["ppg_green"]


def _best_energy_channel(window: dict[str, np.ndarray], names: tuple[str, ...]) -> str:
    energies = [
        float(np.nansum((np.asarray(window[name], dtype=float) - np.nanmean(window[name])) ** 2))
        for name in names
    ]
    return names[int(np.argmax(energies))]


def _extract_hr(
    signal: np.ndarray,
    fs: int,
    previous_hr: float | None,
    params: ProtocolTrialParams,
    *,
    enable_penalty: bool,
    penalty_ref: np.ndarray | None,
) -> float:
    """Extract HR using Hamming FFT, optional spectral penalty, and slew limit."""

    freq, amp = _spectrum(signal, fs)
    band = (freq >= 0.5) & (freq <= 4.0)
    if not band.any():
        return float(previous_hr) if previous_hr is not None else float("nan")
    amp_work = amp.copy()

    if enable_penalty and penalty_ref is not None:
        motion_freq = _dominant_frequency(penalty_ref, fs, 0.2, 5.0)
        if np.isfinite(motion_freq) and motion_freq > 0:
            width = float(params.Spec_Penalty_Width)
            weight = float(params.Spec_Penalty_Weight)
            mask = (np.abs(freq - motion_freq) < width) | (
                np.abs(freq - 2.0 * motion_freq) < width
            )
            amp_work[mask] *= weight

    candidates = np.flatnonzero(band)
    if previous_hr is not None and np.isfinite(previous_hr) and previous_hr > 0:
        prev_hz = float(previous_hr) / 60.0
        restricted = candidates[np.abs(freq[candidates] - prev_hz) <= float(params.hr_range_hz)]
        if restricted.size:
            order = restricted[np.argsort(amp_work[restricted])[::-1]]
            near, which = find_near_biggest(
                freq[order],
                prev_hz,
                float(params.hr_range_hz),
                -float(params.hr_range_hz),
            )
            chosen_hz = float(near) if which else float(freq[order[0]])
        else:
            chosen_hz = prev_hz
        raw_bpm = chosen_hz * 60.0
        diff = raw_bpm - float(previous_hr)
        if diff > float(params.slew_limit_bpm):
            return float(previous_hr) + float(params.slew_step_bpm)
        if diff < -float(params.slew_limit_bpm):
            return float(previous_hr) - float(params.slew_step_bpm)
        return raw_bpm

    idx = candidates[int(np.argmax(amp_work[candidates]))]
    return float(freq[idx] * 60.0)


def _spectrum(signal: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    """Return Hamming-windowed FFT amplitude spectrum."""

    sig = np.asarray(signal, dtype=float)
    sig = sig.copy()
    sig[~np.isfinite(sig)] = 0.0
    if sig.size == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    sig = sig - float(np.mean(sig))
    sig = sig * hamming(sig.size, sym=False)
    nfft = max(8192, 1 << int(np.ceil(np.log2(max(sig.size, 1)))))
    freq = np.fft.rfftfreq(nfft, d=1.0 / fs)
    amp = np.abs(np.fft.rfft(sig, n=nfft))
    return freq, amp


def _dominant_frequency(signal: np.ndarray, fs: int, low_hz: float, high_hz: float) -> float:
    """Find the dominant spectral peak in a frequency band."""

    freq, amp = _spectrum(signal, fs)
    mask = (freq >= low_hz) & (freq <= high_hz)
    if not mask.any():
        return float("nan")
    peaks, _ = find_peaks(amp[mask])
    valid_idx = np.flatnonzero(mask)
    if peaks.size:
        idx = valid_idx[peaks[int(np.argmax(amp[valid_idx][peaks]))]]
    else:
        idx = valid_idx[int(np.argmax(amp[valid_idx]))]
    return float(freq[idx])


def _normalise_window(window: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Min-max normalize all 13 protocol channels inside one window."""

    out: dict[str, np.ndarray] = {}
    for name in PROTOCOL_CHANNELS:
        arr = np.asarray(window[name], dtype=float)
        arr = arr.copy()
        arr[~np.isfinite(arr)] = np.nan
        finite = np.isfinite(arr)
        if not finite.any():
            out[name] = np.zeros_like(arr, dtype=float)
            continue
        mn = float(np.nanmin(arr))
        mx = float(np.nanmax(arr))
        if mx - mn <= 1e-12:
            out[name] = np.zeros_like(arr, dtype=float)
        else:
            filled = arr.copy()
            filled[~finite] = mn
            out[name] = (filled - mn) / (mx - mn)
    return out


def _label_in_scope(label: str, scope: TargetScope) -> bool:
    if scope == TargetScope.MOTION_ONLY:
        return label == "motion"
    if scope == TargetScope.MOTION_POST10:
        return label == "motion"
    return label in {"motion", "recovery"}


def _window_in_scope(
    label: str,
    scope: TargetScope,
    center_s: float,
    segment_info: SegmentInfo | None,
) -> bool:
    """Return whether one window should run adaptive filtering.

    中文说明：``motion_post10`` 包含 motion 和运动结束后 10 秒；若文件不足 10 秒，
    已存在窗口自然只覆盖到文件末尾。
    """

    if scope == TargetScope.MOTION_POST10:
        if label == "motion":
            return True
        if segment_info is None or not np.isfinite(segment_info.motion_end_s):
            return False
        return float(segment_info.motion_end_s) < float(center_s) <= float(segment_info.motion_end_s) + 10.0
    return _label_in_scope(label, scope)


def _filtered_segment_mask(
    labels: np.ndarray,
    scope: TargetScope,
    centers_s: np.ndarray | None = None,
    segment_info: SegmentInfo | None = None,
) -> np.ndarray:
    """Build the boolean metric mask for a target scope."""

    if centers_s is None:
        centers_s = np.full(len(labels), np.nan, dtype=float)
    return np.asarray(
        [
            _window_in_scope(str(label), scope, float(center), segment_info)
            for label, center in zip(labels, centers_s, strict=False)
        ],
        dtype=bool,
    )


def _mean(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def _accuracy(abs_err: np.ndarray) -> float:
    """Return percent of windows whose absolute HR error is <= 5 bpm."""

    arr = np.asarray(abs_err, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr <= 5.0) * 100.0)


def _failed(
    scope: TargetScope,
    scheme: CascadeScheme,
    params: ProtocolTrialParams,
    reason: str,
    segment_info: SegmentInfo | None = None,
    alignment_info: AlignmentInfo | None = None,
) -> ProtocolRunResult:
    """Build a failed run result with stable metric fields."""

    return ProtocolRunResult(
        success=False,
        reason=reason,
        target_scope=scope,
        cascade_scheme=scheme,
        adaptive_filter=str(getattr(params, "adaptive_filter", "lms")),
        params=params,
        objective_aae_bpm=float("inf"),
        baseline_aae_bpm=float("nan"),
        adaptive_aae_bpm=float("nan"),
        baseline_acc_pct=float("nan"),
        adaptive_acc_pct=float("nan"),
        frame=pd.DataFrame(),
        segment_info=segment_info,
        alignment_info=alignment_info,
        motion_frequency=None,
    )
