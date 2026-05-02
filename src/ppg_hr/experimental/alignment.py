"""Signed PPG-to-reference HR delay alignment for the protocol.

中文说明：本模块把“全局 Tdelay 估计”和“后续训练窗口构建”拆成两步。
全局 Tdelay 只使用固定的 ``alignment_TW``（默认 8 s）在静息段提取 tracked
PPG-HR 并搜索时延；后续自适应滤波训练/验证/测试窗口继续使用 trial 的
``train_TW`` / ``params.TW``。二者物理含义不同，不能混用。

Signed Tdelay 的通道平移语义继续保持旧逻辑：delay_s > 0 时传感器侧左移；
delay_s < 0 时用首样本补头并右移；delay_s == 0 时不移动。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from threading import Lock
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.signal.windows import hamming

from ..core.find_near_biggest import find_near_biggest
from ..preprocess.utils import smoothdata_movmedian
from .preprocess_protocol import ProtocolDataset
from .segmentation import SegmentInfo

__all__ = [
    "AlignedDataset",
    "AlignmentInfo",
    "RestHrResult",
    "TDelayEstimateResult",
    "align_ppg_to_ref_hr",
    "apply_rest_hr_slew_limit",
    "build_aligned_training_windows",
    "compute_rest_alignment_diagnostic_curve",
    "estimate_global_tdelay_from_rest",
    "extract_rest_ppg_hr_tracked",
    "smooth_rest_hr_sequence",
]

_PRINTED_ALIGNMENT_SAMPLE_STEMS: set[str] = set()
_PRINTED_ALIGNMENT_LOCK = Lock()
_REST_ALIGNMENT_SCORE_START_S = 5.0
DEFAULT_ALIGNMENT_TW = 8.0
DEFAULT_ALIGNMENT_STEP_S = 1.0
DEFAULT_REST_HR_BAND_BPM = (40.0, 180.0)
DEFAULT_REST_HR_TRACK_BAND_BPM = 30.0
DEFAULT_REST_HR_SLEW_LIMIT_BPM = 6.0
DEFAULT_REST_HR_SLEW_STEP_BPM = 4.0
DEFAULT_REST_HR_SMOOTH_METHOD = "median"
DEFAULT_REST_HR_SMOOTH_WIN = 3
DEFAULT_REST_HR_PEAK_PERCENT = 0.3
DEFAULT_REST_SPEC_PENALTY_ENABLE = True
DEFAULT_REST_SPEC_PENALTY_WEIGHT = 0.2
DEFAULT_REST_SPEC_PENALTY_WIDTH_HZ = 0.2


@dataclass(frozen=True)
class RestHrResult:
    """Tracked rest-segment PPG HR curve used for global Tdelay scoring."""

    times_s: np.ndarray
    hr_bpm_raw: np.ndarray
    hr_bpm_tracked: np.ndarray
    hr_bpm_smooth: np.ndarray
    quality: dict[str, Any] | None = None


@dataclass(frozen=True)
class TDelayEstimateResult:
    """Global Tdelay estimate and per-candidate rest-alignment score table."""

    best_tdelay_s: float
    best_score: float
    score_table: pd.DataFrame
    rest_ppg_hr: RestHrResult
    debug: dict[str, Any] | None = None


@dataclass(frozen=True)
class AlignmentInfo:
    """Chosen delay and per-candidate alignment diagnostics."""

    best_tdelay_s: float
    std_by_delay: dict[float, float]
    ref_shift_s: float
    num_windows: int
    status: str = "ok"
    reason: str = ""
    alignment_tw_s: float = DEFAULT_ALIGNMENT_TW
    train_tw_s: float = 0.0
    best_score: float = float("nan")
    n_valid_score_windows: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""

        return {
            "best_tdelay_s": float(self.best_tdelay_s),
            "std_by_delay": {str(k): float(v) for k, v in self.std_by_delay.items()},
            "ref_shift_s": float(self.ref_shift_s),
            "num_windows": int(self.num_windows),
            "status": self.status,
            "reason": self.reason,
            "alignment_tw_s": float(self.alignment_tw_s),
            "train_tw_s": float(self.train_tw_s),
            "best_score": float(self.best_score),
            "n_valid_score_windows": int(self.n_valid_score_windows),
        }


@dataclass(frozen=True)
class AlignedDataset:
    """Dataset after sensor left-shift and reference half-window alignment."""

    dataset: ProtocolDataset
    segment_info: SegmentInfo
    alignment_info: AlignmentInfo
    window_starts_s: np.ndarray
    window_centers_s: np.ndarray
    segment_labels: np.ndarray
    ref_hr_bpm: np.ndarray
    rest_indices: np.ndarray
    motion_indices: np.ndarray
    recovery_indices: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        """Return a compact JSON-friendly metadata representation."""

        return {
            "alignment_info": self.alignment_info.to_dict(),
            "segment_info": self.segment_info.to_dict(),
            "rest_indices": self.rest_indices.tolist(),
            "motion_indices": self.motion_indices.tolist(),
            "recovery_indices": self.recovery_indices.tolist(),
        }


def align_ppg_to_ref_hr(
    dataset: ProtocolDataset,
    segment_info: SegmentInfo,
    TW: int | float,
    fs_target: int,
    *,
    alignment_TW: float | None = None,
    alignment_step_s: float = DEFAULT_ALIGNMENT_STEP_S,
    delay_range_s: tuple[float, float] | None = None,
    delay_step_s: float | None = None,
    rest_hr_kwargs: dict[str, Any] | None = None,
    use_tracked_rest_hr: bool = True,
    return_debug: bool = False,
) -> AlignedDataset:
    """Find global signed PPG delay and build training-window metadata.

    中文说明：该兼容 wrapper 保留原调用 ``align_ppg_to_ref_hr(ds, seg, TW, fs)``。
    这里的 ``TW`` 只表示后续自适应滤波训练/验证/测试切窗的 ``train_TW``；
    全局 Tdelay 估计使用独立的 ``alignment_TW``，默认 8 秒，不参与贝叶斯优化。
    """

    if not segment_info.is_valid:
        raise ValueError(f"Cannot align invalid segment info: {segment_info.reason}")

    fs = int(fs_target)
    train_TW = float(TW)
    align_TW = DEFAULT_ALIGNMENT_TW if alignment_TW is None else float(alignment_TW)
    if not use_tracked_rest_hr:
        rest_hr_kwargs = {**(rest_hr_kwargs or {}), "smooth_method": "none"}
    tdelay = estimate_global_tdelay_from_rest(
        dataset,
        segment_info,
        fs,
        alignment_TW=align_TW,
        alignment_step_s=float(alignment_step_s),
        delay_range_s=delay_range_s,
        delay_step_s=delay_step_s,
        rest_hr_kwargs=rest_hr_kwargs,
        return_debug=return_debug,
    )
    _print_alignment_once(
        dataset.sample_stem,
        tdelay.best_tdelay_s,
        int(tdelay.score_table["n_valid"].max()) if not tdelay.score_table.empty else 0,
    )
    return build_aligned_training_windows(
        dataset,
        segment_info,
        fs,
        train_TW=train_TW,
        best_tdelay_s=tdelay.best_tdelay_s,
        tdelay_result=tdelay,
    )


def extract_rest_ppg_hr_tracked(
    ppg_signal: np.ndarray,
    fs: float,
    tw_s: float = DEFAULT_ALIGNMENT_TW,
    step_s: float = DEFAULT_ALIGNMENT_STEP_S,
    hr_band_bpm: tuple[float, float] = DEFAULT_REST_HR_BAND_BPM,
    init_strategy: str = "max_peak",
    track_band_bpm: float = DEFAULT_REST_HR_TRACK_BAND_BPM,
    slew_limit_bpm: float = DEFAULT_REST_HR_SLEW_LIMIT_BPM,
    slew_step_bpm: float = DEFAULT_REST_HR_SLEW_STEP_BPM,
    smooth_method: str = DEFAULT_REST_HR_SMOOTH_METHOD,
    smooth_win: int = DEFAULT_REST_HR_SMOOTH_WIN,
    peak_percent: float = DEFAULT_REST_HR_PEAK_PERCENT,
    penalty_signal: np.ndarray | None = None,
    spec_penalty_enable: bool = DEFAULT_REST_SPEC_PENALTY_ENABLE,
    spec_penalty_weight: float = DEFAULT_REST_SPEC_PENALTY_WEIGHT,
    spec_penalty_width_hz: float = DEFAULT_REST_SPEC_PENALTY_WIDTH_HZ,
    return_debug: bool = False,
) -> RestHrResult:
    """Extract a reference-style tracked rest-segment PPG HR curve.

    中文说明：该函数专用于全局 Tdelay 估计和静息段诊断。它复刻参考仓库
    ``Helper_Process_Spectrum`` 的静息 FFT 后处理思路：每个窗口先做去均值 +
    Hamming，再找超过主峰一定比例的局部谱峰；谱峰按幅值排序后，只在前 5 个
    候选中寻找靠近上一 HR 的峰，随后用 slew limit/step 限制异常跳变，最后做
    moving median 平滑。

    运动惩罚说明：参考仓库的谱惩罚依赖独立的运动参考通道。本项目默认开启
    ``spec_penalty_enable`` 以贴近参考实现，但只有调用方传入 ``penalty_signal``
    时才会真正生效；生效后会把运动主频及二倍频附近的 PPG 候选谱峰幅值乘以
    ``spec_penalty_weight``。
    """

    fs = float(fs)
    win_len = int(round(float(tw_s) * fs))
    step_len = max(1, int(round(float(step_s) * fs)))
    sig = np.asarray(ppg_signal, dtype=float).ravel()
    if win_len < 4 or sig.size < win_len:
        empty = np.asarray([], dtype=float)
        return RestHrResult(empty, empty, empty, empty, {"reason": "signal shorter than one rest HR window"})

    starts = np.arange(0, sig.size - win_len + 1, step_len, dtype=int)
    times_s = starts.astype(float) / fs + float(tw_s) / 2.0
    raw_hr = np.full(starts.size, np.nan, dtype=float)
    tracked_hr = np.full(starts.size, np.nan, dtype=float)
    peak_amp = np.full(starts.size, np.nan, dtype=float)
    which_peak = np.zeros(starts.size, dtype=int)
    used_fallback = np.zeros(starts.size, dtype=bool)
    slew_limited = np.zeros(starts.size, dtype=bool)
    penalty_applied = np.zeros(starts.size, dtype=bool)
    penalty_freq_hz = np.full(starts.size, np.nan, dtype=float)
    penalty_arr = None if penalty_signal is None else np.asarray(penalty_signal, dtype=float).ravel()

    prev_hr: float | None = None
    for row, start in enumerate(starts):
        penalty_window = None
        if penalty_arr is not None and start + win_len <= penalty_arr.size:
            penalty_window = penalty_arr[start : start + win_len]
        peak_info = _rest_spectrum_candidates_like_reference(
            sig[start : start + win_len],
            int(round(fs)),
            hr_band_bpm,
            peak_percent=float(peak_percent),
            penalty_window=penalty_window,
            spec_penalty_enable=bool(spec_penalty_enable),
            spec_penalty_weight=float(spec_penalty_weight),
            spec_penalty_width_hz=float(spec_penalty_width_hz),
        )
        fre_hz = peak_info["freq_hz"]
        amp = peak_info["amp"]
        if fre_hz.size == 0:
            continue
        raw = float(fre_hz[0] * 60.0)
        raw_hr[row] = raw
        peak_amp[row] = float(amp[0])
        penalty_applied[row] = bool(peak_info["penalty_applied"])
        penalty_freq_hz[row] = float(peak_info["penalty_freq_hz"])
        if prev_hr is None or not np.isfinite(prev_hr):
            candidate = raw if init_strategy == "max_peak" else raw
            which_peak[row] = 1
        else:
            # 中文说明：参考实现的 find_near_biggest 只检查按幅值排序后的前 5 个峰；
            # 找不到邻近峰时保持上一 HR，比直接回退全局最大峰更不容易被倍频带走。
            candidate_hz, which = find_near_biggest(
                fre_hz,
                float(prev_hr) / 60.0,
                float(track_band_bpm) / 60.0,
                -float(track_band_bpm) / 60.0,
            )
            candidate = float(candidate_hz * 60.0)
            which_peak[row] = int(which)
            if which == 0:
                used_fallback[row] = True
        limited = _limit_hr_transition(prev_hr, candidate, slew_limit_bpm, slew_step_bpm)
        slew_limited[row] = bool(prev_hr is not None and np.isfinite(prev_hr) and abs(candidate - prev_hr) > slew_limit_bpm)
        tracked_hr[row] = limited
        prev_hr = limited if np.isfinite(limited) else prev_hr

    smooth_hr = smooth_rest_hr_sequence(tracked_hr, method=smooth_method, smooth_win=smooth_win)
    quality = None
    if return_debug:
        quality = {
            "peak_amp": peak_amp,
            "which_peak": which_peak,
            "used_fallback": used_fallback,
            "slew_limited": slew_limited,
            "penalty_applied": penalty_applied,
            "penalty_freq_hz": penalty_freq_hz,
            "tw_s": float(tw_s),
            "step_s": float(step_s),
            "hr_band_bpm": tuple(float(x) for x in hr_band_bpm),
            "track_band_bpm": float(track_band_bpm),
            "peak_percent": float(peak_percent),
            "spec_penalty_enable": bool(spec_penalty_enable),
            "spec_penalty_weight": float(spec_penalty_weight),
            "spec_penalty_width_hz": float(spec_penalty_width_hz),
        }
    return RestHrResult(
        times_s=times_s,
        hr_bpm_raw=raw_hr,
        hr_bpm_tracked=tracked_hr,
        hr_bpm_smooth=smooth_hr,
        quality=quality,
    )


def apply_rest_hr_slew_limit(
    hr_bpm: np.ndarray,
    slew_limit_bpm: float = DEFAULT_REST_HR_SLEW_LIMIT_BPM,
    slew_step_bpm: float = DEFAULT_REST_HR_SLEW_STEP_BPM,
) -> np.ndarray:
    """Apply the rest-HR maximum transition rule to an HR sequence.

    中文说明：测试和诊断可直接调用该函数验证防跳峰逻辑；真实频谱提取时还会先在
    上一 HR 邻域内选择候选峰，然后再调用同一个限制规则。
    """

    arr = np.asarray(hr_bpm, dtype=float)
    out = np.full(arr.size, np.nan, dtype=float)
    prev: float | None = None
    for idx, value in enumerate(arr):
        out[idx] = _limit_hr_transition(prev, float(value), slew_limit_bpm, slew_step_bpm)
        if np.isfinite(out[idx]):
            prev = float(out[idx])
    return out


def smooth_rest_hr_sequence(
    hr_bpm: np.ndarray,
    method: str = DEFAULT_REST_HR_SMOOTH_METHOD,
    smooth_win: int = DEFAULT_REST_HR_SMOOTH_WIN,
) -> np.ndarray:
    """Smooth rest HR with a short moving median, safely handling short arrays.

    中文说明：默认只做 3 点移动中位数，用于压制孤立跳峰；有效点过少、窗口过短
    或 smooth_method="none" 时直接返回原序列副本，不中断 notebook 批处理。
    """

    arr = np.asarray(hr_bpm, dtype=float)
    if arr.size == 0:
        return arr.copy()
    method = str(method).lower()
    if method in {"", "none", "off"}:
        return arr.copy()
    if method != "median":
        raise ValueError(f"Unsupported rest HR smooth_method: {method}")
    win = int(smooth_win)
    if win <= 1 or arr.size < 3:
        return arr.copy()
    if win % 2 == 0:
        win += 1
    win = min(win, arr.size if arr.size % 2 == 1 else arr.size - 1)
    if win <= 1:
        return arr.copy()
    return smoothdata_movmedian(arr, win)


def estimate_global_tdelay_from_rest(
    ds: ProtocolDataset,
    segment: SegmentInfo,
    fs: float,
    alignment_TW: float = DEFAULT_ALIGNMENT_TW,
    alignment_step_s: float = DEFAULT_ALIGNMENT_STEP_S,
    delay_range_s: tuple[float, float] | None = None,
    delay_step_s: float | None = None,
    rest_hr_kwargs: dict[str, Any] | None = None,
    return_debug: bool = False,
) -> TDelayEstimateResult:
    """Estimate global signed Tdelay from tracked rest-segment PPG HR.

    中文说明：本函数只负责全局 Tdelay 搜索，使用固定 ``alignment_TW``；它不构建
    后续训练窗口，也不读取 Optuna trial 的 ``params.TW``。候选 delay 的正负号
    和边界补值继续复用当前项目的 signed Tdelay 逻辑。
    """

    if not segment.is_valid:
        raise ValueError(f"Cannot estimate Tdelay for invalid segment info: {segment.reason}")
    fs_i = int(round(float(fs)))
    align_tw = float(alignment_TW)
    neg_limit = min(5.0, align_tw / 2.0)
    if delay_range_s is None:
        delay_range_s = (-neg_limit, 5.0)
    step = 0.1 if delay_step_s is None else float(delay_step_s)
    delay_grid = np.round(np.arange(float(delay_range_s[0]), float(delay_range_s[1]) + 1e-9, step), 10)
    kwargs = dict(rest_hr_kwargs or {})
    kwargs.setdefault("tw_s", align_tw)
    kwargs.setdefault("step_s", float(alignment_step_s))
    kwargs.setdefault("return_debug", return_debug)

    rows: list[dict[str, Any]] = []
    best_delay = 0.0
    best_score = float("inf")
    best_rest: RestHrResult | None = None
    best_ref = np.asarray([], dtype=float)
    for delay_s in delay_grid:
        rest = _rest_ppg_hr_tracked_for_delay(
            ds.ppg_green,
            fs_i,
            delay_s=float(delay_s),
            motion_start_s=float(segment.motion_start_s),
            score_start_s=_REST_ALIGNMENT_SCORE_START_S,
            rest_hr_kwargs=kwargs,
            penalty_signal=ds.accz,
        )
        ref_hr = _reference_hr_for_alignment_times(ds, rest.times_s, align_tw)
        ppg_hr = np.asarray(rest.hr_bpm_smooth, dtype=float)
        valid = np.isfinite(ppg_hr) & np.isfinite(ref_hr)
        n_valid = int(valid.sum())
        if n_valid < 2:
            score = mae = rmse = float("inf")
        else:
            diff = ppg_hr[valid] - ref_hr[valid]
            score = float(np.nanstd(diff))
            mae = float(np.nanmean(np.abs(diff)))
            rmse = float(np.sqrt(np.nanmean(diff**2)))
        rows.append(
            {
                "delay_s": float(delay_s),
                "score_std": score,
                "mae": mae,
                "rmse": rmse,
                "n_valid": n_valid,
            }
        )
        if score < best_score:
            best_score = score
            best_delay = float(delay_s)
            best_rest = rest
            best_ref = ref_hr

    score_table = pd.DataFrame(rows, columns=["delay_s", "score_std", "mae", "rmse", "n_valid"])
    if best_rest is None or not np.isfinite(best_score):
        max_valid = int(score_table["n_valid"].max()) if not score_table.empty else 0
        raise ValueError(
            "alignment failed: fewer than 2 comparable tracked rest windows "
            f"(max_valid_windows={max_valid})"
        )
    debug: dict[str, Any] = {
        "alignment_TW": align_tw,
        "alignment_step_s": float(alignment_step_s),
        "delay_range_s": tuple(float(x) for x in delay_range_s),
        "delay_step_s": step,
    }
    if return_debug:
        debug["best_ref_hr_bpm"] = best_ref
    return TDelayEstimateResult(
        best_tdelay_s=best_delay,
        best_score=best_score,
        score_table=score_table,
        rest_ppg_hr=best_rest,
        debug=debug,
    )


def build_aligned_training_windows(
    ds: ProtocolDataset,
    segment: SegmentInfo,
    fs: float,
    train_TW: float,
    best_tdelay_s: float,
    tdelay_result: TDelayEstimateResult | None = None,
) -> AlignedDataset:
    """Build downstream training windows from a pre-estimated global Tdelay.

    中文说明：该函数只使用 ``train_TW`` 生成 window_starts/window_centers/labels 和
    参考 HR 半窗中心校正；它不会重新估计 Tdelay，也不会把 ``alignment_TW`` 用作
    训练窗口长度。
    """

    if not segment.is_valid:
        raise ValueError(f"Cannot build windows for invalid segment info: {segment.reason}")
    fs_i = int(round(float(fs)))
    train_tw = float(train_TW)
    shifted_dataset = _shift_dataset_by_delay(ds, best_tdelay_s, fs_i)
    shifted_start = max(0.0, float(segment.motion_start_s) - float(best_tdelay_s))
    shifted_end = max(shifted_start, float(segment.motion_end_s) - float(best_tdelay_s))

    win_len = int(round(train_tw * fs_i))
    starts_idx = np.arange(0, len(shifted_dataset.time_s) - win_len + 1, fs_i, dtype=int)
    starts_s = starts_idx.astype(float) / fs_i
    centers_s = starts_s + train_tw / 2.0
    labels = np.where(
        centers_s < shifted_start,
        "rest",
        np.where(centers_s <= shifted_end, "motion", "recovery"),
    )

    ref_time_shifted = ds.ref_time_s - train_tw / 2.0
    ref_hr = np.interp(
        centers_s,
        ref_time_shifted,
        ds.ref_hr_bpm,
        left=np.nan,
        right=np.nan,
    )
    valid = np.isfinite(ref_hr)
    starts_s = starts_s[valid]
    centers_s = centers_s[valid]
    labels = labels[valid]
    ref_hr = ref_hr[valid]

    updated_segment = replace(
        segment,
        motion_start_s=shifted_start,
        motion_end_s=shifted_end,
        window_starts_s=starts_s,
        window_centers_s=centers_s,
        labels=labels,
    )
    score_table = tdelay_result.score_table if tdelay_result is not None else pd.DataFrame()
    std_by_delay = (
        dict(zip(score_table["delay_s"].astype(float), score_table["score_std"].astype(float)))
        if not score_table.empty
        else {float(best_tdelay_s): float("nan")}
    )
    best_row = (
        score_table.loc[score_table["delay_s"].astype(float) == float(best_tdelay_s)].head(1)
        if not score_table.empty
        else pd.DataFrame()
    )
    n_valid_score = int(best_row["n_valid"].iloc[0]) if not best_row.empty else 0
    alignment_info = AlignmentInfo(
        best_tdelay_s=float(best_tdelay_s),
        std_by_delay=std_by_delay,
        ref_shift_s=train_tw / 2.0,
        num_windows=int(starts_s.size),
        alignment_tw_s=(
            float(tdelay_result.debug.get("alignment_TW", DEFAULT_ALIGNMENT_TW))
            if tdelay_result is not None and tdelay_result.debug is not None
            else DEFAULT_ALIGNMENT_TW
        ),
        train_tw_s=train_tw,
        best_score=float(tdelay_result.best_score) if tdelay_result is not None else float("nan"),
        n_valid_score_windows=n_valid_score,
    )
    rest_idx = np.flatnonzero(labels == "rest")
    motion_idx = np.flatnonzero(labels == "motion")
    recovery_idx = np.flatnonzero(labels == "recovery")
    return AlignedDataset(
        dataset=shifted_dataset,
        segment_info=updated_segment,
        alignment_info=alignment_info,
        window_starts_s=starts_s,
        window_centers_s=centers_s,
        segment_labels=labels,
        ref_hr_bpm=ref_hr,
        rest_indices=rest_idx,
        motion_indices=motion_idx,
        recovery_indices=recovery_idx,
    )


def _print_alignment_once(sample_stem: str, best_delay: float, rest_windows: int) -> None:
    """Print one Tdelay summary per sample in the current Python process."""

    key = str(sample_stem)
    with _PRINTED_ALIGNMENT_LOCK:
        if key in _PRINTED_ALIGNMENT_SAMPLE_STEMS:
            return
        _PRINTED_ALIGNMENT_SAMPLE_STEMS.add(key)
    print(f"[alignment] {key}: best_tdelay_s={best_delay:.1f}, rest_windows={int(rest_windows)}")


def _reference_sequence_after_half_window(
    dataset: ProtocolDataset,
    TW: float,
    score_start_s: float = 0.0,
) -> np.ndarray:
    """Return finite reference HR values after the half-window shift."""

    mask = np.asarray(dataset.ref_time_s, dtype=float) >= float(score_start_s) + TW / 2.0
    seq = np.asarray(dataset.ref_hr_bpm, dtype=float)[mask]
    return seq[np.isfinite(seq)]


def _limit_hr_transition(
    prev_hr: float | None,
    candidate_hr: float,
    slew_limit_bpm: float,
    slew_step_bpm: float,
) -> float:
    """Limit one rest-HR transition according to the configured slew rule."""

    if not np.isfinite(candidate_hr):
        return float("nan") if prev_hr is None else float(prev_hr)
    if prev_hr is None or not np.isfinite(prev_hr):
        return float(candidate_hr)
    diff = float(candidate_hr) - float(prev_hr)
    if abs(diff) <= float(slew_limit_bpm):
        return float(candidate_hr)
    return float(prev_hr) + float(np.sign(diff)) * float(slew_step_bpm)


def _rest_spectrum_candidates_like_reference(
    x: np.ndarray,
    fs: int,
    hr_band_bpm: tuple[float, float],
    *,
    peak_percent: float,
    penalty_window: np.ndarray | None,
    spec_penalty_enable: bool,
    spec_penalty_weight: float,
    spec_penalty_width_hz: float,
) -> dict[str, Any]:
    """Return reference-style sorted spectral peak candidates for one rest window.

    中文说明：参考仓库先通过 ``fft_peaks(..., percent=0.3)`` 找局部谱峰，再用
    ``find_maxpeak`` 按幅值降序排列候选频率。这里保留同样的后处理口径，同时
    允许静息段 HR 频带继续由本项目的 ``hr_band_bpm`` 控制。
    """

    sig = _prepare_rest_fft_window(x)
    fre_hz, amp = _fft_peak_candidates(sig, fs, hr_band_bpm, peak_percent)
    amp = amp.astype(float, copy=True)
    penalty_applied = False
    penalty_freq_hz = float("nan")

    if spec_penalty_enable and penalty_window is not None and fre_hz.size:
        # 中文说明：运动惩罚需要独立参考通道；若传入 ACC 参考，就按参考实现
        # 压低运动主频和二倍频附近的 PPG 候选峰幅值。
        ref_sig = _prepare_penalty_fft_window(penalty_window)
        ref_freq, ref_amp = _fft_peak_candidates(ref_sig, fs, hr_band_bpm, peak_percent)
        if ref_freq.size:
            penalty_freq_hz = float(ref_freq[int(np.argmax(ref_amp))])
            mask = (np.abs(fre_hz - penalty_freq_hz) < float(spec_penalty_width_hz)) | (
                np.abs(fre_hz - 2.0 * penalty_freq_hz) < float(spec_penalty_width_hz)
            )
            if mask.any():
                amp[mask] *= float(spec_penalty_weight)
                penalty_applied = True

    if fre_hz.size:
        order = np.argsort(-amp, kind="stable")
        fre_hz = fre_hz[order]
        amp = amp[order]

    return {
        "freq_hz": fre_hz,
        "amp": amp,
        "penalty_applied": penalty_applied,
        "penalty_freq_hz": penalty_freq_hz,
    }


def _prepare_rest_fft_window(x: np.ndarray) -> np.ndarray:
    """Demean and Hamming-window one PPG segment before reference-style peak search."""

    sig = np.asarray(x, dtype=float).ravel()
    if sig.size == 0:
        return sig.copy()
    sig = sig.copy()
    sig[~np.isfinite(sig)] = 0.0
    sig = sig - float(np.mean(sig))
    # 中文说明：参考仓库在进入频谱后处理前使用 scipy 的默认对称 Hamming 窗。
    return sig * hamming(sig.size)


def _prepare_penalty_fft_window(x: np.ndarray) -> np.ndarray:
    """Prepare an optional motion-penalty reference window for peak extraction."""

    sig = np.asarray(x, dtype=float).ravel()
    if sig.size == 0:
        return sig.copy()
    sig = sig.copy()
    sig[~np.isfinite(sig)] = 0.0
    return sig - float(np.mean(sig))


def _fft_peak_candidates(
    signal: np.ndarray,
    fs: int,
    hr_band_bpm: tuple[float, float],
    percent: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Find local FFT peaks above a relative amplitude threshold.

    中文说明：这是参考仓库 ``fft_peaks`` 的项目化版本。不同之处是心率频带由
    ``hr_band_bpm`` 控制，而不是写死 0.7-4 Hz，便于静息诊断继续使用 0.5-2 Hz
    或 40-180 BPM 等配置。
    """

    sig = np.asarray(signal, dtype=float).ravel()
    if sig.size == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    nfft = 1 << 13
    spectrum = np.fft.fft(sig, nfft)
    amp_full = np.abs(spectrum) / max(sig.size, 1)
    half = nfft // 2
    amp = amp_full[:half].copy()
    amp[1:] *= 2.0
    freq = float(fs) * np.arange(half, dtype=float) / float(nfft)
    peaks_idx, _ = find_peaks(amp)
    if peaks_idx.size == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    low_hz = float(hr_band_bpm[0]) / 60.0
    high_hz = float(hr_band_bpm[1]) / 60.0
    valid = (freq[peaks_idx] >= low_hz) & (freq[peaks_idx] <= high_hz)
    valid_idx = peaks_idx[valid]
    if valid_idx.size == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    threshold = float(np.max(amp[valid_idx])) * float(percent)
    keep_idx = valid_idx[amp[valid_idx] > threshold]
    if keep_idx.size == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    return freq[keep_idx], amp[keep_idx]


def _window_fft_hr_spectrum(
    x: np.ndarray,
    fs: int,
    hr_band_bpm: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return BPM frequency bins and amplitudes for one Hamming-windowed PPG segment."""

    sig = np.asarray(x, dtype=float)
    if sig.size < 4 or not np.isfinite(sig).any():
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    sig = sig.copy()
    sig[~np.isfinite(sig)] = 0.0
    sig = sig - float(np.mean(sig))
    sig = sig * hamming(sig.size, sym=False)
    nfft = max(8192, 1 << int(np.ceil(np.log2(max(sig.size, 1)))))
    freq = np.fft.rfftfreq(nfft, d=1.0 / fs)
    amp = np.abs(np.fft.rfft(sig, n=nfft))
    low_hz = float(hr_band_bpm[0]) / 60.0
    high_hz = float(hr_band_bpm[1]) / 60.0
    mask = (freq >= low_hz) & (freq <= high_hz)
    if not mask.any():
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    return freq[mask] * 60.0, amp[mask]


def _rest_ppg_hr_tracked_for_delay(
    ppg_green: np.ndarray,
    fs: int,
    *,
    delay_s: float,
    motion_start_s: float,
    score_start_s: float,
    rest_hr_kwargs: dict[str, Any],
    penalty_signal: np.ndarray | None = None,
) -> RestHrResult:
    """Shift PPG by candidate Tdelay and extract comparable tracked rest HR."""

    tw_s = float(rest_hr_kwargs.get("tw_s", DEFAULT_ALIGNMENT_TW))
    step_s = float(rest_hr_kwargs.get("step_s", DEFAULT_ALIGNMENT_STEP_S))
    shifted = _shift_channel_for_delay(ppg_green, delay_s, fs)
    # 中文说明：沿用旧逻辑，候选 delay 会同步改变 shifted 轴上的可用静息截止时间。
    usable_rest_s = max(0.0, float(motion_start_s) - float(delay_s))
    win_len = int(round(tw_s * fs))
    max_start = min(len(shifted) - win_len, int(round((usable_rest_s - tw_s) * fs)))
    min_start = max(0, int(round(float(score_start_s) * fs)))
    if max_start < min_start:
        empty = np.asarray([], dtype=float)
        return RestHrResult(empty, empty, empty, empty, {"reason": "no comparable rest window"})
    max_end = max_start + win_len
    segment = shifted[min_start:max_end]
    penalty_segment = None
    if penalty_signal is not None:
        shifted_penalty = _shift_channel_for_delay(penalty_signal, delay_s, fs)
        penalty_segment = shifted_penalty[min_start:max_end]
    kwargs = dict(rest_hr_kwargs)
    kwargs["tw_s"] = tw_s
    kwargs["step_s"] = step_s
    # 中文说明：参考实现的纯 FFT 路径使用 ACC 作为谱惩罚参考；这里传入同一
    # 时间片的 ACC 窗口，只有 spec_penalty_enable=True 时才会压低运动频率峰。
    if penalty_segment is not None:
        kwargs["penalty_signal"] = penalty_segment
    result = extract_rest_ppg_hr_tracked(segment, fs, **kwargs)
    absolute_times = result.times_s + min_start / float(fs)
    return RestHrResult(
        times_s=absolute_times,
        hr_bpm_raw=result.hr_bpm_raw,
        hr_bpm_tracked=result.hr_bpm_tracked,
        hr_bpm_smooth=result.hr_bpm_smooth,
        quality=result.quality,
    )


def _reference_hr_for_alignment_times(
    dataset: ProtocolDataset,
    ppg_times_s: np.ndarray,
    alignment_TW: float,
) -> np.ndarray:
    """Interpolate reference HR onto rest PPG-HR window-center times.

    中文说明：``RestHrResult.times_s`` 已经是 ``start + alignment_TW/2`` 的窗口中心，
    因此这里不再像旧的“窗口起点序列”那样额外平移参考时间轴；半窗中心校正已经
    体现在 PPG HR 时间戳本身。
    """

    times = np.asarray(ppg_times_s, dtype=float)
    if times.size == 0:
        return np.asarray([], dtype=float)
    ref_hr = np.asarray(dataset.ref_hr_bpm, dtype=float)
    return np.interp(times, np.asarray(dataset.ref_time_s, dtype=float), ref_hr, left=np.nan, right=np.nan)


def compute_rest_alignment_diagnostic_curve(
    dataset: ProtocolDataset,
    segment_info: SegmentInfo,
    aligned: AlignedDataset,
    TW: float,
    fs_target: int,
    rest_hr_kwargs: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Return comparable rest-window reference HR and PPG FFT HR curves.

    中文说明：诊断曲线使用 ``aligned.alignment_info.best_tdelay_s`` 和全局对齐的
    ``alignment_tw_s``，复用 tracked 静息段 PPG-HR 提取逻辑。参数 ``TW`` 仅为
    旧调用的兼容 fallback；不会把训练窗口 TW 误用到 Tdelay 诊断曲线中。
    """

    delay_s = float(aligned.alignment_info.best_tdelay_s)
    alignment_tw = float(getattr(aligned.alignment_info, "alignment_tw_s", TW) or TW)
    score_start_s = _REST_ALIGNMENT_SCORE_START_S
    kwargs = {"tw_s": alignment_tw, "step_s": DEFAULT_ALIGNMENT_STEP_S}
    kwargs.update(rest_hr_kwargs or {})
    kwargs["tw_s"] = alignment_tw
    kwargs.setdefault("step_s", DEFAULT_ALIGNMENT_STEP_S)
    rest = _rest_ppg_hr_tracked_for_delay(
        dataset.ppg_green,
        int(fs_target),
        delay_s=delay_s,
        motion_start_s=float(segment_info.motion_start_s),
        score_start_s=score_start_s,
        rest_hr_kwargs=kwargs,
        penalty_signal=dataset.accz,
    )
    ref_hr = _reference_hr_for_alignment_times(dataset, rest.times_s, alignment_tw)
    ppg_hr = np.asarray(rest.hr_bpm_smooth, dtype=float)
    valid = np.isfinite(ppg_hr) & np.isfinite(ref_hr)
    if not valid.any():
        return pd.DataFrame(columns=["time_s", "ppg_hr_bpm", "ref_hr_bpm"])
    return pd.DataFrame(
        {
            "time_s": rest.times_s[valid],
            "ppg_hr_bpm": ppg_hr[valid],
            "ref_hr_bpm": ref_hr[valid],
        }
    )


def _reference_curve_after_half_window(
    dataset: ProtocolDataset,
    TW: float,
    score_start_s: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite reference HR curve after the same TW/2 correction.

    中文说明：这里与全局 Tdelay 搜索的参考 HR 对齐口径一致，把参考时间减去
    ``TW/2``，使其对应 PPG FFT 窗口的起点序列。
    """

    ref_time = np.asarray(dataset.ref_time_s, dtype=float)
    ref_hr = np.asarray(dataset.ref_hr_bpm, dtype=float)
    mask = ref_time >= float(score_start_s) + TW / 2.0
    time_s = ref_time[mask] - TW / 2.0
    values = ref_hr[mask]
    finite = np.isfinite(values)
    return time_s[finite].astype(float), values[finite].astype(float)


def _shift_channel_for_delay(values: np.ndarray, delay_s: float, fs: int) -> np.ndarray:
    """Shift one channel according to signed global Tdelay.

    Convention:
    - delay_s > 0: sensor lags reference HR; shift sensor left by trimming the head.
    - delay_s == 0: unchanged.
    - delay_s < 0: sensor leads reference HR; shift sensor right by padding the head
      with the first sample arr[0] and trimming the tail.

    For delay_s < 0, padding uses arr[0], not literal zero.
    """
    arr = np.asarray(values, dtype=float)

    if arr.size == 0:
        return arr.copy()

    samples = int(round(float(delay_s) * float(fs)))

    if samples == 0:
        return arr.copy()

    if samples > 0:
        if samples >= arr.size:
            return arr[:0].copy()
        return arr[samples:].copy()

    pad = -samples

    if pad >= arr.size:
        return np.full(arr.shape, arr[0], dtype=float)

    prefix = np.full(pad, arr[0], dtype=float)
    return np.concatenate([prefix, arr[:-pad]]).astype(float, copy=False)


def _shift_dataset_by_delay(dataset: ProtocolDataset, delay_s: float, fs: int) -> ProtocolDataset:
    """Apply signed global Tdelay to every protocol sensor channel.

    All channels returned by dataset.channels() must be shifted by the same delay_s.
    This keeps PPG, ACC, GYRO, HF, CF and other channels synchronized.
    """
    channels = {
        name: _shift_channel_for_delay(values, delay_s, fs)
        for name, values in dataset.channels().items()
    }
    return dataset.replace_channels(channels, fs)


def _rest_ppg_hr_for_delay(
    ppg_green: np.ndarray,
    fs: int,
    TW: float,
    delay_s: float,
    motion_start_s: float,
    score_start_s: float = 0.0,
) -> np.ndarray:
    """Estimate rest-window PPG HR for one candidate delay."""

    _, hr = _rest_ppg_hr_curve_for_delay(
        ppg_green,
        fs,
        TW,
        delay_s,
        motion_start_s,
        score_start_s,
    )
    return hr


def _rest_ppg_hr_curve_for_delay(
    ppg_green: np.ndarray,
    fs: int,
    TW: float,
    delay_s: float,
    motion_start_s: float,
    score_start_s: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate rest-window PPG HR and its comparable rest-window start times.

    中文说明：这是 ``_rest_ppg_hr_for_delay`` 的曲线版封装，仍然调用同一个
    Hamming + FFT HR 提取逻辑，避免诊断图和真实搜索使用两套算法。
    """

    shifted = _shift_channel_for_delay(ppg_green, delay_s, fs)
    # Keep this as motion_start_s - delay_s for both signs: positive delay shifts
    # the sensor left and advances the motion boundary on the shifted axis, while
    # negative delay shifts the sensor right and delays it by abs(delay_s).
    usable_rest_s = max(0.0, float(motion_start_s) - delay_s)
    win_len = int(round(TW * fs))
    max_start = min(len(shifted) - win_len, int(round((usable_rest_s - TW) * fs)))
    min_start = max(0, int(round(float(score_start_s) * fs)))
    if max_start < min_start:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    starts = np.arange(min_start, max_start + 1, fs, dtype=int)
    hrs = [_window_fft_hr(shifted[s : s + win_len], fs, 0.5, 2.0) for s in starts]
    hrs_arr = np.asarray(hrs, dtype=float)
    finite = np.isfinite(hrs_arr)
    return starts[finite].astype(float) / float(fs), hrs_arr[finite]


def _window_fft_hr(x: np.ndarray, fs: int, low_hz: float, high_hz: float) -> float:
    """Extract the dominant frequency in one Hamming-windowed segment."""

    sig = np.asarray(x, dtype=float)
    if sig.size < 4 or not np.isfinite(sig).any():
        return float("nan")
    sig = sig - np.nanmean(sig)
    sig = sig * hamming(sig.size, sym=False)
    nfft = max(8192, 1 << int(np.ceil(np.log2(max(sig.size, 1)))))
    freq = np.fft.rfftfreq(nfft, d=1.0 / fs)
    amp = np.abs(np.fft.rfft(sig, n=nfft))
    mask = (freq >= low_hz) & (freq <= high_hz)
    if not mask.any():
        return float("nan")
    idx = np.flatnonzero(mask)[int(np.argmax(amp[mask]))]
    return float(freq[idx] * 60.0)


def _left_shift_dataset(dataset: ProtocolDataset, delay_s: float, fs: int) -> ProtocolDataset:
    """Compatibility wrapper for signed global Tdelay shifting."""

    return _shift_dataset_by_delay(dataset, delay_s, fs)
