"""Sensor-center window plans and post-prediction reference evaluation.

Reference offsets affect scoring and plots only. Sensor signals and prediction
windows are never shifted or cropped to match reference coverage.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from .penalty_policy import PenaltyPolicy

import numpy as np
import pandas as pd

from ..core.find_near_biggest import find_near_biggest
from ..preprocess.utils import smoothdata_movmedian
from .preprocess_protocol import ProtocolDataset, SensorDataset
from .segmentation import SegmentInfo
from .spectral_utils import (
    compute_power_spectrum,
    fft_peak_candidates,
    prepare_fft_window,
)

__all__ = [
    "SensorWindowPlan",
    "AlignmentInfo",
    "RestHrResult",
    "TDelayEstimateResult",
    "TimeBiasAfterResult",
    "evaluate_rest_alignment",
    "apply_rest_hr_slew_limit",
    "build_sensor_window_plan",
    "compute_rest_alignment_diagnostic_curve",
    "estimate_global_tdelay_from_rest",
    "extract_rest_ppg_hr_tracked",
    "search_time_bias_after",
    "smooth_rest_hr_sequence",
]

# 静息窗数据
_REST_ALIGNMENT_SCORE_START_S = 5.0
DEFAULT_EVALUATION_WINDOW_S = 8.0
DEFAULT_EVALUATION_STEP_S = 1.0
DEFAULT_REST_HR_BAND_BPM = (40.0, 180.0)
DEFAULT_REST_HR_TRACK_BAND_BPM = 30.0
DEFAULT_REST_HR_SLEW_LIMIT_BPM = 6.0
DEFAULT_REST_HR_SLEW_STEP_BPM = 4.0
DEFAULT_REST_HR_SMOOTH_METHOD = "median"
DEFAULT_REST_HR_SMOOTH_WIN = 7
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
    alignment_score_mode: str = "aae"
    best_score_aae: float = float("nan")
    best_score_std: float = float("nan")
    best_rmse: float = float("nan")
    debug: dict[str, Any] | None = None


@dataclass(frozen=True)
class TimeBiasAfterResult:
    """Post-filter curve-level time-bias result used only for diagnostics.

    中文说明：``time_bias_after_s = b`` 的符号约定固定为：
    对预测时间点 ``t_pred_s``，从原始参考心率曲线的
    ``t_pred_s + b`` 位置取样，即 ``np.interp(pred_time_s + b,
    ref_time_s, ref_hr_bpm)``。因此 ``b > 0`` 表示预测 HR 与更晚的
    参考 HR 比较；画图时等价于把参考曲线向左移动 ``b`` 秒。

    该结果是自适应滤波已经完成之后的评价/可视化层参数，不能回写到
    原始 PPG、补偿信号、窗口切分或滤波器运行流程中。
    """

    time_bias_after_s: float
    best_aae_bpm: float
    best_std_bpm: float
    best_rmse_bpm: float
    n_valid: int
    score_table: pd.DataFrame
    ref_hr_after_bpm: np.ndarray
    status: str = "ok"
    reason: str = ""
    mode: str = "posthoc_oracle_alignment"
    search_range_s: tuple[float, float] = (-5.0, 5.0)
    search_step_s: float = 1.0

    def to_dict(self, *, include_score_table: bool = True) -> dict[str, Any]:
        """Return a JSON-friendly representation of the post-hoc alignment."""

        def finite_or_none(value: Any) -> float | None:
            value_f = float(value)
            return value_f if np.isfinite(value_f) else None

        payload: dict[str, Any] = {
            "time_bias_after_s": finite_or_none(self.time_bias_after_s),
            "best_aae_bpm": finite_or_none(self.best_aae_bpm),
            "best_std_bpm": finite_or_none(self.best_std_bpm),
            "best_rmse_bpm": finite_or_none(self.best_rmse_bpm),
            "n_valid": int(self.n_valid),
            "status": self.status,
            "reason": self.reason,
            "mode": self.mode,
            "search_range_s": [float(x) for x in self.search_range_s],
            "search_step_s": float(self.search_step_s),
        }
        if include_score_table:
            records = self.score_table.to_dict(orient="records")
            payload["score_table"] = [
                {
                    str(key): (
                        finite_or_none(value)
                        if not isinstance(value, (bool, np.bool_))
                        and isinstance(value, (int, float, np.integer, np.floating))
                        else value
                    )
                    for key, value in row.items()
                }
                for row in records
            ]
        return payload


@dataclass(frozen=True)
class AlignmentInfo:
    """Chosen delay and per-candidate alignment diagnostics."""

    best_tdelay_s: float
    std_by_delay: dict[float, float]
    ref_shift_s: float
    num_windows: int
    status: str = "ok"
    reason: str = ""
    evaluation_window_s: float = DEFAULT_EVALUATION_WINDOW_S
    hr_window_s: float = 0.0
    best_score: float = float("nan")
    n_valid_score_windows: int = 0
    aae_by_delay: dict[float, float] = field(default_factory=dict)
    rmse_by_delay: dict[float, float] = field(default_factory=dict)
    alignment_score_mode: str = "aae"
    best_score_aae: float = float("nan")
    best_score_std: float = float("nan")
    best_rmse: float = float("nan")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""

        return {
            "best_tdelay_s": float(self.best_tdelay_s),
            "std_by_delay": {str(k): float(v) for k, v in self.std_by_delay.items()},
            "aae_by_delay": {str(k): float(v) for k, v in self.aae_by_delay.items()},
            "rmse_by_delay": {str(k): float(v) for k, v in self.rmse_by_delay.items()},
            "ref_shift_s": float(self.ref_shift_s),
            "num_windows": int(self.num_windows),
            "status": self.status,
            "reason": self.reason,
            "evaluation_window_s": float(self.evaluation_window_s),
            "hr_window_s": float(self.hr_window_s),
            "alignment_score_mode": str(self.alignment_score_mode),
            "best_score": float(self.best_score),
            "best_score_aae": float(self.best_score_aae),
            "best_score_std": float(self.best_score_std),
            "best_rmse": float(self.best_rmse),
            "n_valid_score_windows": int(self.n_valid_score_windows),
        }


@dataclass(frozen=True)
class SensorWindowPlan:
    """Dataset after sensor left-shift and reference half-window alignment."""

    dataset: SensorDataset
    segment_info: SegmentInfo
    alignment_info: AlignmentInfo
    window_starts_s: np.ndarray
    window_centers_s: np.ndarray
    segment_labels: np.ndarray
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


def search_time_bias_after(
    pred_time_s: np.ndarray,
    pred_hr_bpm: np.ndarray,
    ref_time_s: np.ndarray,
    ref_hr_bpm: np.ndarray,
    *,
    search_range_s: tuple[float, float] = (-5.0, 5.0),
    search_step_s: float = 1.0,
    min_valid: int = 2,
    mode: str = "posthoc_oracle_alignment",
) -> TimeBiasAfterResult:
    """Search a post-filter HR-curve time bias for diagnostics only.

    Sign convention is intentionally fixed here and in tests:
    ``time_bias_after_s = b`` means that each predicted HR sample at
    ``t_pred_s`` is compared with the reference HR sampled at
    ``t_pred_s + b``::

        ref_shifted = np.interp(
            pred_time_s + b,
            ref_time_s,
            ref_hr_bpm,
            left=np.nan,
            right=np.nan,
        )

    Positive ``b`` compares the prediction with a later reference HR value;
    on a plot this is equivalent to moving the reference curve left by
    ``b`` seconds.

    中文说明：这个搜索只能发生在 HR 序列已经生成之后，用于 post-hoc
    评价和可视化诊断。它不允许改变 PPG、补偿信号、自适应滤波输入、
    窗口划分、滤波器输出。当 OPTIMIZATION_OBJECTIVE 设为 "posthoc_aae" 时，
    后对齐 AAE 可作为 Optuna 优化目标；否则仅用于诊断报告。
    """

    pred_time = np.asarray(pred_time_s, dtype=float).ravel()
    pred_hr = np.asarray(pred_hr_bpm, dtype=float).ravel()
    ref_time = np.asarray(ref_time_s, dtype=float).ravel()
    ref_hr = np.asarray(ref_hr_bpm, dtype=float).ravel()

    if pred_time.size != pred_hr.size:
        raise ValueError("pred_time_s and pred_hr_bpm must have the same length")
    if ref_time.size != ref_hr.size:
        raise ValueError("ref_time_s and ref_hr_bpm must have the same length")

    step = float(search_step_s)
    if not np.isfinite(step) or step <= 0:
        raise ValueError("search_step_s must be a positive finite value")
    lo, hi = (float(search_range_s[0]), float(search_range_s[1]))
    if lo > hi:
        raise ValueError("search_range_s must be ordered as (min_s, max_s)")

    # 中文说明：候选偏置表保持稳定列名，便于 CSV/JSON 输出和后续审计。
    bias_grid = np.round(np.arange(lo, hi + step * 0.5, step), 10)
    bias_grid = bias_grid[bias_grid <= hi + 1e-9]
    rows: list[dict[str, Any]] = []
    best_idx: int | None = None
    best_key: tuple[float, float, float, float] | None = None

    for idx, bias_s in enumerate(bias_grid):
        ref_shifted = np.interp(
            pred_time + float(bias_s), ref_time, ref_hr, left=np.nan, right=np.nan
        )
        valid = np.isfinite(pred_hr) & np.isfinite(ref_shifted)
        n_valid = int(valid.sum())
        if n_valid < int(min_valid):
            aae = std = rmse = float("inf")
        else:
            diff = pred_hr[valid] - ref_shifted[valid]
            aae = float(np.nanmean(np.abs(diff)))
            std = float(np.nanstd(diff))
            rmse = float(np.sqrt(np.nanmean(diff**2)))
            # 中文说明：主评分是 AAE；并列时按“更小绝对偏置、RMSE、STD”的顺序裁决。
            key = (aae, abs(float(bias_s)), rmse, std)
            if best_key is None or _time_bias_after_key_is_better(key, best_key):
                best_key = key
                best_idx = idx
        rows.append(
            {
                "time_bias_after_s": float(bias_s),
                "aae_bpm": aae,
                "std_bpm": std,
                "rmse_bpm": rmse,
                "n_valid": n_valid,
                "selected": False,
            }
        )

    score_table = pd.DataFrame(
        rows,
        columns=[
            "time_bias_after_s",
            "aae_bpm",
            "std_bpm",
            "rmse_bpm",
            "n_valid",
            "selected",
        ],
    )
    if best_idx is None:
        ref_after = np.interp(pred_time, ref_time, ref_hr, left=np.nan, right=np.nan)
        reason = "no candidate has enough finite overlapping HR samples"
        return TimeBiasAfterResult(
            time_bias_after_s=0.0,
            best_aae_bpm=float("nan"),
            best_std_bpm=float("nan"),
            best_rmse_bpm=float("nan"),
            n_valid=0,
            score_table=score_table,
            ref_hr_after_bpm=ref_after,
            status="failed",
            reason=reason,
            mode=str(mode),
            search_range_s=(lo, hi),
            search_step_s=step,
        )

    score_table.loc[int(best_idx), "selected"] = True
    best_row = score_table.iloc[int(best_idx)]
    best_bias = float(best_row["time_bias_after_s"])
    ref_after = np.interp(
        pred_time + best_bias, ref_time, ref_hr, left=np.nan, right=np.nan
    )
    return TimeBiasAfterResult(
        time_bias_after_s=best_bias,
        best_aae_bpm=float(best_row["aae_bpm"]),
        best_std_bpm=float(best_row["std_bpm"]),
        best_rmse_bpm=float(best_row["rmse_bpm"]),
        n_valid=int(best_row["n_valid"]),
        score_table=score_table,
        ref_hr_after_bpm=ref_after,
        status="ok",
        reason="",
        mode=str(mode),
        search_range_s=(lo, hi),
        search_step_s=step,
    )


def _time_bias_after_key_is_better(
    candidate: tuple[float, float, float, float],
    best: tuple[float, float, float, float],
) -> bool:
    """Return whether one post-hoc bias tie-break key beats another."""

    for cand_value, best_value in zip(candidate, best, strict=True):
        if np.isclose(cand_value, best_value, rtol=0.0, atol=1e-9):
            continue
        return bool(cand_value < best_value)
    return False


def evaluate_rest_alignment(
    dataset: ProtocolDataset,
    segment_info: SegmentInfo,
    hr_window_s: int | float,
    fs_target: int,
    *,
    evaluation_window_s: float | None = None,
    evaluation_step_s: float = DEFAULT_EVALUATION_STEP_S,
    delay_range_s: tuple[float, float] | None = None,
    delay_step_s: float | None = None,
    rest_hr_kwargs: dict[str, Any] | None = None,
    use_tracked_rest_hr: bool = True,
    alignment_score_mode: str = "aae",
    return_debug: bool = False,
) -> SensorWindowPlan:
    """Attach a post-prediction rest evaluation offset without moving sensor data."""

    if not segment_info.is_valid:
        raise ValueError(f"Cannot align invalid segment info: {segment_info.reason}")

    fs = int(fs_target)
    hr_window_s = float(hr_window_s)
    evaluation_window = (
        DEFAULT_EVALUATION_WINDOW_S
        if evaluation_window_s is None
        else float(evaluation_window_s)
    )
    if not use_tracked_rest_hr:
        rest_hr_kwargs = {**(rest_hr_kwargs or {}), "smooth_method": "none"}
    tdelay = estimate_global_tdelay_from_rest(
        dataset,
        segment_info,
        fs,
        evaluation_window_s=evaluation_window,
        evaluation_step_s=float(evaluation_step_s),
        delay_range_s=delay_range_s,
        delay_step_s=delay_step_s,
        rest_hr_kwargs=rest_hr_kwargs,
        alignment_score_mode=alignment_score_mode,
        return_debug=return_debug,
    )
    plan = build_sensor_window_plan(
        dataset.sensor_view(),
        segment_info,
        fs,
        hr_window_s=hr_window_s,
    )
    return replace(
        plan,
        alignment_info=replace(
            plan.alignment_info,
            best_tdelay_s=tdelay.best_tdelay_s,
            best_score=tdelay.best_score,
            best_score_aae=tdelay.best_score_aae,
            best_score_std=tdelay.best_score_std,
            best_rmse=tdelay.best_rmse,
            alignment_score_mode=tdelay.alignment_score_mode,
        ),
    )


def extract_rest_ppg_hr_tracked(
    ppg_signal: np.ndarray,
    fs: float,
    tw_s: float = DEFAULT_EVALUATION_WINDOW_S,
    step_s: float = DEFAULT_EVALUATION_STEP_S,
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
        return RestHrResult(
            empty,
            empty,
            empty,
            empty,
            {"reason": "signal shorter than one rest HR window"},
        )

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
    penalty_arr = (
        None
        if penalty_signal is None
        else np.asarray(penalty_signal, dtype=float).ravel()
    )

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
        limited = _limit_hr_transition(
            prev_hr, candidate, slew_limit_bpm, slew_step_bpm
        )
        slew_limited[row] = bool(
            prev_hr is not None
            and np.isfinite(prev_hr)
            and abs(candidate - prev_hr) > slew_limit_bpm
        )
        tracked_hr[row] = limited
        prev_hr = limited if np.isfinite(limited) else prev_hr

    smooth_hr = smooth_rest_hr_sequence(
        tracked_hr, method=smooth_method, smooth_win=smooth_win
    )
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
        out[idx] = _limit_hr_transition(
            prev, float(value), slew_limit_bpm, slew_step_bpm
        )
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
    evaluation_window_s: float = DEFAULT_EVALUATION_WINDOW_S,
    evaluation_step_s: float = DEFAULT_EVALUATION_STEP_S,
    delay_range_s: tuple[float, float] | None = None,
    delay_step_s: float | None = None,
    rest_hr_kwargs: dict[str, Any] | None = None,
    alignment_score_mode: str = "aae",
    return_debug: bool = False,
) -> TDelayEstimateResult:
    """Score reference offsets after extracting a fixed sensor-center rest curve."""
    kwargs = dict(rest_hr_kwargs or {})
    kwargs.update(tw_s=evaluation_window_s, step_s=evaluation_step_s)
    full = extract_rest_ppg_hr_tracked(
        ds.ppg_green, fs, penalty_signal=ds.accz, **kwargs
    )
    labels = segment.states_at(full.times_s)[0]
    keep = (labels == "rest") & (full.times_s >= _REST_ALIGNMENT_SCORE_START_S)
    rest = RestHrResult(
        full.times_s[keep],
        full.hr_bpm_raw[keep],
        full.hr_bpm_tracked[keep],
        full.hr_bpm_smooth[keep],
        full.quality,
    )
    lo, hi = delay_range_s if delay_range_s is not None else (-5.0, 5.0)
    step = 0.1 if delay_step_s is None else delay_step_s
    mode = alignment_score_mode
    if mode not in {"aae", "std"}:
        raise ValueError("alignment_score_mode must be aae or std")
    rows = []
    for delay in np.arange(lo, hi + step / 2, step):
        ref = np.interp(
            rest.times_s + delay,
            ds.ref_time_s,
            ds.ref_hr_bpm,
            left=np.nan,
            right=np.nan,
        )
        valid = np.isfinite(ref) & np.isfinite(rest.hr_bpm_smooth)
        diff = rest.hr_bpm_smooth[valid] - ref[valid]
        aae = float(np.mean(np.abs(diff))) if len(diff) >= 2 else float("inf")
        std = float(np.std(diff)) if len(diff) >= 2 else float("inf")
        rmse = float(np.sqrt(np.mean(diff**2))) if len(diff) >= 2 else float("inf")
        rows.append(
            dict(
                delay_s=float(delay),
                score_aae=aae,
                score_std=std,
                mae=aae,
                rmse=rmse,
                n_valid=int(valid.sum()),
                selected_score=aae if mode == "aae" else std,
                selected=False,
            )
        )
    table = pd.DataFrame(rows)
    finite = table[np.isfinite(table.selected_score)]
    if finite.empty:
        raise ValueError(
            "evaluation alignment has fewer than 2 comparable rest windows"
        )
    idx = min(
        finite.index,
        key=lambda i: (
            table.loc[i, "selected_score"],
            abs(table.loc[i, "delay_s"]),
            table.loc[i, "rmse"],
        ),
    )
    table.loc[idx, "selected"] = True
    best = table.loc[idx]
    return TDelayEstimateResult(
        float(best.delay_s),
        float(best.selected_score),
        table,
        rest,
        mode,
        float(best.score_aae),
        float(best.score_std),
        float(best.rmse),
        dict(
            evaluation_window_s=evaluation_window_s, evaluation_step_s=evaluation_step_s
        ),
    )


def build_sensor_window_plan(
    ds: SensorDataset,
    segment: SegmentInfo,
    fs: float,
    hr_window_s: float,
) -> SensorWindowPlan:
    """Build full 8 s / 1 s sensor windows independently of reference HR."""

    if not segment.is_valid:
        raise ValueError(f"Invalid detector result: {segment.reason}")
    fs_i = int(round(float(fs)))
    train_tw = float(hr_window_s)
    if train_tw != 8.0:
        raise ValueError("HR observation is frozen at 8 s / 1 s")
    # Reference values are attached only by the evaluation layer after prediction.
    duration = (
        len(ds.source_imu.time_s) / ds.source_imu.fs
        if ds.source_imu is not None
        else len(ds.time_s) / fs_i
    )
    win_len = int(round(train_tw * fs_i))
    max_start = min(
        len(ds.time_s) - win_len, int(np.floor((duration - train_tw) * fs_i + 1e-8))
    )
    starts_idx = np.arange(0, max_start + 1, fs_i, dtype=int)
    origin = float(ds.time_s[0]) if len(ds.time_s) else 0.0
    starts = origin + starts_idx.astype(float) / fs_i
    centers = starts + train_tw / 2
    labels, _, _ = segment.states_at(centers)
    info = AlignmentInfo(
        best_tdelay_s=0.0,
        std_by_delay={},
        ref_shift_s=0.0,
        num_windows=len(starts),
        hr_window_s=train_tw,
        alignment_score_mode="sensor_only",
    )
    return SensorWindowPlan(
        ds,
        segment,
        info,
        starts,
        centers,
        labels,
        np.flatnonzero(labels == "rest"),
        np.flatnonzero(labels == "motion"),
        np.flatnonzero(labels == "recovery"),
    )


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
            policy = PenaltyPolicy.build(ref_freq, ref_amp, fre_hz,
                width_hz=spec_penalty_width_hz, base_retention=spec_penalty_weight)
            weights = policy.build_weights(fre_hz)
            amp *= weights
            penalty_applied = bool(np.any(weights < 1))


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

    # 中文说明：参考仓库在进入频谱后处理前使用 scipy 的默认对称 Hamming 窗。
    return prepare_fft_window(x, apply_hamming=True, demean=True, hamming_sym=True)


def _prepare_penalty_fft_window(x: np.ndarray) -> np.ndarray:
    """Prepare an optional motion-penalty reference window for peak extraction."""

    return prepare_fft_window(x, apply_hamming=False, demean=True)


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

    return fft_peak_candidates(signal, fs, hr_band_bpm, percent, nfft=1 << 13)


def _window_fft_hr_spectrum(
    x: np.ndarray,
    fs: int,
    hr_band_bpm: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return BPM frequency bins and amplitudes for one Hamming-windowed PPG segment."""

    sig = np.asarray(x, dtype=float)
    if sig.size < 4 or not np.isfinite(sig).any():
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    freq, amp = compute_power_spectrum(sig, fs, apply_hamming=True, demean=True)
    low_hz = float(hr_band_bpm[0]) / 60.0
    high_hz = float(hr_band_bpm[1]) / 60.0
    mask = (freq >= low_hz) & (freq <= high_hz)
    if not mask.any():
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    return freq[mask] * 60.0, amp[mask]


def compute_rest_alignment_diagnostic_curve(
    dataset: ProtocolDataset,
    segment_info: SegmentInfo,
    aligned: SensorWindowPlan,
    hr_window_s: float,
    fs_target: int,
    *,
    rest_hr_kwargs: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Evaluate a complete sensor-center rest curve; no sensor shift or HR feedback."""
    kwargs = dict(rest_hr_kwargs or {})
    kwargs.update(tw_s=hr_window_s, step_s=DEFAULT_EVALUATION_STEP_S)
    rest = extract_rest_ppg_hr_tracked(
        dataset.ppg_green, fs_target, penalty_signal=dataset.accz, **kwargs
    )
    ref = np.interp(
        rest.times_s + aligned.alignment_info.best_tdelay_s,
        dataset.ref_time_s,
        dataset.ref_hr_bpm,
        left=np.nan,
        right=np.nan,
    )
    keep = (
        (segment_info.states_at(rest.times_s)[0] == "rest")
        & np.isfinite(ref)
        & np.isfinite(rest.hr_bpm_smooth)
    )
    return pd.DataFrame(
        dict(
            time_s=rest.times_s[keep],
            ppg_hr_bpm=rest.hr_bpm_smooth[keep],
            ref_hr_bpm=ref[keep],
        )
    )


def _window_fft_hr(x: np.ndarray, fs: int, low_hz: float, high_hz: float) -> float:
    """Extract the dominant frequency in one Hamming-windowed segment."""

    sig = np.asarray(x, dtype=float)
    if sig.size < 4 or not np.isfinite(sig).any():
        return float("nan")
    freq, amp = compute_power_spectrum(sig, fs, apply_hamming=True, demean=True)
    mask = (freq >= low_hz) & (freq <= high_hz)
    if not mask.any():
        return float("nan")
    idx = np.flatnonzero(mask)[int(np.argmax(amp[mask]))]
    return float(freq[idx] * 60.0)
