"""Signed PPG-to-reference HR delay alignment for the protocol.

中文说明：本模块把“全局 Tdelay 估计”和“后续训练窗口构建”拆成两步。
全局 Tdelay 只使用固定的 ``alignment_TW``（默认 8 s）在静息段提取 tracked
PPG-HR 并搜索时延；后续自适应滤波训练/验证/测试窗口继续使用 trial 的
``train_TW`` / ``params.TW``。二者物理含义不同，不能混用。

Signed Tdelay 的通道平移语义继续保持旧逻辑：delay_s > 0 时传感器侧左移；
delay_s < 0 时用首样本补头并右移；delay_s == 0 时不移动。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from threading import Lock
from typing import Any

import numpy as np
import pandas as pd

from ..core.find_near_biggest import find_near_biggest
from ..preprocess.utils import smoothdata_movmedian
from .preprocess_protocol import ProtocolDataset
from .segmentation import SegmentInfo
from .spectral_utils import compute_power_spectrum, fft_peak_candidates, prepare_fft_window

__all__ = [
    "AlignedDataset",
    "AlignmentInfo",
    "RestHrResult",
    "TDelayEstimateResult",
    "TimeBiasAfterResult",
    "align_ppg_to_ref_hr",
    "apply_rest_hr_slew_limit",
    "build_aligned_training_windows",
    "compute_rest_alignment_diagnostic_curve",
    "estimate_global_tdelay_from_rest",
    "extract_rest_ppg_hr_tracked",
    "search_time_bias_after",
    "smooth_rest_hr_sequence",
]

#静息窗数据
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
    alignment_tw_s: float = DEFAULT_ALIGNMENT_TW
    train_tw_s: float = 0.0
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
            "alignment_tw_s": float(self.alignment_tw_s),
            "train_tw_s": float(self.train_tw_s),
            "alignment_score_mode": str(self.alignment_score_mode),
            "best_score": float(self.best_score),
            "best_score_aae": float(self.best_score_aae),
            "best_score_std": float(self.best_score_std),
            "best_rmse": float(self.best_rmse),
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
        ref_shifted = np.interp(pred_time + float(bias_s), ref_time, ref_hr, left=np.nan, right=np.nan)
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
        columns=["time_bias_after_s", "aae_bpm", "std_bpm", "rmse_bpm", "n_valid", "selected"],
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
    ref_after = np.interp(pred_time + best_bias, ref_time, ref_hr, left=np.nan, right=np.nan)
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


def _tdelay_score_key_is_better(
    candidate: tuple[float, float, float, float],
    best: tuple[float, float, float, float],
) -> bool:
    """Return whether a Tdelay candidate wins under the configured score mode.

    中文说明：key 的顺序是“主评分、绝对 delay、RMSE、非主评分”。主评分
    可以是 AAE 或 STD；并列时优先选择更小的绝对延迟，避免无意义的大偏移。
    """

    for cand_value, best_value in zip(candidate, best, strict=True):
        if np.isclose(cand_value, best_value, rtol=0.0, atol=1e-9):
            continue
        return bool(cand_value < best_value)
    return False


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
    alignment_score_mode: str = "aae",
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
        alignment_score_mode=alignment_score_mode,
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
    alignment_score_mode: str = "aae",
    return_debug: bool = False,
) -> TDelayEstimateResult:
    """Estimate global signed Tdelay from tracked rest-segment PPG HR.

    中文说明：本函数只负责全局 Tdelay 搜索，使用固定 ``alignment_TW``；它不构建
    后续训练窗口，也不读取 Optuna trial 的 ``params.TW``。候选 delay 的正负号
    和边界补值继续复用当前项目的 signed Tdelay 逻辑。
    """

    if not segment.is_valid:
        raise ValueError(f"Cannot estimate Tdelay for invalid segment info: {segment.reason}")
    score_mode = str(alignment_score_mode).lower()
    if score_mode == "mae":
        score_mode = "aae"
    if score_mode not in {"aae", "std"}:
        raise ValueError("alignment_score_mode must be 'aae', 'mae', or 'std'")
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
    best_score_aae = float("inf")
    best_score_std = float("inf")
    best_rmse = float("inf")
    best_rest: RestHrResult | None = None
    best_ref = np.asarray([], dtype=float)
    best_key: tuple[float, float, float, float] | None = None
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
            score_std = score_aae = mae = rmse = selected_score = float("inf")
        else:
            diff = ppg_hr[valid] - ref_hr[valid]
            score_std = float(np.nanstd(diff))
            score_aae = float(np.nanmean(np.abs(diff)))
            mae = score_aae
            rmse = float(np.sqrt(np.nanmean(diff**2)))
            selected_score = score_aae if score_mode == "aae" else score_std
        rows.append(
            {
                "delay_s": float(delay_s),
                "score_std": score_std,
                "score_aae": score_aae,
                "mae": mae,
                "rmse": rmse,
                "n_valid": n_valid,
                "selected_score": selected_score,
                "selected": False,
            }
        )
        if n_valid >= 2 and np.isfinite(selected_score):
            other_score = score_std if score_mode == "aae" else score_aae
            candidate_key = (selected_score, abs(float(delay_s)), rmse, other_score)
            if best_key is None or _tdelay_score_key_is_better(candidate_key, best_key):
                best_key = candidate_key
                best_score = selected_score
                best_score_aae = score_aae
                best_score_std = score_std
                best_rmse = rmse
                best_delay = float(delay_s)
                best_rest = rest
                best_ref = ref_hr

    score_table = pd.DataFrame(
        rows,
        columns=["delay_s", "score_std", "score_aae", "mae", "rmse", "n_valid", "selected_score", "selected"],
    )
    if best_rest is None or not np.isfinite(best_score):
        max_valid = int(score_table["n_valid"].max()) if not score_table.empty else 0
        raise ValueError(
            "alignment failed: fewer than 2 comparable tracked rest windows "
            f"(max_valid_windows={max_valid})"
        )
    selected_mask = np.isclose(score_table["delay_s"].to_numpy(dtype=float), float(best_delay), rtol=0.0, atol=1e-9)
    if selected_mask.any():
        score_table.loc[selected_mask, "selected"] = True
    debug: dict[str, Any] = {
        "alignment_TW": align_tw,
        "alignment_step_s": float(alignment_step_s),
        "delay_range_s": tuple(float(x) for x in delay_range_s),
        "delay_step_s": step,
        "alignment_score_mode": score_mode,
    }
    if return_debug:
        debug["best_ref_hr_bpm"] = best_ref
    return TDelayEstimateResult(
        best_tdelay_s=best_delay,
        best_score=best_score,
        score_table=score_table,
        rest_ppg_hr=best_rest,
        alignment_score_mode=score_mode,
        best_score_aae=best_score_aae,
        best_score_std=best_score_std,
        best_rmse=best_rmse,
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
        if not score_table.empty and "score_std" in score_table
        else {float(best_tdelay_s): float("nan")}
    )
    aae_col = "score_aae" if "score_aae" in score_table else "mae"
    aae_by_delay = (
        dict(zip(score_table["delay_s"].astype(float), score_table[aae_col].astype(float)))
        if not score_table.empty and aae_col in score_table
        else {float(best_tdelay_s): float("nan")}
    )
    rmse_by_delay = (
        dict(zip(score_table["delay_s"].astype(float), score_table["rmse"].astype(float)))
        if not score_table.empty and "rmse" in score_table
        else {float(best_tdelay_s): float("nan")}
    )
    best_row = (
        score_table.loc[score_table["delay_s"].astype(float) == float(best_tdelay_s)].head(1)
        if not score_table.empty
        else pd.DataFrame()
    )
    n_valid_score = int(best_row["n_valid"].iloc[0]) if not best_row.empty else 0
    alignment_score_mode = (
        str(tdelay_result.alignment_score_mode)
        if tdelay_result is not None and getattr(tdelay_result, "alignment_score_mode", "")
        else "aae"
    )
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
        aae_by_delay=aae_by_delay,
        rmse_by_delay=rmse_by_delay,
        alignment_score_mode=alignment_score_mode,
        best_score_aae=(
            float(tdelay_result.best_score_aae)
            if tdelay_result is not None and np.isfinite(getattr(tdelay_result, "best_score_aae", np.nan))
            else (float(best_row[aae_col].iloc[0]) if not best_row.empty and aae_col in best_row else float("nan"))
        ),
        best_score_std=(
            float(tdelay_result.best_score_std)
            if tdelay_result is not None and np.isfinite(getattr(tdelay_result, "best_score_std", np.nan))
            else (float(best_row["score_std"].iloc[0]) if not best_row.empty and "score_std" in best_row else float("nan"))
        ),
        best_rmse=(
            float(tdelay_result.best_rmse)
            if tdelay_result is not None and np.isfinite(getattr(tdelay_result, "best_rmse", np.nan))
            else (float(best_row["rmse"].iloc[0]) if not best_row.empty and "rmse" in best_row else float("nan"))
        ),
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
    freq, amp = compute_power_spectrum(sig, fs, apply_hamming=True, demean=True)
    mask = (freq >= low_hz) & (freq <= high_hz)
    if not mask.any():
        return float("nan")
    idx = np.flatnonzero(mask)[int(np.argmax(amp[mask]))]
    return float(freq[idx] * 60.0)


def _left_shift_dataset(dataset: ProtocolDataset, delay_s: float, fs: int) -> ProtocolDataset:
    """Compatibility wrapper for signed global Tdelay shifting."""

    return _shift_dataset_by_delay(dataset, delay_s, fs)
