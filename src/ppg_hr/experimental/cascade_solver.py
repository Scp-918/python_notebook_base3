"""Trial execution for protocol cascade filtering and HR extraction.

中文说明：本模块执行“一组协议参数”的完整 trial：重采样、分段、对齐、估计
Fmove、逐窗归一化、包络时延、级联自适应滤波、频谱惩罚提取 HR，最后计算
目标段 AAE 和 accuracy。Optuna 每个 trial 都会调用这里。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..core.find_near_biggest import find_near_biggest
from ..params import CascadeScheme, TargetScope
from ..preprocess.utils import smoothdata_movmedian
from .alignment import (
    AlignedDataset,
    AlignmentInfo,
    TDelayEstimateResult,
    TimeBiasAfterResult,
    build_aligned_training_windows,
    estimate_global_tdelay_from_rest,
    search_time_bias_after,
)
from .envelope_delay import DelayEstimate, estimate_envelope_delays
from .fusion import fuse_final_hr
from .klms import noncausal_klms_filter
from .motion_frequency import estimate_motion_frequency
from .noncausal_lms import map_delay_to_lms_params, noncausal_lms_filter
from .preprocess_protocol import (
    PROTOCOL_CHANNELS,
    ProtocolDataset,
    apply_ppg_input_transform,
    resample_protocol_dataset,
)
from .protocol_search_space import ProtocolTrialParams
from .rff_lms import noncausal_rff_lms_filter
from .segmentation import SegmentInfo, detect_activity_segments
from .spectral_utils import compute_power_spectrum, dominant_frequency_in_band
from .volterra import noncausal_volterra_filter

__all__ = [
    "MetricArrays",
    "ProtocolRunResult",
    "aggregate_metric_arrays",
    "clear_all_caches",
    "clear_trial_caches",
    "clear_trial_heavy_caches",
    "extract_hr_with_penalty",
    "extract_plain_fft_hr",
    "run_protocol_trial",
]

MetricArrays = dict[str, np.ndarray]
_PRINTED_GLOBAL_ALIGNMENT_KEYS: set[tuple[Any, ...]] = set()


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
    final_aae_bpm: float
    baseline_acc_pct: float
    adaptive_acc_pct: float
    final_acc_pct: float
    frame: pd.DataFrame
    segment_info: SegmentInfo | None
    alignment_info: AlignmentInfo | None
    motion_frequency: float | None
    metric_arrays: MetricArrays = field(default_factory=dict)
    posthoc_adaptive_aae_bpm: float = float("nan")
    posthoc_adaptive_acc_pct: float = float("nan")
    posthoc_adaptive_hit_rate_5bpm: float = float("nan")
    posthoc_final_aae_bpm: float = float("nan")
    posthoc_final_acc_pct: float = float("nan")
    posthoc_baseline_aae_bpm: float = float("nan")
    posthoc_baseline_acc_pct: float = float("nan")
    posthoc_n_valid_windows: int = 0
    time_bias_after: TimeBiasAfterResult | None = None


@dataclass
class _TrialBase:
    """Cached trial base that depends on Fs_Target and training TW."""

    dataset: ProtocolDataset
    fs: int
    segment_info: SegmentInfo
    aligned: AlignedDataset | None
    motion_frequency: float | None
    failure_reason: str = ""
    norm_window_cache: dict[str, Any] = field(default_factory=dict)
    delay_estimate_cache: dict[tuple[Any, ...], DelayEstimate] = field(default_factory=dict)
    spectral_cache: dict[tuple[Any, ...], tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)


@dataclass
class _GlobalAlignmentBase:
    """Cached global Tdelay base that is independent of training TW."""

    dataset: ProtocolDataset
    fs: int
    segment_info: SegmentInfo
    tdelay_result: TDelayEstimateResult | None
    failure_reason: str = ""
    cache_hit: bool = False


@dataclass
class _NormalisedWindowCache:
    """Compact normalized-window cache for one sample/Fs_Target/TW base."""

    norm_by_channel: dict[str, np.ndarray]
    adaptive_norm_by_channel: dict[str, np.ndarray]
    window_idx: np.ndarray
    start_idx: np.ndarray
    adaptive_start_idx: np.ndarray
    adaptive_source_start_idx: np.ndarray
    fft_offset_idx: np.ndarray
    start_s: np.ndarray
    center_s: np.ndarray
    segment_label: np.ndarray
    ref_hr_bpm: np.ndarray
    win_len: int
    adaptive_win_len: int
    tw_f_context_status: np.ndarray


@dataclass
class _WindowRunPayload:
    """Window-loop output in either light or full collection mode."""

    frame: pd.DataFrame
    metric_arrays: MetricArrays


def run_protocol_trial(
    dataset: ProtocolDataset,
    cascade_scheme: CascadeScheme | str,
    target_scope: TargetScope | str,
    params: ProtocolTrialParams,
    collect_frame: bool = True,
    collect_stages: bool = True,
) -> ProtocolRunResult:
    """Run one concrete protocol trial and return AAE/accuracy metrics.

    中文说明：自适应滤波权重只在单个窗口内在线更新，不跨窗口、不跨文件保存。
    因此这里的“训练”本质是评估一组超参数，而不是拟合持久模型。

    ``collect_frame=False`` 是 Optuna trial 阶段使用的轻量路径：不构造完整
    DataFrame，也不序列化 stage JSON；但仍保留统一的 metric_arrays，因此 AAE
    和 accuracy 与 full 模式共用同一套计算逻辑。
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

        payload = _run_windows(
            base,
            scheme,
            scope,
            params,
            base.motion_frequency,
            collect_frame=bool(collect_frame),
            collect_stages=bool(collect_stages),
        )
        arrays = payload.metric_arrays
        if arrays.get("ref_hr_bpm", np.asarray([], dtype=float)).size == 0:
            return _failed(
                scope,
                scheme,
                params,
                "no valid windows after alignment",
                base.segment_info,
                base.aligned.alignment_info,
            )

        filtered_mask = _filtered_segment_mask(
            arrays["segment_label"].astype(str),
            scope,
            arrays["time_s"].astype(float),
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

        adaptive = arrays["adaptive_hr_bpm"].astype(float, copy=True)
        adaptive[filtered_mask] = smoothdata_movmedian(
            adaptive[filtered_mask],
            int(params.smooth_win_len),
        )
        fusion = fuse_final_hr(
            time_s=arrays["time_s"].astype(float),
            baseline_hr_bpm=arrays["baseline_hr_bpm"].astype(float),
            adaptive_hr_bpm=adaptive,
            segment_label=arrays["segment_label"].astype(object),
            qc_status=arrays.get("qc_status", np.full(adaptive.size, "ok", dtype=object)),
            adaptive_filter=adaptive_filter,
            motion_end_s=(
                float(base.aligned.segment_info.motion_end_s)
                if base.aligned.segment_info is not None
                else float("nan")
            ),
            params=params,
            target_scope=scope.value,
        )
        baseline_abs_err = np.abs(arrays["baseline_hr_bpm"].astype(float) - arrays["ref_hr_bpm"].astype(float))
        adaptive_abs_err = np.abs(adaptive - arrays["ref_hr_bpm"].astype(float))
        final_abs_err = np.abs(fusion.final_hr_bpm - arrays["ref_hr_bpm"].astype(float))
        metric_arrays: MetricArrays = {
            **arrays,
            "adaptive_hr_bpm": adaptive,
            "final_hr_bpm": fusion.final_hr_bpm,
            "final_source": fusion.final_source,
            "fusion_reason": fusion.fusion_reason,
            "baseline_abs_err_bpm": baseline_abs_err,
            "adaptive_abs_err_bpm": adaptive_abs_err,
            "final_abs_err_bpm": final_abs_err,
            "filtered_mask": filtered_mask,
        }
        time_bias_after = _attach_time_bias_after_metrics(
            metric_arrays,
            base.dataset,
            params,
            filtered_mask,
        )

        metrics = aggregate_metric_arrays(metric_arrays, split_name="")
        baseline_aae = float(metrics["baseline_aae_bpm"])
        adaptive_aae = float(metrics["adaptive_aae_bpm"])
        final_aae = float(metrics["final_aae_bpm"])
        baseline_acc = float(metrics["baseline_acc_pct"])
        adaptive_acc = float(metrics["adaptive_acc_pct"])
        final_acc = float(metrics["final_acc_pct"])

        frame = payload.frame
        if collect_frame:
            frame["is_filtered_segment"] = filtered_mask
            frame["adaptive_hr_bpm"] = adaptive
            frame["baseline_hr_bpm"] = arrays["baseline_hr_bpm"].astype(float)
            frame["final_hr_bpm"] = fusion.final_hr_bpm
            frame["final_source"] = fusion.final_source
            frame["fusion_reason"] = fusion.fusion_reason
            frame["is_recovery"] = frame["segment_label"].astype(str) == "recovery"
            frame["baseline_abs_err_bpm"] = baseline_abs_err
            frame["adaptive_abs_err_bpm"] = adaptive_abs_err
            frame["final_abs_err_bpm"] = final_abs_err
            frame["baseline_aae_bpm"] = baseline_aae
            frame["baseline_acc_pct"] = baseline_acc
            frame["adaptive_aae_bpm"] = adaptive_aae
            frame["adaptive_acc_pct"] = adaptive_acc
            frame["final_aae_bpm"] = final_aae
            frame["final_acc_pct"] = final_acc
            frame["best_tdelay_s"] = (
                float(base.aligned.alignment_info.best_tdelay_s)
                if base.aligned.alignment_info is not None
                else float("nan")
            )
            if "ref_hr_after_bpm" in metric_arrays:
                frame["ref_hr_after_bpm"] = metric_arrays["ref_hr_after_bpm"]
                frame["final_abs_err_after_bpm"] = metric_arrays["final_abs_err_after_bpm"]
                frame["adaptive_abs_err_after_bpm"] = metric_arrays["adaptive_abs_err_after_bpm"]
                frame["baseline_abs_err_after_bpm"] = metric_arrays["baseline_abs_err_after_bpm"]
                frame["time_bias_after_s"] = metric_arrays["time_bias_after_s"]
                frame["posthoc_alignment_mode"] = str(
                    time_bias_after.mode if time_bias_after is not None else ""
                )
                frame["posthoc_adaptive_aae_bpm"] = metrics.get("posthoc_adaptive_aae_bpm", float("nan"))
                frame["posthoc_adaptive_acc_pct"] = metrics.get("posthoc_adaptive_acc_pct", float("nan"))
                frame["posthoc_adaptive_hit_rate_5bpm"] = metrics.get(
                    "posthoc_adaptive_hit_rate_5bpm",
                    float("nan"),
                )
                frame["posthoc_final_aae_bpm"] = metrics.get("posthoc_final_aae_bpm", float("nan"))
                frame["posthoc_final_acc_pct"] = metrics.get("posthoc_final_acc_pct", float("nan"))

        return ProtocolRunResult(
            success=True,
            reason="",
            target_scope=scope,
            cascade_scheme=scheme,
            adaptive_filter=adaptive_filter,
            params=params,
            objective_aae_bpm=final_aae,
            baseline_aae_bpm=baseline_aae,
            adaptive_aae_bpm=adaptive_aae,
            final_aae_bpm=final_aae,
            baseline_acc_pct=baseline_acc,
            adaptive_acc_pct=adaptive_acc,
            final_acc_pct=final_acc,
            frame=frame,
            segment_info=base.aligned.segment_info,
            alignment_info=base.aligned.alignment_info,
            motion_frequency=base.motion_frequency,
            metric_arrays=metric_arrays,
            posthoc_adaptive_aae_bpm=float(metrics.get("posthoc_adaptive_aae_bpm", float("nan"))),
            posthoc_adaptive_acc_pct=float(metrics.get("posthoc_adaptive_acc_pct", float("nan"))),
            posthoc_adaptive_hit_rate_5bpm=float(
                metrics.get("posthoc_adaptive_hit_rate_5bpm", float("nan"))
            ),
            posthoc_final_aae_bpm=float(metrics.get("posthoc_final_aae_bpm", float("nan"))),
            posthoc_final_acc_pct=float(metrics.get("posthoc_final_acc_pct", float("nan"))),
            posthoc_baseline_aae_bpm=float(metrics.get("posthoc_baseline_aae_bpm", float("nan"))),
            posthoc_baseline_acc_pct=float(metrics.get("posthoc_baseline_acc_pct", float("nan"))),
            posthoc_n_valid_windows=int(metrics.get("posthoc_n_valid_windows", 0) or 0),
            time_bias_after=time_bias_after,
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
            final_aae_bpm=float("nan"),
            baseline_acc_pct=float("nan"),
            adaptive_acc_pct=float("nan"),
            final_acc_pct=float("nan"),
            frame=empty,
            segment_info=None,
            alignment_info=None,
            motion_frequency=None,
            metric_arrays={},
        )


def _attach_time_bias_after_metrics(
    metric_arrays: MetricArrays,
    dataset: ProtocolDataset,
    params: ProtocolTrialParams,
    filtered_mask: np.ndarray,
) -> TimeBiasAfterResult | None:
    """Attach post-hoc curve-level alignment arrays to one completed run.

    中文说明：这里的后对齐只读取已经生成的 ``adaptive_hr_bpm`` 和
    ``baseline_hr_bpm``，然后在原始参考 HR 曲线上按 ``time_s + bias``
    取样。它不会重新切窗、不会改写 PPG/补偿信号，也不会重新运行自适应滤波。
    """

    if not bool(getattr(params, "Enable_Time_Bias_After", True)):
        return None
    time_s = np.asarray(metric_arrays.get("time_s", []), dtype=float)
    adaptive = np.asarray(metric_arrays.get("adaptive_hr_bpm", []), dtype=float)
    baseline = np.asarray(metric_arrays.get("baseline_hr_bpm", []), dtype=float)
    final = np.asarray(metric_arrays.get("final_hr_bpm", adaptive), dtype=float)
    mask = np.asarray(filtered_mask, dtype=bool)
    n = min(time_s.size, adaptive.size, baseline.size, final.size, mask.size)
    if n == 0:
        return None
    time_s = time_s[:n]
    adaptive = adaptive[:n]
    baseline = baseline[:n]
    final = final[:n]
    mask = mask[:n]
    search_range = tuple(float(x) for x in getattr(params, "Time_Bias_After_Range_S", (-5.0, 5.0)))
    if len(search_range) != 2:
        raise ValueError("Time_Bias_After_Range_S must contain exactly two values")
    result = search_time_bias_after(
        time_s[mask],
        final[mask],
        np.asarray(dataset.ref_time_s, dtype=float),
        np.asarray(dataset.ref_hr_bpm, dtype=float),
        search_range_s=(float(search_range[0]), float(search_range[1])),
        search_step_s=float(getattr(params, "Time_Bias_After_Step_S", 1.0)),
        min_valid=2,
        mode=str(getattr(params, "Time_Bias_After_Mode", "posthoc_oracle_alignment")),
    )
    bias = float(result.time_bias_after_s)
    ref_after = np.interp(
        time_s + bias,
        np.asarray(dataset.ref_time_s, dtype=float),
        np.asarray(dataset.ref_hr_bpm, dtype=float),
        left=np.nan,
        right=np.nan,
    )
    # 中文说明：将标量 bias 重复成窗口等长数组，方便 LOGO/多样本聚合时逐窗拼接；
    # 聚合函数只聚合已应用各自样本 bias 后的误差，不会重新搜索全局 bias。
    metric_arrays["ref_hr_after_bpm"] = ref_after
    metric_arrays["final_abs_err_after_bpm"] = np.abs(final - ref_after)
    metric_arrays["adaptive_abs_err_after_bpm"] = np.abs(adaptive - ref_after)
    metric_arrays["baseline_abs_err_after_bpm"] = np.abs(baseline - ref_after)
    metric_arrays["time_bias_after_s"] = np.full(n, bias, dtype=float)
    metric_arrays["time_bias_after_n_valid"] = np.full(n, int(result.n_valid), dtype=float)
    metric_arrays["time_bias_after_step_s"] = np.full(n, float(result.search_step_s), dtype=float)
    metric_arrays["time_bias_after_range_s"] = np.full(
        n,
        f"({result.search_range_s[0]:.1f}, {result.search_range_s[1]:.1f})",
        dtype=object,
    )
    metric_arrays["time_bias_after_mode"] = np.full(n, str(result.mode), dtype=object)
    return result


def _get_trial_base(dataset: ProtocolDataset, params: ProtocolTrialParams) -> _TrialBase:
    """Return trial windows cached by Fs_Target and training TW.

    中文说明：全局 Tdelay 先用固定 Alignment_TW 独立估计并缓存，不依赖
    ``params.TW``；随后才用贝叶斯 trial 的 ``params.TW`` 构建训练/验证/测试窗口。
    因此不同训练 TW 可以复用同一个 best_tdelay_s，但仍得到不同训练窗口。
    """

    fs_target = int(params.Fs_Target)
    tw = float(params.TW)
    align_key = _global_tdelay_cache_key(dataset, params)
    key = (fs_target, tw, align_key, "tracked_tdelay_train_windows_v2")
    cache = getattr(dataset, "_trial_base_cache", None)
    if cache is None:
        cache = {}
        setattr(dataset, "_trial_base_cache", cache)
    if key in cache:
        # 中文注释：小型 LRU，避免 Notebook 长时间运行后保留过多 Fs/TW 大数组。
        base = cache.pop(key)
        cache[key] = base
        return base

    global_base = _get_global_alignment_base(dataset, params)
    ds = global_base.dataset
    fs = int(global_base.fs)
    segment = global_base.segment_info
    if global_base.failure_reason:
        base = _TrialBase(
            dataset=ds,
            fs=fs,
            segment_info=segment,
            aligned=None,
            motion_frequency=None,
            failure_reason=global_base.failure_reason,
        )
        _store_trial_base(cache, key, base)
        return base

    try:
        if global_base.tdelay_result is None:
            raise RuntimeError("global Tdelay cache is empty")
        aligned = build_aligned_training_windows(
            ds,
            segment,
            fs,
            train_TW=tw,
            best_tdelay_s=global_base.tdelay_result.best_tdelay_s,
            tdelay_result=global_base.tdelay_result,
        )
    except Exception as exc:
        base = _TrialBase(
            dataset=ds,
            fs=fs,
            segment_info=segment,
            aligned=None,
            motion_frequency=None,
            failure_reason=f"alignment failed: {exc}",
        )
        _store_trial_base(cache, key, base)
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
        _store_trial_base(cache, key, base)
        return base

    fmove = estimate_motion_frequency(ds.accx, ds.accy, ds.accz, aligned.segment_info, fs)
    base = _TrialBase(
        dataset=ds,
        fs=fs,
        segment_info=aligned.segment_info,
        aligned=aligned,
        motion_frequency=fmove,
    )
    _store_trial_base(cache, key, base)
    return base


def _get_global_alignment_base(dataset: ProtocolDataset, params: ProtocolTrialParams) -> _GlobalAlignmentBase:
    """Return resampling/segmentation/global Tdelay cached independently of train_TW."""

    key = _global_tdelay_cache_key(dataset, params)
    cache = getattr(dataset, "_global_tdelay_cache", None)
    if cache is None:
        cache = {}
        setattr(dataset, "_global_tdelay_cache", cache)
    if key in cache:
        cached = cache.pop(key)
        cached.cache_hit = True
        cache[key] = cached
        return cached

    fs_target = int(params.Fs_Target)
    # 中文说明：PPG 输入策略会改变后续静息段对齐、分段窗口和最终 HR，
    # 必须在重采样前对源 dataset 应用，并由 _global_tdelay_cache_key 隔离缓存。
    transformed_dataset = apply_ppg_input_transform(dataset, params)
    ds = resample_protocol_dataset(transformed_dataset, fs_target)
    fs = int(ds.fs)
    alignment_tw = float(getattr(params, "Alignment_TW", 8.0))
    # 中文说明：分段用于界定静息段和运动边界，因此也固定使用 Alignment_TW，
    # 避免 train_TW 改变时牵连全局 Tdelay 估计。
    segment = detect_activity_segments(ds.accx, ds.accy, ds.accz, fs, alignment_tw)
    if not segment.is_valid:
        base = _GlobalAlignmentBase(
            dataset=ds,
            fs=fs,
            segment_info=segment,
            tdelay_result=None,
            failure_reason=f"segmentation failed: {segment.reason}",
        )
        _store_global_alignment_base(cache, key, base)
        return base

    try:
        tdelay = estimate_global_tdelay_from_rest(
            ds,
            segment,
            fs,
            alignment_TW=alignment_tw,
            alignment_step_s=float(getattr(params, "Alignment_Step", 1.0)),
            delay_range_s=None,
            delay_step_s=0.1,
            rest_hr_kwargs=_rest_hr_kwargs_from_params(params),
            alignment_score_mode=str(getattr(params, "Rest_Alignment_Score_Mode", "aae")),
            return_debug=False,
        )
    except Exception as exc:
        base = _GlobalAlignmentBase(
            dataset=ds,
            fs=fs,
            segment_info=segment,
            tdelay_result=None,
            failure_reason=f"alignment failed: {exc}",
        )
        _store_global_alignment_base(cache, key, base)
        return base

    base = _GlobalAlignmentBase(
        dataset=ds,
        fs=fs,
        segment_info=segment,
        tdelay_result=tdelay,
        cache_hit=False,
    )
    _store_global_alignment_base(cache, key, base)
    return base


def _store_global_alignment_base(cache: dict[Any, _GlobalAlignmentBase], key: Any, base: _GlobalAlignmentBase) -> None:
    """Store global Tdelay cache with a small LRU cap independent of train_TW."""

    cache[key] = base
    while len(cache) > 9:
        cache.pop(next(iter(cache)))


def _global_tdelay_cache_key(dataset: ProtocolDataset, params: ProtocolTrialParams) -> tuple[Any, ...]:
    """Build a Tdelay cache key that intentionally excludes params.TW."""

    alignment_tw = float(getattr(params, "Alignment_TW", 8.0))
    delay_range = (-min(5.0, alignment_tw / 2.0), 5.0)
    return (
        str(dataset.sample_stem),
        int(params.Fs_Target),
        round(alignment_tw, 8),
        round(float(getattr(params, "Alignment_Step", 1.0)), 8),
        tuple(round(float(x), 8) for x in delay_range),
        round(0.1, 8),
        tuple(round(float(x), 8) for x in getattr(params, "Rest_HR_Band_BPM", (40.0, 180.0))),
        round(float(getattr(params, "Rest_HR_Track_Band_BPM", 30.0)), 8),
        round(float(getattr(params, "Rest_HR_Slew_Limit_BPM", 6.0)), 8),
        round(float(getattr(params, "Rest_HR_Slew_Step_BPM", 4.0)), 8),
        str(getattr(params, "Rest_HR_Smooth_Method", "median")).lower(),
        int(getattr(params, "Rest_HR_Smooth_Win", 3)),
        round(float(getattr(params, "Rest_HR_Peak_Percent", 0.3)), 8),
        bool(getattr(params, "Rest_HR_Spec_Penalty_Enable", True)),
        round(float(getattr(params, "Rest_HR_Spec_Penalty_Weight", 0.2)), 8),
        round(float(getattr(params, "Rest_HR_Spec_Penalty_Width_Hz", 0.2)), 8),
        _normalise_alignment_score_mode(getattr(params, "Rest_Alignment_Score_Mode", "aae")),
        str(getattr(params, "ppg_input_transform", "raw_bandpass")).lower(),
        str(getattr(params, "log_absorbance_baseline_mode", "rolling_median")).lower(),
        round(float(getattr(params, "log_absorbance_baseline_window_s", 5.0)), 8),
        round(float(getattr(params, "log_absorbance_eps", 1e-6)), 12),
        tuple(round(float(x), 12) for x in getattr(params, "log_absorbance_ratio_clip", (1e-3, 1e3))),
        "tracked_rest_hr_tdelay_v3",
    )


def _rest_hr_kwargs_from_params(params: ProtocolTrialParams) -> dict[str, Any]:
    """Collect rest-HR extraction knobs from non-Optuna alignment params."""

    return {
        "hr_band_bpm": tuple(float(x) for x in getattr(params, "Rest_HR_Band_BPM", (40.0, 180.0))),
        "track_band_bpm": float(getattr(params, "Rest_HR_Track_Band_BPM", 30.0)),
        "slew_limit_bpm": float(getattr(params, "Rest_HR_Slew_Limit_BPM", 6.0)),
        "slew_step_bpm": float(getattr(params, "Rest_HR_Slew_Step_BPM", 4.0)),
        "smooth_method": str(getattr(params, "Rest_HR_Smooth_Method", "median")),
        "smooth_win": int(getattr(params, "Rest_HR_Smooth_Win", 3)),
        "peak_percent": float(getattr(params, "Rest_HR_Peak_Percent", 0.3)),
        "spec_penalty_enable": bool(getattr(params, "Rest_HR_Spec_Penalty_Enable", True)),
        "spec_penalty_weight": float(getattr(params, "Rest_HR_Spec_Penalty_Weight", 0.2)),
        "spec_penalty_width_hz": float(getattr(params, "Rest_HR_Spec_Penalty_Width_Hz", 0.2)),
    }


def _normalise_alignment_score_mode(value: Any) -> str:
    """Normalise the rest-alignment score mode used in cache keys and logs."""

    mode = str(value).lower()
    if mode == "mae":
        mode = "aae"
    if mode not in {"aae", "std"}:
        raise ValueError("Rest_Alignment_Score_Mode must be 'aae', 'mae', or 'std'")
    return mode


def _log_global_alignment_once(
    dataset: ProtocolDataset,
    params: ProtocolTrialParams,
    tdelay: TDelayEstimateResult,
    *,
    train_tw: float,
    cache_hit: bool,
) -> None:
    """Print a concise Chinese global-alignment diagnostic once per train TW."""

    alignment_tw = float(getattr(params, "Alignment_TW", 8.0))
    delay_range = (-min(5.0, alignment_tw / 2.0), 5.0)
    key = (
        str(dataset.sample_stem),
        int(params.Fs_Target),
        round(alignment_tw, 3),
        round(float(train_tw), 3),
        _normalise_alignment_score_mode(getattr(params, "Rest_Alignment_Score_Mode", "aae")),
        bool(cache_hit),
    )
    if key in _PRINTED_GLOBAL_ALIGNMENT_KEYS:
        return
    _PRINTED_GLOBAL_ALIGNMENT_KEYS.add(key)
    best_row = (
        tdelay.score_table.loc[
            np.isclose(tdelay.score_table["delay_s"].to_numpy(dtype=float), float(tdelay.best_tdelay_s))
        ].head(1)
        if not tdelay.score_table.empty
        else pd.DataFrame()
    )
    n_valid = int(best_row["n_valid"].iloc[0]) if not best_row.empty else 0
    score_mode = str(getattr(tdelay, "alignment_score_mode", "aae"))
    score_aae = float(getattr(tdelay, "best_score_aae", float("nan")))
    score_std = float(getattr(tdelay, "best_score_std", float("nan")))
    score_rmse = float(getattr(tdelay, "best_rmse", float("nan")))
    print(
        f"[全局对齐] group={dataset.sample_stem}, Fs={int(params.Fs_Target)}, "
        f"alignment_TW={alignment_tw:.1f}s, train_TW={float(train_tw):.1f}s, "
        f"delay_range=[{delay_range[0]:.1f}, {delay_range[1]:.1f}]s, step=0.1s, "
        f"cache={'hit' if cache_hit else 'miss'}"
    )
    print(
        f"[全局对齐] 使用静息段 tracked PPG-HR 估计 Tdelay: "
        f"best_tdelay={tdelay.best_tdelay_s:.2f}s, score_mode={score_mode}, "
        f"best_score={tdelay.best_score:.3f} BPM, score_aae={score_aae:.3f} BPM, "
        f"score_std={score_std:.3f} BPM, rmse={score_rmse:.3f} BPM, n_valid={n_valid}"
    )
    print(
        f"[全局对齐] 后续训练窗口使用 train_TW={float(train_tw):.1f}s，"
        "不使用 alignment_TW 切训练窗"
    )


def _store_trial_base(cache: dict[Any, _TrialBase], key: Any, base: _TrialBase) -> None:
    """Store one TrialBase with a tiny insertion-ordered LRU cap.

    中文说明：每个 TrialBase 可能持有归一化窗口大数组；当前默认搜索空间通常是
    3 个 Fs_Target × 3 个 TW，因此保留最近 9 个组合，兼顾复用率和内存上限。
    """

    cache[key] = base
    while len(cache) > 9:
        cache.pop(next(iter(cache)))


def clear_trial_heavy_caches(dataset: ProtocolDataset) -> None:
    """Clear trial/window caches while preserving global Tdelay alignment.

    This releases the heavy TrialBase objects and their normalized windows,
    per-window delay estimates, and spectra. It intentionally keeps
    ``_global_tdelay_cache`` so LOGO folds can reuse sample/Fs/Alignment_TW
    alignment results.
    """

    cache = getattr(dataset, "_trial_base_cache", None)
    if isinstance(cache, dict):
        cache.clear()


def clear_all_caches(dataset: ProtocolDataset) -> None:
    """Clear all dataset-level caches created by this module."""

    clear_trial_heavy_caches(dataset)
    align_cache = getattr(dataset, "_global_tdelay_cache", None)
    if isinstance(align_cache, dict):
        align_cache.clear()


def clear_trial_caches(dataset: ProtocolDataset) -> None:
    """Backward-compatible full cache clear.

    New LOGO batch code uses :func:`clear_trial_heavy_caches` after each fold
    and :func:`clear_all_caches` at motion_type boundaries.
    """

    clear_all_caches(dataset)


def extract_hr_with_penalty(
    filtered_ppg: np.ndarray,
    penalty_ref: np.ndarray,
    previous_hr: float | None,
    params: ProtocolTrialParams,
    fs: int,
    precomputed_spectrum: tuple[np.ndarray, np.ndarray] | None = None,
) -> float:
    """Extract HR in BPM with motion-frequency spectral penalties and slew limits."""

    return _extract_hr(
        filtered_ppg,
        fs,
        previous_hr,
        params,
        enable_penalty=True,
        penalty_ref=penalty_ref,
        precomputed_spectrum=precomputed_spectrum,
    )


def extract_plain_fft_hr(
    ppg: np.ndarray,
    previous_hr: float | None,
    params: ProtocolTrialParams,
    fs: int,
    precomputed_spectrum: tuple[np.ndarray, np.ndarray] | None = None,
) -> float:
    """Extract baseline HR from PPG Green without adaptive filtering or penalty."""

    return _extract_hr(
        ppg,
        fs,
        previous_hr,
        params,
        enable_penalty=False,
        penalty_ref=None,
        precomputed_spectrum=precomputed_spectrum,
    )


def _run_windows(
    base: _TrialBase,
    scheme: CascadeScheme,
    scope: TargetScope,
    params: ProtocolTrialParams,
    fmove: float,
    *,
    collect_frame: bool,
    collect_stages: bool,
) -> _WindowRunPayload:
    """Run baseline/adaptive HR extraction with optional frame/stage collection."""

    if base.aligned is None:
        return _WindowRunPayload(pd.DataFrame(), {})

    aligned = base.aligned
    ds = aligned.dataset
    fs = int(ds.fs)
    norm_cache = _get_normalised_window_cache(base, params)

    rows: list[dict[str, Any]] = []
    window_idx_out: list[int] = []
    time_out: list[float] = []
    labels_out: list[str] = []
    ref_out: list[float] = []
    baseline_out: list[float] = []
    adaptive_out: list[float] = []
    qc_status_out: list[str] = []
    qc_reason_out: list[str] = []
    qc_stat_rows: list[dict[str, Any]] = []
    prev_baseline: float | None = None
    prev_adaptive: float | None = None
    for row_idx, window_idx in enumerate(norm_cache.window_idx):
        start_idx = int(norm_cache.start_idx[row_idx])
        adaptive_source_start_idx = int(norm_cache.adaptive_source_start_idx[row_idx])
        fft_offset_idx = int(norm_cache.fft_offset_idx[row_idx])
        center_s = float(norm_cache.center_s[row_idx])
        fft_start_s = float(norm_cache.start_s[row_idx])
        fft_end_s = fft_start_s + float(params.TW)
        adaptive_start_s = fft_start_s - float(getattr(params, "TW_F", 0.0))
        adaptive_source_start_s = float(adaptive_source_start_idx) / float(fs)
        label = str(norm_cache.segment_label[row_idx])
        ref_hr = float(norm_cache.ref_hr_bpm[row_idx])
        norm = {name: values[row_idx] for name, values in norm_cache.norm_by_channel.items()}
        adaptive_norm = {name: values[row_idx] for name, values in norm_cache.adaptive_norm_by_channel.items()}
        qc_stats = _window_qc_stats(ds, start_idx, int(norm_cache.win_len))
        qc_status, qc_reason, qc_should_skip = _qc_status_for_policy(qc_stats, params)
        penalty_ref_channel = ""

        spec_key = ("baseline_ppg", int(window_idx))
        baseline_spectrum = base.spectral_cache.get(spec_key)
        if baseline_spectrum is None:
            baseline_spectrum = _spectrum(norm["ppg_green"], fs)
            base.spectral_cache[spec_key] = baseline_spectrum
        baseline_hr = extract_plain_fft_hr(
            norm["ppg_green"],
            prev_baseline,
            params,
            fs,
            precomputed_spectrum=baseline_spectrum,
        )
        prev_baseline = baseline_hr if np.isfinite(baseline_hr) else prev_baseline

        if qc_should_skip:
            adaptive_hr = float("nan") if qc_status in {"dropped", "interpolate_pending"} else baseline_hr
            adaptive_stages = []
        elif _window_in_scope(str(label), scope, float(center_s), aligned.segment_info):
            filtered_context, penalty_ref_context, adaptive_stages, penalty_ref_channel = _cascade_filter_window(
                adaptive_norm,
                scheme,
                params,
                fmove,
                fs,
                delay_cache=base.delay_estimate_cache,
                window_idx=int(window_idx),
                collect_stages=collect_stages,
                adaptive_input_samples=int(norm_cache.adaptive_win_len),
                fft_offset_samples=fft_offset_idx,
                fft_input_samples=int(norm_cache.win_len),
            )
            fft_stop_idx = fft_offset_idx + int(norm_cache.win_len)
            filtered = np.asarray(filtered_context, dtype=float)[fft_offset_idx:fft_stop_idx]
            penalty_ref = np.asarray(penalty_ref_context, dtype=float)[fft_offset_idx:fft_stop_idx]
            adaptive_hr = extract_hr_with_penalty(filtered, penalty_ref, prev_adaptive, params, fs)
        else:
            adaptive_hr = baseline_hr
            adaptive_stages = []

        if np.isfinite(adaptive_hr):
            prev_adaptive = adaptive_hr

        window_idx_out.append(int(window_idx))
        time_out.append(float(center_s))
        labels_out.append(str(label))
        ref_out.append(float(ref_hr))
        baseline_out.append(float(baseline_hr))
        adaptive_out.append(float(adaptive_hr))
        qc_status_out.append(qc_status)
        qc_reason_out.append(qc_reason)
        qc_stat_rows.append(qc_stats)

        if collect_frame:
            stages_json = json.dumps(adaptive_stages, ensure_ascii=False) if collect_stages else ""
            rows.append(
                {
                    "sample": ds.sample_stem,
                    "group_id": ds.sample_stem.removeprefix("multi_"),
                    "target_scope": scope.value,
                    "cascade_scheme": scheme.value,
                    "adaptive_filter": str(getattr(params, "adaptive_filter", "lms")),
                    "window_idx": int(window_idx),
                    "time_s": float(center_s),
                    "center_s": float(center_s),
                    "fft_start_s": float(fft_start_s),
                    "fft_end_s": float(fft_end_s),
                    "adaptive_start_s": float(adaptive_start_s),
                    "adaptive_source_start_s": float(adaptive_source_start_s),
                    "TW_F": float(getattr(params, "TW_F", 0.0)),
                    "fft_input_samples": int(norm_cache.win_len),
                    "adaptive_input_samples": int(norm_cache.adaptive_win_len),
                    "fft_offset_samples": int(fft_offset_idx),
                    "tw_f_context_status": str(norm_cache.tw_f_context_status[row_idx]),
                    "segment_label": str(label),
                    "ref_hr_bpm": float(ref_hr),
                    "baseline_ppg_hr_bpm": float(baseline_hr),
                    "adaptive_hr_bpm": float(adaptive_hr),
                    "penalty_ref_channel": penalty_ref_channel,
                    "adaptive_stages_json": stages_json,
                    "lms_stages_json": stages_json,
                    "qc_status": qc_status,
                    "qc_reason": qc_reason,
                    **qc_stats,
                }
            )

    adaptive_arr = _apply_qc_interpolation_policy(
        np.asarray(adaptive_out, dtype=float),
        np.asarray(baseline_out, dtype=float),
        np.asarray(time_out, dtype=float),
        qc_status_out,
    )
    adaptive_out = adaptive_arr.astype(float).tolist()
    if collect_frame and rows:
        for row, value, status in zip(rows, adaptive_out, qc_status_out, strict=False):
            row["adaptive_hr_bpm"] = float(value)
            if status == "interpolate_pending":
                row["qc_status"] = "interpolated"

    metric_arrays: MetricArrays = {
        "window_idx": np.asarray(window_idx_out, dtype=int),
        "time_s": np.asarray(time_out, dtype=float),
        "segment_label": np.asarray(labels_out, dtype=object),
        "ref_hr_bpm": np.asarray(ref_out, dtype=float),
        "baseline_hr_bpm": np.asarray(baseline_out, dtype=float),
        "adaptive_hr_bpm": np.asarray(adaptive_out, dtype=float),
        "center_s": np.asarray(time_out, dtype=float),
        "fft_start_s": norm_cache.start_s.astype(float),
        "fft_end_s": norm_cache.start_s.astype(float) + float(params.TW),
        "adaptive_start_s": norm_cache.start_s.astype(float) - float(getattr(params, "TW_F", 0.0)),
        "adaptive_source_start_s": norm_cache.adaptive_source_start_idx.astype(float) / float(fs),
        "fft_input_samples": np.full(len(time_out), int(norm_cache.win_len), dtype=int),
        "adaptive_input_samples": np.full(len(time_out), int(norm_cache.adaptive_win_len), dtype=int),
        "fft_offset_samples": norm_cache.fft_offset_idx.astype(int),
        "tw_f_context_status": norm_cache.tw_f_context_status.astype(object),
        "qc_status": np.asarray(
            ["interpolated" if s == "interpolate_pending" else s for s in qc_status_out],
            dtype=object,
        ),
        "qc_reason": np.asarray(qc_reason_out, dtype=object),
    }
    for key in _QC_NUMERIC_FIELDS:
        metric_arrays[key] = np.asarray([float(row.get(key, np.nan)) for row in qc_stat_rows], dtype=float)
    return _WindowRunPayload(pd.DataFrame(rows) if collect_frame else pd.DataFrame(), metric_arrays)


_QC_NUMERIC_FIELDS = (
    "missing_ratio",
    "max_consecutive_missing",
    "ValidFlag_ratio",
    "InterpFlag_ratio",
    "GapLen_max",
    "GapLen_mean",
    "MissingBefore_max",
    "MissingBefore_mean",
    "qc_bad_window",
)


def _window_qc_stats(dataset: ProtocolDataset, start_idx: int, win_len: int) -> dict[str, Any]:
    """Summarise sample-level QC metadata inside one FFT/adaptive window.

    中文说明：统计只依赖传感器 CSV 的 QC/缺失元数据，不读取参考 HR；后续融合策略
    可以用这些无监督质量信号做保守回退。
    """

    qc = dataset.qc_frame()
    start = max(0, int(start_idx))
    end = min(len(qc), start + max(0, int(win_len)))
    if end <= start:
        return {
            "missing_ratio": float("nan"),
            "max_consecutive_missing": 0.0,
            "ValidFlag_ratio": float("nan"),
            "InterpFlag_ratio": float("nan"),
            "GapLen_max": float("nan"),
            "GapLen_mean": float("nan"),
            "MissingBefore_max": float("nan"),
            "MissingBefore_mean": float("nan"),
            "qc_bad_window": 1.0,
        }
    window = qc.iloc[start:end]
    valid_flag = pd.to_numeric(window["ValidFlag"], errors="coerce").to_numpy(dtype=float)
    interp_flag = pd.to_numeric(window["InterpFlag"], errors="coerce").to_numpy(dtype=float)
    gap_len = pd.to_numeric(window["GapLen"], errors="coerce").to_numpy(dtype=float)
    missing_before = pd.to_numeric(window["MissingBefore"], errors="coerce").to_numpy(dtype=float)
    raw_missing = pd.to_numeric(window["raw_missing_any"], errors="coerce").fillna(0).to_numpy(dtype=float) > 0
    invalid = np.isfinite(valid_flag) & (valid_flag <= 0)
    interpolated = np.isfinite(interp_flag) & (interp_flag > 0)
    missing_mask = raw_missing | invalid | interpolated
    missing_ratio = float(np.mean(missing_mask)) if missing_mask.size else float("nan")
    valid_ratio = float(np.nanmean(valid_flag > 0)) if valid_flag.size else float("nan")
    interp_ratio = float(np.nanmean(interp_flag > 0)) if interp_flag.size else float("nan")
    max_gap = float(np.nanmax(gap_len)) if gap_len.size else float("nan")
    bad = (
        (np.isfinite(missing_ratio) and missing_ratio > 0.20)
        or (np.isfinite(valid_ratio) and valid_ratio < 0.80)
        or (np.isfinite(interp_ratio) and interp_ratio > 0.50)
        or (np.isfinite(max_gap) and max_gap > 0 and np.isfinite(missing_ratio) and missing_ratio > 0.0)
    )
    return {
        "missing_ratio": missing_ratio,
        "max_consecutive_missing": float(_max_consecutive_true(missing_mask)),
        "ValidFlag_ratio": valid_ratio,
        "InterpFlag_ratio": interp_ratio,
        "GapLen_max": max_gap,
        "GapLen_mean": float(np.nanmean(gap_len)) if gap_len.size else float("nan"),
        "MissingBefore_max": float(np.nanmax(missing_before)) if missing_before.size else float("nan"),
        "MissingBefore_mean": float(np.nanmean(missing_before)) if missing_before.size else float("nan"),
        "qc_bad_window": float(bad),
    }


def _qc_status_for_policy(qc_stats: dict[str, Any], params: ProtocolTrialParams) -> tuple[str, str, bool]:
    """Return status/reason and whether adaptive filtering should be skipped."""

    bad = bool(float(qc_stats.get("qc_bad_window", 0.0)) > 0.5)
    if not bad:
        return "ok", "", False
    policy = str(getattr(params, "qc_policy", "fallback_baseline")).lower()
    if policy not in {"keep", "fallback_baseline", "drop", "interpolate"}:
        policy = "fallback_baseline"
    reasons = []
    if float(qc_stats.get("missing_ratio", 0.0)) > 0.20:
        reasons.append(f"missing_ratio={float(qc_stats['missing_ratio']):.3f}")
    if float(qc_stats.get("ValidFlag_ratio", 1.0)) < 0.80:
        reasons.append(f"ValidFlag_ratio={float(qc_stats['ValidFlag_ratio']):.3f}")
    if float(qc_stats.get("InterpFlag_ratio", 0.0)) > 0.50:
        reasons.append(f"InterpFlag_ratio={float(qc_stats['InterpFlag_ratio']):.3f}")
    if float(qc_stats.get("GapLen_max", 0.0)) > 0:
        reasons.append(f"GapLen_max={float(qc_stats['GapLen_max']):.1f}")
    reason = "bad_window: " + ", ".join(reasons or ["quality thresholds exceeded"])
    if policy == "keep":
        return "bad_kept", reason + "; policy=keep", False
    if policy == "drop":
        return "dropped", reason + "; policy=drop", True
    if policy == "interpolate":
        return "interpolate_pending", reason + "; policy=interpolate", True
    return "fallback_baseline", reason + "; policy=fallback_baseline", True


def _apply_qc_interpolation_policy(
    adaptive: np.ndarray,
    baseline: np.ndarray,
    time_s: np.ndarray,
    qc_status: list[str],
) -> np.ndarray:
    """Fill ``interpolate`` QC windows from neighbouring adaptive HR values."""

    out = np.asarray(adaptive, dtype=float).copy()
    interp_mask = np.asarray([s == "interpolate_pending" for s in qc_status], dtype=bool)
    if not interp_mask.any():
        return out
    x = np.asarray(time_s, dtype=float)
    good = np.isfinite(out) & ~interp_mask
    if good.sum() >= 2:
        out[interp_mask] = np.interp(x[interp_mask], x[good], out[good])
    else:
        out[interp_mask] = np.asarray(baseline, dtype=float)[interp_mask]
    return out


def _max_consecutive_true(mask: np.ndarray) -> int:
    """Return the longest run of true values in a boolean mask."""

    best = 0
    current = 0
    for value in np.asarray(mask, dtype=bool):
        current = current + 1 if bool(value) else 0
        best = max(best, current)
    return int(best)


def _get_normalised_window_cache(base: _TrialBase, params: ProtocolTrialParams) -> _NormalisedWindowCache:
    """Return compact per-window min-max normalized arrays for one TrialBase.

    中文说明：这个缓存只依赖样本、Fs_Target、TW 和对齐后的窗口边界；不依赖级联
    方案、滤波器类型或 Optuna 超参数，因此可以在同一 TrialBase 下安全复用。
    """

    mode = str(getattr(params, "normalization_mode", "minmax")).lower()
    key = ("normalised_protocol_windows_v2", mode)
    cached = base.norm_window_cache.get(key)
    if isinstance(cached, _NormalisedWindowCache):
        return cached
    if base.aligned is None:
        empty = _NormalisedWindowCache(
            norm_by_channel={name: np.empty((0, 0), dtype=np.float32) for name in PROTOCOL_CHANNELS},
            adaptive_norm_by_channel={name: np.empty((0, 0), dtype=np.float32) for name in PROTOCOL_CHANNELS},
            window_idx=np.empty(0, dtype=int),
            start_idx=np.empty(0, dtype=int),
            adaptive_start_idx=np.empty(0, dtype=int),
            adaptive_source_start_idx=np.empty(0, dtype=int),
            fft_offset_idx=np.empty(0, dtype=int),
            start_s=np.empty(0, dtype=float),
            center_s=np.empty(0, dtype=float),
            segment_label=np.empty(0, dtype=object),
            ref_hr_bpm=np.empty(0, dtype=float),
            win_len=0,
            adaptive_win_len=0,
            tw_f_context_status=np.empty(0, dtype=object),
        )
        base.norm_window_cache[key] = empty
        return empty

    aligned = base.aligned
    ds = aligned.dataset
    fs = int(ds.fs)
    win_len = int(round(float(params.TW) * fs))
    tw_f_len = max(0, int(round(float(getattr(params, "TW_F", 0.0)) * fs)))
    adaptive_win_len = win_len + tw_f_len
    starts = np.rint(aligned.window_starts_s.astype(float) * fs).astype(int)
    valid = starts + win_len <= len(ds.time_s)
    valid_idx = np.flatnonzero(valid)
    starts = starts[valid]
    channels = ds.channels()
    n_windows = int(starts.size)
    norm_by_channel = {
        name: np.zeros((n_windows, win_len), dtype=np.float32)
        for name in PROTOCOL_CHANNELS
    }
    adaptive_norm_by_channel = {
        name: np.zeros((n_windows, adaptive_win_len), dtype=np.float32)
        for name in PROTOCOL_CHANNELS
    }
    adaptive_starts = starts - tw_f_len
    adaptive_source_starts = np.maximum(0, adaptive_starts)
    fft_offsets = starts - adaptive_starts
    context_status = np.where(adaptive_starts < 0, "padded_left", "full").astype(object)

    for out_idx, start in enumerate(starts):
        end = int(start) + win_len
        adaptive_start = int(adaptive_starts[out_idx])
        source_start = int(adaptive_source_starts[out_idx])
        left_pad = max(0, -adaptive_start)
        for name in PROTOCOL_CHANNELS:
            arr = np.asarray(channels[name][start:end], dtype=float)
            norm_by_channel[name][out_idx] = _normalise_array(arr, mode=mode).astype(np.float32, copy=False)
            context = np.asarray(channels[name][source_start:end], dtype=float)
            if left_pad:
                pad_value = float(context[0]) if context.size else 0.0
                context = np.concatenate([np.full(left_pad, pad_value, dtype=float), context])
            if context.size < adaptive_win_len:
                pad_value = float(context[-1]) if context.size else 0.0
                context = np.concatenate([context, np.full(adaptive_win_len - context.size, pad_value, dtype=float)])
            adaptive_norm_by_channel[name][out_idx] = _normalise_array(
                context[:adaptive_win_len],
                mode=mode,
            ).astype(np.float32, copy=False)

    cache = _NormalisedWindowCache(
        norm_by_channel=norm_by_channel,
        adaptive_norm_by_channel=adaptive_norm_by_channel,
        window_idx=valid_idx.astype(int),
        start_idx=starts.astype(int),
        adaptive_start_idx=adaptive_starts.astype(int),
        adaptive_source_start_idx=adaptive_source_starts.astype(int),
        fft_offset_idx=fft_offsets.astype(int),
        start_s=aligned.window_starts_s[valid].astype(float),
        center_s=aligned.window_centers_s[valid].astype(float),
        segment_label=aligned.segment_labels[valid].astype(object),
        ref_hr_bpm=aligned.ref_hr_bpm[valid].astype(float),
        win_len=win_len,
        adaptive_win_len=adaptive_win_len,
        tw_f_context_status=context_status,
    )
    base.norm_window_cache[key] = cache
    return cache


def _cascade_filter_window(
    window: dict[str, np.ndarray],
    scheme: CascadeScheme,
    params: ProtocolTrialParams,
    fmove: float,
    fs: int,
    *,
    delay_cache: dict[tuple[Any, ...], DelayEstimate] | None = None,
    window_idx: int | None = None,
    collect_stages: bool = True,
    adaptive_input_samples: int | None = None,
    fft_offset_samples: int = 0,
    fft_input_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], str]:
    """Run the configured adaptive-filter cascade on one normalized window."""

    delay_mode = str(getattr(params, "delay_estimation_mode", "envelope"))
    delay_key = (
        int(window_idx) if window_idx is not None else -1,
        round(float(params.Kstop), 8),
        round(float(fmove), 8),
        delay_mode,
    )
    delay_est = delay_cache.get(delay_key) if delay_cache is not None else None
    if delay_est is None:
        delay_est = estimate_envelope_delays(
            window,
            fmove,
            params.Kstop,
            fs,
            mode=delay_mode,
        )
        if delay_cache is not None:
            # 中文注释：只缓存 DelayEstimate 小对象，不保存包络/相关数组，避免内存放大。
            delay_cache[delay_key] = delay_est
    current = np.asarray(window["ppg_green"], dtype=float)
    stages: list[dict[str, Any]] = []
    reference_ranking = {key: list(value) for key, value in delay_est.order_by_type.items()}
    for sensor_type, max_count in _scheme_plan(scheme):
        for channel in delay_est.order_by_type.get(sensor_type, [])[:max_count]:
            delay = delay_est.by_channel[channel]
            design = map_delay_to_lms_params(delay, sensor_type, params, fs)
            filter_type = str(getattr(params, "adaptive_filter", "lms"))
            stage_extra: dict[str, Any] = {}
            stage_mu = float(design.u)
            before_stage = np.asarray(current, dtype=float).copy()
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
                rff_update_mode = str(getattr(params, "rff_update_mode", "nlms"))
                rff_nlms_eps = float(getattr(params, "rff_nlms_eps", 1e-6))
                rff_leakage = float(getattr(params, "rff_leakage", 0.0))
                rff_err_clip = getattr(params, "rff_err_clip", None)
                rff_theta_norm_guard = getattr(params, "rff_theta_norm_guard", None)
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
                    update_mode=rff_update_mode,
                    nlms_eps=rff_nlms_eps,
                    leakage=rff_leakage,
                    err_clip=rff_err_clip,
                    theta_norm_guard=rff_theta_norm_guard,
                    return_diagnostics=False,
                )
                stage_extra.update(
                    {
                        "D": rff_D,
                        "sigma": rff_sigma,
                        "rff_seed": rff_seed,
                        "update_mode": rff_update_mode,
                        "nlms_eps": rff_nlms_eps,
                        "leakage": rff_leakage,
                        "err_clip": rff_err_clip,
                        "theta_norm_guard": rff_theta_norm_guard,
                    }
                )
            elif filter_type == "klms":
                klms_step_size = float(getattr(params, "klms_step_size", 0.05))
                klms_sigma = float(getattr(params, "klms_sigma", 1.0))
                klms_epsilon = float(getattr(params, "klms_epsilon", 0.1))
                klms_max_dictionary_size = int(getattr(params, "klms_max_dictionary_size", 300))
                klms_center_prune_policy = str(getattr(params, "klms_center_prune_policy", "freeze_new_centers"))
                klms_distance_mode = str(getattr(params, "klms_distance_mode", "normalized"))
                klms_normalized_update = bool(getattr(params, "klms_normalized_update", True))
                klms_nlms_eps = float(getattr(params, "klms_nlms_eps", 1e-6))
                stage_mu = klms_step_size
                current = noncausal_klms_filter(
                    window[channel],
                    current,
                    M=design.M,
                    K=design.K,
                    step_size=klms_step_size,
                    sigma=klms_sigma,
                    epsilon=klms_epsilon,
                    max_dictionary_size=klms_max_dictionary_size,
                    center_prune_policy=klms_center_prune_policy,
                    distance_mode=klms_distance_mode,
                    normalized_update=klms_normalized_update,
                    nlms_eps=klms_nlms_eps,
                    return_diagnostics=False,
                )
                stage_extra.update(
                    {
                        "klms_step_size": klms_step_size,
                        "sigma": klms_sigma,
                        "epsilon": klms_epsilon,
                        "max_dictionary_size": klms_max_dictionary_size,
                        "center_prune_policy": klms_center_prune_policy,
                        "distance_mode": klms_distance_mode,
                        "normalized_update": klms_normalized_update,
                        "nlms_eps": klms_nlms_eps,
                    }
                )
            else:
                raise ValueError(f"Unsupported adaptive_filter: {filter_type}")

            current, guard_record = _evaluate_cascade_rms_guard(before_stage, current, params)
            stage_extra.update(guard_record)

            if collect_stages:
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
                        "mu": stage_mu,
                        "filter_type": filter_type,
                        "mode": design.mode,
                        "delay_estimation_mode": delay_mode,
                        "reference_channel_ranking": reference_ranking,
                        "adaptive_input_samples": int(
                            adaptive_input_samples if adaptive_input_samples is not None else len(current)
                        ),
                        "fft_offset_samples": int(fft_offset_samples),
                        "fft_input_samples": int(fft_input_samples if fft_input_samples is not None else len(current)),
                        # 中文说明：collect_stages=True 只用于回放/诊断等完整输出场景；保存
                        # 每级输出可直接重画某窗口的级联波形，默认批量 Optuna 不收集该字段。
                        "output_signal": np.asarray(current, dtype=float).round(8).tolist(),
                        **stage_extra,
                    }
                )
    penalty_ref, penalty_ref_channel = _penalty_reference_with_channel(window, delay_est, scheme)
    if collect_stages:
        for stage in stages:
            stage["penalty_ref_channel"] = penalty_ref_channel
    return current, penalty_ref, stages, penalty_ref_channel


def _evaluate_cascade_rms_guard(
    before_signal: np.ndarray,
    after_signal: np.ndarray,
    params: ProtocolTrialParams,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the optional per-stage RMS guard and return short stage metrics.

    中文说明：``none`` 策略只记录 accepted=True，不改变级联输出；``rms_guard``
    使用有限 zscore 后的 RMS ratio 判断本级滤波是否异常。拒绝时回退到本级输入，
    但后续级联仍继续运行。
    """

    before = np.asarray(before_signal, dtype=float)
    after = np.asarray(after_signal, dtype=float)
    policy = str(getattr(params, "cascade_guard_policy", "none")).lower()
    record = {
        "guard_policy": policy,
        "rms_before": float("nan"),
        "rms_after": float("nan"),
        "rms_ratio": float("nan"),
        "accepted": True,
        "reject_reason": "",
    }
    if policy == "none":
        return after, record
    if policy != "rms_guard":
        raise ValueError("cascade_guard_policy must be 'none' or 'rms_guard'")

    before_eval = _finite_zscore_for_guard(before) if bool(getattr(params, "cascade_guard_use_finite_zscore", True)) else before
    after_eval = _finite_zscore_for_guard(after) if bool(getattr(params, "cascade_guard_use_finite_zscore", True)) else after
    flat_eps = float(getattr(params, "cascade_guard_flat_std_eps", 1e-6))
    ratio_min = float(getattr(params, "cascade_guard_ratio_min", 0.05))
    ratio_max = float(getattr(params, "cascade_guard_ratio_max", 5.0))
    before_finite = np.isfinite(before_eval)
    after_finite = np.isfinite(after_eval)
    if before_eval.size != after_eval.size:
        record["accepted"] = False
        record["reject_reason"] = "length_mismatch"
        return before.copy(), record
    if not before_finite.all() or not after_finite.all():
        record["accepted"] = False
        record["reject_reason"] = "non_finite_signal"
        return before.copy(), record
    rms_before = float(np.sqrt(np.mean(before_eval * before_eval))) if before_eval.size else 0.0
    rms_after = float(np.sqrt(np.mean(after_eval * after_eval))) if after_eval.size else 0.0
    record["rms_before"] = rms_before
    record["rms_after"] = rms_after
    record["rms_ratio"] = float(rms_after / rms_before) if rms_before > 0.0 else float("inf")
    if rms_before <= flat_eps:
        record["accepted"] = False
        record["reject_reason"] = "before_flat"
    elif rms_after <= flat_eps:
        record["accepted"] = False
        record["reject_reason"] = "after_flat"
    elif float(np.std(after_eval)) <= flat_eps:
        record["accepted"] = False
        record["reject_reason"] = "after_nearly_flat"
    elif not ratio_min <= record["rms_ratio"] <= ratio_max:
        record["accepted"] = False
        record["reject_reason"] = "rms_ratio_out_of_range"
    if not record["accepted"]:
        return before.copy(), record
    return after, record


def _finite_zscore_for_guard(values: np.ndarray) -> np.ndarray:
    """Return finite values for RMS guard comparison.

    中文说明：字段名沿用 finite_zscore 是为了和任务配置保持一致；这里保留原始
    幅值尺度，只替换非有限值，因为 RMS ratio 必须能识别输出幅值爆炸。
    """

    arr = np.asarray(values, dtype=float).copy()
    finite = np.isfinite(arr)
    if not finite.any():
        return np.full_like(arr, np.nan, dtype=float)
    mean = float(np.mean(arr[finite]))
    arr[~finite] = mean
    return arr


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

    ref, _ = _penalty_reference_with_channel(window, delay_est, scheme)
    return ref


def _penalty_reference_with_channel(
    window: dict[str, np.ndarray],
    delay_est: DelayEstimate,
    scheme: CascadeScheme,
) -> tuple[np.ndarray, str]:
    """Choose the spectral-penalty reference and return its channel name."""

    if scheme in {CascadeScheme.ACC3, CascadeScheme.ACC3_HF2, CascadeScheme.HF2_ACC3}:
        channel = _best_energy_channel(window, ("accx", "accy", "accz"))
        return window[channel], channel
    if scheme in {CascadeScheme.HF2, CascadeScheme.HF2_CF2, CascadeScheme.CF2_HF2}:
        channel = delay_est.primary_by_type.get("HF") or "hf1"
        return window[channel], channel
    if scheme == CascadeScheme.CF2:
        channel = delay_est.primary_by_type.get("CF") or "cf1"
        return window[channel], channel
    return window["ppg_green"], "ppg_green"


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
    precomputed_spectrum: tuple[np.ndarray, np.ndarray] | None = None,
) -> float:
    """Extract HR using Hamming FFT, optional spectral penalty, and slew limit."""

    freq, amp = precomputed_spectrum if precomputed_spectrum is not None else _spectrum(signal, fs)
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

    return compute_power_spectrum(signal, fs, apply_hamming=True, demean=True)


def _dominant_frequency(signal: np.ndarray, fs: int, low_hz: float, high_hz: float) -> float:
    """Find the dominant spectral peak in a frequency band."""

    return dominant_frequency_in_band(signal, fs, low_hz, high_hz)


def _normalise_window(window: dict[str, np.ndarray], mode: str = "minmax") -> dict[str, np.ndarray]:
    """Normalize all 13 protocol channels inside one window."""

    out: dict[str, np.ndarray] = {}
    for name in PROTOCOL_CHANNELS:
        out[name] = _normalise_array(window[name], mode=mode)
    return out


def _normalise_array(values: np.ndarray, mode: str = "minmax") -> np.ndarray:
    """Normalize one window and return a finite float array.

    中文说明：默认 ``minmax`` 完全保留旧行为；``zscore`` 和 ``none`` 是固定实验
    参数，便于对比归一化策略，但不进入 Optuna 搜索空间。
    """

    arr = np.asarray(values, dtype=float).copy()
    mode = str(mode).lower()
    if mode not in {"minmax", "zscore", "none"}:
        raise ValueError("normalization_mode must be one of: minmax, zscore, none")
    arr[~np.isfinite(arr)] = np.nan
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=float)
    if mode == "none":
        arr[~finite] = 0.0
        return arr
    if mode == "zscore":
        mu = float(np.nanmean(arr))
        sd = float(np.nanstd(arr))
        arr[~finite] = mu
        if not np.isfinite(sd) or sd <= 1e-12:
            return arr - mu
        return (arr - mu) / sd
    mn = float(np.nanmin(arr))
    mx = float(np.nanmax(arr))
    if mx - mn <= 1e-12:
        return np.zeros_like(arr, dtype=float)
    arr[~finite] = mn
    return (arr - mn) / (mx - mn)


def _label_in_scope(label: str, scope: TargetScope) -> bool:
    if scope == TargetScope.GLOBAL:
        return label in {"rest", "motion", "recovery"}
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
    if scope == TargetScope.GLOBAL:
        return str(label) in {"rest", "motion", "recovery"}
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


def aggregate_metric_arrays(metric_arrays: MetricArrays, split_name: str = "") -> dict[str, Any]:
    """Aggregate baseline/adaptive metrics from compact window arrays.

    中文说明：full DataFrame 和 light objective 都先生成同一组 ``metric_arrays``，
    再用本函数统一计算 AAE/accuracy；LOGO 汇总也会拼接 held-out 窗口后调用这里，
    因此不会退化成 fold 均值。
    """

    ref = np.asarray(metric_arrays.get("ref_hr_bpm", []), dtype=float)
    if ref.size == 0:
        return {
            "split": split_name,
            "baseline_aae_bpm": float("nan"),
            "adaptive_aae_bpm": float("nan"),
            "final_aae_bpm": float("nan"),
            "baseline_acc_pct": float("nan"),
            "adaptive_acc_pct": float("nan"),
            "final_acc_pct": float("nan"),
            "time_bias_after_s": float("nan"),
            "time_bias_after_mode": "",
            "time_bias_after_range_s": "",
            "time_bias_after_step_s": float("nan"),
            "time_bias_after_n_valid": 0,
            "posthoc_adaptive_aae_bpm": float("nan"),
            "posthoc_adaptive_acc_pct": float("nan"),
            "posthoc_adaptive_hit_rate_5bpm": float("nan"),
            "posthoc_final_aae_bpm": float("nan"),
            "posthoc_final_acc_pct": float("nan"),
            "posthoc_baseline_aae_bpm": float("nan"),
            "posthoc_baseline_acc_pct": float("nan"),
            "posthoc_n_valid_windows": 0,
            "num_windows": 0,
        }
    mask = np.asarray(metric_arrays.get("filtered_mask", np.ones(ref.size, dtype=bool)), dtype=bool)
    baseline_err = metric_arrays.get("baseline_abs_err_bpm")
    adaptive_err = metric_arrays.get("adaptive_abs_err_bpm")
    final_err = metric_arrays.get("final_abs_err_bpm")
    if baseline_err is None:
        baseline_err = np.abs(np.asarray(metric_arrays["baseline_hr_bpm"], dtype=float) - ref)
    if adaptive_err is None:
        adaptive_err = np.abs(np.asarray(metric_arrays["adaptive_hr_bpm"], dtype=float) - ref)
    if final_err is None:
        final_err = np.abs(np.asarray(metric_arrays.get("final_hr_bpm", metric_arrays["adaptive_hr_bpm"]), dtype=float) - ref)
    baseline_err = np.asarray(baseline_err, dtype=float)
    adaptive_err = np.asarray(adaptive_err, dtype=float)
    final_err = np.asarray(final_err, dtype=float)
    mask = mask[: min(mask.size, baseline_err.size, adaptive_err.size, final_err.size)]
    baseline_target = baseline_err[: mask.size][mask]
    adaptive_target = adaptive_err[: mask.size][mask]
    final_target = final_err[: mask.size][mask]
    posthoc = _aggregate_posthoc_metric_arrays(metric_arrays, mask)
    return {
        "split": split_name,
        "baseline_aae_bpm": _mean(baseline_target),
        "adaptive_aae_bpm": _mean(adaptive_target),
        "final_aae_bpm": _mean(final_target),
        "baseline_acc_pct": _accuracy(baseline_target),
        "adaptive_acc_pct": _accuracy(adaptive_target),
        "final_acc_pct": _accuracy(final_target),
        **posthoc,
        "num_windows": int(np.isfinite(final_target).sum()),
    }


def _aggregate_posthoc_metric_arrays(metric_arrays: MetricArrays, mask: np.ndarray) -> dict[str, Any]:
    """Aggregate post-hoc errors that already include per-run time bias.

    中文说明：这里不重新搜索 ``time_bias_after_s``。多样本/LOGO 聚合只是把每个
    run 已经按自身 bias 计算好的窗口级误差拼接后求指标，避免把 oracle 后对齐
    混同成一个全测试集共享的泛化参数。
    """

    adaptive_after = metric_arrays.get("adaptive_abs_err_after_bpm")
    baseline_after = metric_arrays.get("baseline_abs_err_after_bpm")
    final_after = metric_arrays.get("final_abs_err_after_bpm")
    if adaptive_after is None or baseline_after is None or final_after is None:
        return {
            "time_bias_after_s": float("nan"),
            "time_bias_after_mode": "",
            "time_bias_after_range_s": "",
            "time_bias_after_step_s": float("nan"),
            "time_bias_after_n_valid": 0,
            "posthoc_adaptive_aae_bpm": float("nan"),
            "posthoc_adaptive_acc_pct": float("nan"),
            "posthoc_adaptive_hit_rate_5bpm": float("nan"),
            "posthoc_final_aae_bpm": float("nan"),
            "posthoc_final_acc_pct": float("nan"),
            "posthoc_baseline_aae_bpm": float("nan"),
            "posthoc_baseline_acc_pct": float("nan"),
            "posthoc_n_valid_windows": 0,
        }
    adaptive_arr = np.asarray(adaptive_after, dtype=float)
    baseline_arr = np.asarray(baseline_after, dtype=float)
    final_arr = np.asarray(final_after, dtype=float)
    n = min(mask.size, adaptive_arr.size, baseline_arr.size, final_arr.size)
    target_adaptive = adaptive_arr[:n][mask[:n]]
    target_baseline = baseline_arr[:n][mask[:n]]
    target_final = final_arr[:n][mask[:n]]
    time_bias = np.asarray(metric_arrays.get("time_bias_after_s", []), dtype=float)
    finite_bias = np.unique(np.round(time_bias[np.isfinite(time_bias)], 10)) if time_bias.size else np.asarray([])
    bias_value = float(finite_bias[0]) if finite_bias.size == 1 else float("nan")
    if finite_bias.size == 0:
        mode_value = ""
    elif finite_bias.size == 1:
        mode_values = np.asarray(metric_arrays.get("time_bias_after_mode", []), dtype=object)
        finite_modes = [str(x) for x in mode_values.ravel().tolist() if str(x)]
        unique_modes = sorted(set(finite_modes))
        mode_value = unique_modes[0] if len(unique_modes) == 1 else "posthoc_oracle_alignment"
    else:
        mode_value = "per_run_posthoc_oracle_alignment"
    range_values = np.asarray(metric_arrays.get("time_bias_after_range_s", []), dtype=object)
    finite_ranges = [str(x) for x in range_values.ravel().tolist() if str(x)]
    unique_ranges = sorted(set(finite_ranges))
    step_values = np.asarray(metric_arrays.get("time_bias_after_step_s", []), dtype=float)
    finite_steps = np.unique(np.round(step_values[np.isfinite(step_values)], 10)) if step_values.size else np.asarray([])
    n_valid_arr = np.asarray(metric_arrays.get("time_bias_after_n_valid", []), dtype=float)
    finite_n_valid = n_valid_arr[np.isfinite(n_valid_arr)]
    n_valid = int(np.nanmax(finite_n_valid)) if finite_n_valid.size and finite_bias.size == 1 else int(
        np.isfinite(target_adaptive).sum()
    )
    hit_rate = _accuracy(target_adaptive)
    return {
        "time_bias_after_s": bias_value,
        "time_bias_after_mode": mode_value,
        "time_bias_after_range_s": unique_ranges[0] if len(unique_ranges) == 1 else "",
        "time_bias_after_step_s": float(finite_steps[0]) if finite_steps.size == 1 else float("nan"),
        "time_bias_after_n_valid": n_valid,
        "posthoc_adaptive_aae_bpm": _mean(target_adaptive),
        "posthoc_adaptive_acc_pct": hit_rate,
        "posthoc_adaptive_hit_rate_5bpm": hit_rate,
        "posthoc_final_aae_bpm": _mean(target_final),
        "posthoc_final_acc_pct": _accuracy(target_final),
        "posthoc_baseline_aae_bpm": _mean(target_baseline),
        "posthoc_baseline_acc_pct": _accuracy(target_baseline),
        "posthoc_n_valid_windows": int(np.isfinite(target_adaptive).sum()),
    }


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
        final_aae_bpm=float("nan"),
        baseline_acc_pct=float("nan"),
        adaptive_acc_pct=float("nan"),
        final_acc_pct=float("nan"),
        frame=pd.DataFrame(),
        segment_info=segment_info,
        alignment_info=alignment_info,
        motion_frequency=None,
        metric_arrays={},
    )
