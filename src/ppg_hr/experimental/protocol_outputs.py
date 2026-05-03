"""CSV, JSON, and signal-figure outputs for the batch adaptive protocol.

中文说明：主训练流程只写 QC 表、训练曲线数据、汇总表和 13 路信号图；HR 对比图
只由 Notebook 末尾的“手动重画最佳参数”函数输出。
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..params import CascadeScheme, TargetScope
from .alignment import (
    align_ppg_to_ref_hr,
    compute_rest_alignment_diagnostic_curve,
    extract_rest_ppg_hr_tracked,
)
from .batch_pairing import SamplePair, UnpairedSample
from .preprocess_protocol import (
    PROTOCOL_CHANNELS,
    ProtocolDataset,
    load_protocol_raw_clean_frames,
    resample_protocol_dataset,
)
from .protocol_optimizer import ProtocolModeResult
from .qc import QcResult
from .segmentation import SegmentInfo

__all__ = [
    "SampleOutputPaths",
    "plot_raw_ppg_and_unaligned_hr_by_motion_type",
    "plot_unaligned_fullfield_ppg_hr_by_motion_type",
    "plot_rest_alignment_diagnostics_by_motion_type",
    "plot_signal_figures",
    "write_batch_summary",
    "write_metric_matrix_tables",
    "write_qc_tables",
    "write_sample_outputs",
]

QC_TABLE_COLUMNS = [
    "group_id",
    "motion_type",
    "data_file",
    "ref_file",
    "file_name",
    "status",
    "reason",
    "std_ut1",
    "std_ut2",
    "outlier_count_ut1",
    "outlier_count_ut2",
    "outlier_ratio_ut1",
    "outlier_ratio_ut2",
    "is_good",
]

UNPAIRED_TABLE_COLUMNS = ["file_name", "file_path", "reason"]


@dataclass(frozen=True)
class SampleOutputPaths:
    """Paths written for one sample.

    中文说明：保留旧字段以兼容旧 Notebook；新主流程不会自动生成 HR compare 图。
    """

    result_csv: Path
    report_json: Path
    motion_png: Path | None = None
    motion_recovery_png: Path | None = None
    signal_png: Path | None = None
    bayes_motion_png: Path | None = None
    bayes_motion_recovery_png: Path | None = None


def write_qc_tables(
    csv_out_dir: str | Path,
    *,
    good: list[QcResult],
    bad: list[QcResult],
    unpaired: list[UnpairedSample],
) -> dict[str, Path]:
    """Write good/bad/unpaired/QC summary tables."""

    out = Path(csv_out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "good": out / "good_samples.csv",
        "bad": out / "bad_samples.csv",
        "unpaired": out / "unpaired_samples.csv",
        "summary": out / "qc_summary.csv",
    }
    pd.DataFrame([r.to_dict() for r in good], columns=QC_TABLE_COLUMNS).to_csv(
        paths["good"], index=False, encoding="utf-8-sig"
    )
    pd.DataFrame([r.to_dict() for r in bad], columns=QC_TABLE_COLUMNS).to_csv(
        paths["bad"], index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(
        [{"file_name": u.file_name, "file_path": str(u.file_path), "reason": u.reason} for u in unpaired],
        columns=UNPAIRED_TABLE_COLUMNS,
    ).to_csv(paths["unpaired"], index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {"metric": "paired_samples", "value": len(good) + len(bad)},
            {"metric": "good_samples", "value": len(good)},
            {"metric": "bad_samples", "value": len(bad)},
            {"metric": "unpaired_samples", "value": len(unpaired)},
        ]
    ).to_csv(paths["summary"], index=False, encoding="utf-8-sig")
    return paths


def plot_rest_alignment_diagnostics_by_motion_type(
    pairs: list[SamplePair],
    datasets: dict[str, ProtocolDataset],
    output_dir: str | Path,
    fs_target: int = 100,
    TW: int = 8,
    alignment_TW: float | None = None,
    alignment_step_s: float = 1.0,
    rest_hr_kwargs: dict[str, Any] | None = None,
    alignment_score_mode: str = "aae",
) -> dict[str, Path]:
    """Plot rest-segment alignment diagnostics grouped by motion type.

    中文说明：每个运动类型输出一张 PNG；每个子图对应一组运动，显示全局 Tdelay
    选定后静息段的参考 HR 与 PPG Green Hamming+FFT HR。该诊断只用于人工检查
    Tdelay 合理性，不改变训练、Optuna objective 或 best_tdelay_s 搜索逻辑。
    """

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plt = _prepare_matplotlib(out)
    grouped: dict[str, list[SamplePair]] = {}
    for pair in pairs:
        if pair.motion_id in datasets:
            grouped.setdefault(pair.motion_type, []).append(pair)

    paths: dict[str, Path] = {}
    for motion_type, members in sorted(grouped.items()):
        members = sorted(members, key=lambda p: (p.motion_index, p.motion_id))
        n = max(1, len(members))
        cols = 2 if n > 1 else 1
        rows = int(math.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(6.4 * cols, 3.4 * rows), squeeze=False)
        flat = axes.ravel()
        for ax in flat[n:]:
            ax.axis("off")
        for ax, pair in zip(flat, members, strict=False):
            ax.grid(True, alpha=0.25)
            ax.set_xlabel("Rest window time (s)")
            ax.set_ylabel("HR (bpm)")
            title = pair.motion_id
            try:
                ds = resample_protocol_dataset(datasets[pair.motion_id], int(fs_target))
                segment = _detect_segments_for_alignment_plot(ds, TW)
                if not segment.is_valid:
                    raise RuntimeError(segment.reason)
                align_tw = float(TW) if alignment_TW is None else float(alignment_TW)
                aligned = align_ppg_to_ref_hr(
                    ds,
                    segment,
                    TW,
                    int(fs_target),
                    alignment_TW=align_tw,
                    alignment_step_s=float(alignment_step_s),
                    rest_hr_kwargs=rest_hr_kwargs,
                    alignment_score_mode=alignment_score_mode,
                )
                curve = compute_rest_alignment_diagnostic_curve(
                    ds,
                    segment,
                    aligned,
                    align_tw,
                    int(fs_target),
                    rest_hr_kwargs=rest_hr_kwargs,
                )
                info = aligned.alignment_info
                title = (
                    f"{pair.motion_id} | mode={info.alignment_score_mode}, "
                    f"Tdelay={info.best_tdelay_s:.2f}s, "
                    f"AAE={info.best_score_aae:.2f}, STD={info.best_score_std:.2f}"
                )
                if curve.empty:
                    ax.text(0.5, 0.5, "no comparable rest windows", ha="center", va="center", transform=ax.transAxes)
                else:
                    ax.plot(
                        curve["time_s"].to_numpy(dtype=float),
                        curve["ref_hr_bpm"].to_numpy(dtype=float),
                        color="black",
                        linewidth=1.5,
                        label="Reference HR",
                    )
                    ax.plot(
                        curve["time_s"].to_numpy(dtype=float),
                        curve["ppg_hr_bpm"].to_numpy(dtype=float),
                        color="#1f77b4",
                        linewidth=1.2,
                        label="PPG FFT HR",
                    )
                    ax.legend(loc="best", fontsize=8)
            except Exception as exc:
                title = f"{pair.motion_id} | Tdelay=nan"
                ax.text(0.5, 0.5, f"alignment failed\n{exc}", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)

        fig.suptitle(
            f"{motion_type} Rest alignment diagnostics, TW={int(TW)}, score_mode={alignment_score_mode}",
            fontsize=14,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        path = out / f"{motion_type}_rest_alignment_TW{int(TW)}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths[motion_type] = path
    return paths


def plot_unaligned_fullfield_ppg_hr_by_motion_type(
    pairs: list[SamplePair],
    datasets: dict[str, ProtocolDataset],
    output_dir: str | Path,
    fs_target: int = 100,
    TW: int | float = 8,
    step_s: float = 1.0,
    hr_band_hz: tuple[float, float] = ( (40.0 / 60.0, 180.0 / 60.0)),
    track_band_bpm: float = 30.0,
    slew_limit_bpm: float = 6.0,
    slew_step_bpm: float = 4.0,
    smooth_method: str = "median",
    smooth_win: int = 3,
    peak_percent: float = 0.3,
    spec_penalty_enable: bool = True,
    spec_penalty_weight: float = 0.2,
    spec_penalty_width_hz: float = 0.2,
) -> dict[str, Path]:
    """按运动类型绘制原始 PPG 全段未对齐 HR 测试图。

    中文说明：该图只用于测试和人工检查，不参与 Tdelay 搜索、Optuna objective 或
    自适应滤波训练。PPG_Green 不做全局对齐，直接在原始时间轴上按 ``TW`` 秒滑窗，
    每 1 秒一步，用 Hamming+FFT 提取 2/3-3 Hz 主频，再做上一 HR 邻域追踪、
    slew limit/step 防跳峰和 moving median 平滑。背景色按同一未对齐时间轴上的
    静息、运动、运动恢复三段标注。
    """

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plt = _prepare_matplotlib(out)
    grouped: dict[str, list[SamplePair]] = {}
    for pair in pairs:
        if pair.motion_id in datasets:
            grouped.setdefault(pair.motion_type, []).append(pair)

    paths: dict[str, Path] = {}
    tw_value = float(TW)
    hr_band_bpm = (float(hr_band_hz[0]) * 60.0, float(hr_band_hz[1]) * 60.0)
    for motion_type, members in sorted(grouped.items()):
        members = sorted(members, key=lambda p: (p.motion_index, p.motion_id))
        n = max(1, len(members))
        cols = 2 if n > 1 else 1
        rows = int(math.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(7.2 * cols, 3.8 * rows), squeeze=False)
        flat = axes.ravel()
        for ax in flat[n:]:
            ax.axis("off")
        for ax, pair in zip(flat, members, strict=False):
            ax.grid(True, alpha=0.25)
            ax.set_xlabel("Original PPG time (s)")
            ax.set_ylabel("HR (bpm)")
            try:
                ds = resample_protocol_dataset(datasets[pair.motion_id], int(fs_target))
                segment = _detect_segments_for_alignment_plot(ds, tw_value)
                _shade_unaligned_segments(ax, ds, segment)
                ppg_hr = extract_rest_ppg_hr_tracked(
                    ds.ppg_green,
                    ds.fs,
                    tw_s=tw_value,
                    step_s=float(step_s),
                    hr_band_bpm=hr_band_bpm,
                    track_band_bpm=float(track_band_bpm),
                    slew_limit_bpm=float(slew_limit_bpm),
                    slew_step_bpm=float(slew_step_bpm),
                    smooth_method=str(smooth_method),
                    smooth_win=int(smooth_win),
                    peak_percent=float(peak_percent),
                    penalty_signal=ds.accz,
                    spec_penalty_enable=bool(spec_penalty_enable),
                    spec_penalty_weight=float(spec_penalty_weight),
                    spec_penalty_width_hz=float(spec_penalty_width_hz),
                )
                ax.plot(
                    ds.ref_time_s,
                    ds.ref_hr_bpm,
                    color="black",
                    linewidth=1.4,
                    label="Reference HR",
                )
                if ppg_hr.times_s.size:
                    ax.plot(
                        ppg_hr.times_s,
                        ppg_hr.hr_bpm_smooth,
                        color="#1f77b4",
                        linewidth=1.2,
                        label="Raw PPG FFT HR",
                    )
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "PPG HR windows are empty",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                title = f"{pair.motion_id} | {_tw_label(tw_value)}"
                if not segment.is_valid:
                    title += " | segment failed"
                    ax.text(0.02, 0.95, segment.reason, ha="left", va="top", fontsize=8, transform=ax.transAxes)
                ax.set_title(title)
                ax.legend(loc="best", fontsize=8)
            except Exception as exc:
                ax.set_title(f"{pair.motion_id} | failed")
                ax.text(0.5, 0.5, str(exc), ha="center", va="center", transform=ax.transAxes)

        fig.suptitle(f"{motion_type} raw unaligned full-record PPG HR, {_tw_label(tw_value)}", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        path = out / f"{motion_type}_all_alignment_{_tw_label(tw_value)}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths[motion_type] = path
    return paths


def _shade_unaligned_segments(plt_ax: Any, dataset: ProtocolDataset, segment: SegmentInfo) -> None:
    """在未对齐 PPG 时间轴上标出静息、运动和运动恢复三段。

    中文说明：这里直接使用未做 Tdelay 平移的 ``dataset.time_s`` 和分段边界，保证
    背景区域与原始 PPG-HR 曲线处在同一个时间轴上。
    """

    if dataset.time_s.size == 0:
        return
    t0 = float(dataset.time_s[0])
    t1 = float(dataset.time_s[-1])
    if not segment.is_valid:
        plt_ax.axvspan(t0, t1, color="#eeeeee", alpha=0.22, label="segment unknown")
        return
    start = float(segment.motion_start_s)
    end = float(segment.motion_end_s)
    plt_ax.axvspan(t0, max(t0, start), color="#8fd19e", alpha=0.16, label="Rest")
    plt_ax.axvspan(max(t0, start), min(t1, end), color="#f6c177", alpha=0.18, label="Motion")
    plt_ax.axvspan(min(t1, end), t1, color="#9ecae1", alpha=0.16, label="Recovery")


def plot_raw_ppg_and_unaligned_hr_by_motion_type(
    pairs: list[SamplePair],
    datasets: dict[str, ProtocolDataset],
    output_dir: str | Path,
    fs_target: int = 100,
    fs_origin: int = 100,
    TW: int | float = 8,
    step_s: float = 1.0,
    hr_band_hz: tuple[float, float] = (40.0 / 60.0, 180.0 / 60.0),
    track_band_bpm: float = 30.0,
    slew_limit_bpm: float = 6.0,
    slew_step_bpm: float = 4.0,
    smooth_method: str = "median",
    smooth_win: int = 3,
    peak_percent: float = 0.3,
    spec_penalty_enable: bool = True,
    spec_penalty_weight: float = 0.2,
    spec_penalty_width_hz: float = 0.2,
) -> dict[str, Path]:
    """绘制静息段原始 PPG 绿光信号和静息段 PPG 解算 HR 的双 y 轴诊断图。

    中文说明：每个运动类型输出一张 PNG，每个子图对应一组运动。左轴绘制传感器
    CSV 中的静息段 PPG_Green 原始信号；右轴绘制未做 Tdelay 对齐的静息段 PPG
    解算 HR 曲线。该图用于判断静息 HR 偏差是否来自原始波形质量、静息段边界或
    频谱选峰流程。
    """

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plt = _prepare_matplotlib(out)
    grouped: dict[str, list[SamplePair]] = {}
    for pair in pairs:
        if pair.motion_id in datasets:
            grouped.setdefault(pair.motion_type, []).append(pair)

    paths: dict[str, Path] = {}
    tw_value = float(TW)
    hr_band_bpm = (float(hr_band_hz[0]) * 60.0, float(hr_band_hz[1]) * 60.0)
    for motion_type, members in sorted(grouped.items()):
        members = sorted(members, key=lambda p: (p.motion_index, p.motion_id))
        n = max(1, len(members))
        cols = 2 if n > 1 else 1
        rows = int(math.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(7.6 * cols, 4.0 * rows), squeeze=False)
        flat = axes.ravel()
        for ax in flat[n:]:
            ax.axis("off")
        for ax_left, pair in zip(flat, members, strict=False):
            ax_left.grid(True, alpha=0.22)
            ax_left.set_xlabel("Rest window time (s)")
            ax_left.set_ylabel("Rest PPG_Green raw", color="#2ca02c")
            ax_right = ax_left.twinx()
            ax_right.set_ylabel("Rest PPG FFT HR (bpm)", color="#d62728")
            try:
                ds = resample_protocol_dataset(datasets[pair.motion_id], int(fs_target))
                segment = _detect_segments_for_alignment_plot(ds, tw_value)
                if not segment.is_valid:
                    raise RuntimeError(segment.reason)
                raw_time, raw_ppg = _load_raw_ppg_green_for_pair(pair, ds, int(fs_origin))
                raw_time, raw_ppg = _slice_time_range(raw_time, raw_ppg, 0.0, float(segment.motion_start_s))
                rest_end_idx = int(round(float(segment.motion_start_s) * int(ds.fs)))
                rest_ppg = np.asarray(ds.ppg_green[: max(0, rest_end_idx)], dtype=float)
                rest_accz = np.asarray(ds.accz[: max(0, rest_end_idx)], dtype=float)
                ppg_hr = extract_rest_ppg_hr_tracked(
                    rest_ppg,
                    ds.fs,
                    tw_s=tw_value,
                    step_s=float(step_s),
                    hr_band_bpm=hr_band_bpm,
                    track_band_bpm=float(track_band_bpm),
                    slew_limit_bpm=float(slew_limit_bpm),
                    slew_step_bpm=float(slew_step_bpm),
                    smooth_method=str(smooth_method),
                    smooth_win=int(smooth_win),
                    peak_percent=float(peak_percent),
                    penalty_signal=rest_accz,
                    spec_penalty_enable=bool(spec_penalty_enable),
                    spec_penalty_weight=float(spec_penalty_weight),
                    spec_penalty_width_hz=float(spec_penalty_width_hz),
                )
                raw_line = ax_left.plot(
                    raw_time,
                    raw_ppg,
                    color="#2ca02c",
                    linewidth=0.7,
                    alpha=0.82,
                    label="Rest PPG_Green raw",
                )
                hr_lines = []
                if ppg_hr.times_s.size:
                    hr_lines = ax_right.plot(
                        ppg_hr.times_s,
                        ppg_hr.hr_bpm_smooth,
                        color="#d62728",
                        linewidth=1.25,
                        label="Rest PPG FFT HR",
                    )
                else:
                    ax_right.text(
                        0.5,
                        0.5,
                        "PPG HR windows are empty",
                        ha="center",
                        va="center",
                        transform=ax_right.transAxes,
                    )
                title = f"{pair.motion_id} | rest 0-{segment.motion_start_s:.1f}s | {_tw_label(tw_value)}"
                ax_left.set_title(title)
                ax_left.set_xlim(0.0, max(float(segment.motion_start_s), tw_value))
                _set_combined_legend(ax_left, raw_line + hr_lines)
            except Exception as exc:
                ax_left.set_title(f"{pair.motion_id} | failed")
                ax_left.text(0.5, 0.5, str(exc), ha="center", va="center", transform=ax_left.transAxes)

        fig.suptitle(f"{motion_type} rest raw PPG and rest PPG-HR, {_tw_label(tw_value)}", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        path = out / f"{motion_type}_rest_raw_ppg_hr_dual_axis_{_tw_label(tw_value)}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths[motion_type] = path
    return paths


def _tw_label(value: float) -> str:
    """返回文件名安全的 TW 标签，例如 TW8 或 TW8p5。"""

    if float(value).is_integer():
        return f"TW{int(value)}"
    return f"TW{str(float(value)).replace('.', 'p')}"


def _load_raw_ppg_green_for_pair(
    pair: SamplePair,
    fallback_dataset: ProtocolDataset,
    fs_origin: int,
) -> tuple[np.ndarray, np.ndarray]:
    """优先读取传感器 CSV 中的原始 PPG_Green；失败时回退到已预处理数据。"""

    try:
        raw_frame, _ = load_protocol_raw_clean_frames(pair.sensor_csv, fs_origin=fs_origin)
        return raw_frame["time_s"].to_numpy(dtype=float), raw_frame["ppg_green"].to_numpy(dtype=float)
    except Exception:
        return fallback_dataset.time_s.astype(float), fallback_dataset.ppg_green.astype(float)


def _slice_time_range(
    time_s: np.ndarray,
    values: np.ndarray,
    start_s: float,
    end_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """按时间范围截取信号，主要用于静息段原始 PPG 诊断图。"""

    t = np.asarray(time_s, dtype=float)
    v = np.asarray(values, dtype=float)
    n = min(t.size, v.size)
    if n == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    t = t[:n]
    v = v[:n]
    mask = (t >= float(start_s)) & (t <= float(end_s))
    return t[mask], v[mask]


def _set_combined_legend(plt_ax: Any, line_handles: list[Any]) -> None:
    """合并双 y 轴线条图例，并保留分段背景的说明。"""

    handles, labels = plt_ax.get_legend_handles_labels()
    for handle in line_handles:
        label = handle.get_label()
        if label not in labels:
            handles.append(handle)
            labels.append(label)
    dedup: dict[str, Any] = {}
    for handle, label in zip(handles, labels, strict=False):
        if label and not label.startswith("_"):
            dedup.setdefault(label, handle)
    if dedup:
        plt_ax.legend(dedup.values(), dedup.keys(), loc="best", fontsize=8)


def _detect_segments_for_alignment_plot(dataset: ProtocolDataset, TW: int | float) -> SegmentInfo:
    """Detect segments for the standalone alignment diagnostic plot."""

    from .segmentation import detect_activity_segments

    return detect_activity_segments(dataset.accx, dataset.accy, dataset.accz, dataset.fs, TW=TW)


def plot_signal_figures(
    *,
    sensor_csv: str | Path,
    dataset: ProtocolDataset,
    segment_info: SegmentInfo | None,
    output_dir: str | Path,
    group_id: str,
    motion_type: str,
    fs_origin: int = 100,
) -> dict[str, Path]:
    """Plot full raw/cleaned signals and motion-segment bandpassed signals.

    中文说明：两张图都使用 5 个子图：HF、CF、PPG、ACC、Gyro。每个子图都有
    title、xlabel、ylabel、legend 和 grid，文件名包含 group_id 与 motion_type。
    """

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plt = _prepare_matplotlib(out)
    raw_frame, clean_frame = load_protocol_raw_clean_frames(sensor_csv, fs_origin=fs_origin)
    full_path = out / f"raw_clean_13ch_{group_id}_{motion_type}.png"
    motion_path = out / f"motion_bandpass_13ch_{group_id}_{motion_type}.png"

    _plot_full_raw_clean(plt, raw_frame, clean_frame, full_path, group_id, motion_type)
    _plot_motion_bandpass(plt, dataset, segment_info, motion_path, group_id, motion_type)
    return {"raw_clean": full_path, "motion_bandpass": motion_path}


def write_sample_outputs(
    sample_stem: str,
    results: list[ProtocolModeResult],
    *,
    csv_out_dir: str | Path,
    report_out_dir: str | Path,
    fig_out_dir: str | Path | None = None,
    dataset: ProtocolDataset | None = None,
) -> SampleOutputPaths:
    """Compatibility writer for old per-sample workflows.

    中文说明：只保存长表和 JSON，不再从主流程输出 HR 对比图。``dataset`` 参数保留
    以兼容旧调用。
    """

    del fig_out_dir, dataset
    csv_dir = Path(csv_out_dir)
    report_dir = Path(report_out_dir)
    csv_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    group_id = _group_id(sample_stem)
    result_csv = csv_dir / f"adaptive_results_{group_id}.csv"
    frames = [
        r.best_run.frame
        for r in results
        if r.best_run.frame is not None and not r.best_run.frame.empty
    ]
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(result_csv, index=False, encoding="utf-8-sig")
    else:
        _empty_result_frame().to_csv(result_csv, index=False, encoding="utf-8-sig")

    report_json = report_dir / f"Best_Params_Result_{group_id}.json"
    payload = {r.mode_key: _mode_payload(r) for r in results}
    report_json.write_text(json.dumps(_jsonify(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return SampleOutputPaths(result_csv=result_csv, report_json=report_json)


def write_batch_summary(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    """Write one row per sample/motion-type summarising batch-level status."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "motion_type",
        "sample",
        "status",
        "reason",
        "split",
        "result_csv",
        "report_json",
    ]
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    return out


def write_metric_matrix_tables(
    out_dir: str | Path,
    results_by_sample: dict[str, list[ProtocolModeResult]],
) -> dict[str, Path]:
    """Compatibility matrix writer for old per-sample workflows."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    specs = {
        "motion_only_aae": (TargetScope.MOTION_ONLY, "adaptive_aae_bpm", "filtered_motion_aae_bpm.csv"),
        "motion_only_accuracy": (TargetScope.MOTION_ONLY, "adaptive_acc_pct", "filtered_motion_accuracy_pct.csv"),
        "motion_recovery_aae": (
            TargetScope.MOTION_AND_RECOVERY,
            "adaptive_aae_bpm",
            "filtered_motion_recovery_aae_bpm.csv",
        ),
        "motion_recovery_accuracy": (
            TargetScope.MOTION_AND_RECOVERY,
            "adaptive_acc_pct",
            "filtered_motion_recovery_accuracy_pct.csv",
        ),
    }
    sample_names = list(results_by_sample.keys())
    group_ids = [_group_id(name) for name in sample_names]
    paths: dict[str, Path] = {}
    for key, (scope, attr, filename) in specs.items():
        rows: list[dict[str, Any]] = []
        for scheme in CascadeScheme:
            row: dict[str, Any] = {"cascade_scheme": scheme.value}
            for sample_name, group_id in zip(sample_names, group_ids, strict=True):
                result = _find_mode_result(results_by_sample.get(sample_name, []), scope, scheme)
                row[group_id] = float(getattr(result, attr)) if result is not None else np.nan
            rows.append(row)
        path = out / filename
        pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
        paths[key] = path
    return paths


def _plot_full_raw_clean(
    plt: Any,
    raw_frame: pd.DataFrame,
    clean_frame: pd.DataFrame,
    out_path: Path,
    group_id: str,
    motion_type: str,
) -> None:
    groups = _plot_groups()
    fig, axes = plt.subplots(5, 1, figsize=(14, 15), sharex=True)
    x = clean_frame["time_s"].to_numpy(dtype=float)
    for ax, (title, ylabel, names) in zip(axes, groups, strict=True):
        for name in names:
            ax.plot(x, raw_frame[name].to_numpy(dtype=float), linewidth=0.7, linestyle="--", alpha=0.45, label=f"{name} raw")
            ax.plot(x, clean_frame[name].to_numpy(dtype=float), linewidth=0.9, label=f"{name} clean")
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right", ncol=min(3, len(names)), fontsize=7)
        ax.grid(True, alpha=0.25)
    fig.suptitle(f"{group_id} / {motion_type} full raw and cleaned 13-channel signals", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_motion_bandpass(
    plt: Any,
    dataset: ProtocolDataset,
    segment_info: SegmentInfo | None,
    out_path: Path,
    group_id: str,
    motion_type: str,
) -> None:
    frame = dataset.to_frame()
    if segment_info is not None and segment_info.is_valid:
        start_s = float(segment_info.motion_start_s)
        end_s = float(segment_info.motion_end_s)
    else:
        start_s = float(frame["time_s"].iloc[0])
        end_s = float(frame["time_s"].iloc[-1])
    mask = (frame["time_s"] >= start_s) & (frame["time_s"] <= end_s)
    if mask.sum() < 2:
        mask = np.ones(len(frame), dtype=bool)
    x = frame.loc[mask, "time_s"].to_numpy(dtype=float)
    if x.size:
        x = x - x[0]

    groups = _plot_groups()
    fig, axes = plt.subplots(5, 1, figsize=(14, 15), sharex=True)
    for ax, (title, ylabel, names) in zip(axes, groups, strict=True):
        for name in names:
            ax.plot(x, frame.loc[mask, name].to_numpy(dtype=float), linewidth=1.0, label=name)
        ax.set_title(title)
        ax.set_xlabel("Motion relative time (s)")
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right", ncol=min(3, len(names)), fontsize=8)
        ax.grid(True, alpha=0.25)
    fig.suptitle(f"{group_id} / {motion_type} motion-segment bandpassed 13-channel signals", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_groups() -> list[tuple[str, str, list[str]]]:
    return [
        ("HF channels", "Amplitude (mV)", ["hf1", "hf2"]),
        ("CF channels", "Ratio", ["cf1", "cf2"]),
        ("PPG channels", "Amplitude (a.u.)", ["ppg_green", "ppg_red", "ppg_ir"]),
        ("ACC channels", "Amplitude (g)", ["accx", "accy", "accz"]),
        ("Gyro channels", "Amplitude (dps)", ["gyrox", "gyroy", "gyroz"]),
    ]


def _mode_payload(result: ProtocolModeResult) -> dict[str, Any]:
    run = result.best_run
    return {
        "target_scope": result.target_scope.value,
        "cascade_scheme": result.cascade_scheme.value,
        "adaptive_filter": result.adaptive_filter,
        "runtime_config": {
            "n_trials": int(result.n_trials),
            "n_repeats": int(result.n_repeats),
            "best_repeat_idx": int(result.best_repeat_idx),
            "DEBUG_MODE": bool(result.debug_mode),
        },
        "best_params": result.best_params.to_dict(),
        "best_aae_bpm": result.best_aae_bpm,
        "baseline_aae_bpm": result.baseline_aae_bpm,
        "adaptive_aae_bpm": result.adaptive_aae_bpm,
        "baseline_acc_pct": result.baseline_acc_pct,
        "adaptive_acc_pct": result.adaptive_acc_pct,
        "param_importance": result.param_importance,
        "trial_history": result.trial_history,
        "adaptive_stage_summary": _adaptive_stage_summary(run.frame),
        "lms_stage_summary": _adaptive_stage_summary(run.frame),
        "segment_info": run.segment_info.to_dict() if run.segment_info is not None else None,
        "alignment_info": run.alignment_info.to_dict() if run.alignment_info is not None else None,
        "motion_frequency": run.motion_frequency,
        "success": run.success,
        "reason": run.reason,
    }


def _find_mode_result(
    results: list[ProtocolModeResult],
    scope: TargetScope,
    scheme: CascadeScheme,
) -> ProtocolModeResult | None:
    for result in results:
        if result.target_scope == scope and result.cascade_scheme == scheme:
            return result
    return None


def _adaptive_stage_summary(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    column = "adaptive_stages_json" if "adaptive_stages_json" in frame.columns else "lms_stages_json"
    if column not in frame.columns:
        return []
    out: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for text in frame[column].dropna().astype(str):
        try:
            stages = json.loads(text)
        except json.JSONDecodeError:
            continue
        for stage in stages:
            key = (
                stage.get("filter_type"),
                stage.get("sensor_type"),
                stage.get("channel"),
                stage.get("D_opt_samples"),
                stage.get("M"),
                stage.get("K"),
                stage.get("rff_seed"),
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(stage)
            if len(out) >= 30:
                return out
    return out


def _group_id(sample_stem: str) -> str:
    return str(sample_stem).removeprefix("multi_")


def _prepare_matplotlib(out_path: Path) -> Any:
    mpl_cache = out_path / ".matplotlib" if out_path.suffix == "" else out_path.parent / ".matplotlib"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    _configure_fonts(plt)
    return plt


def _configure_fonts(plt: Any) -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def _empty_result_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "sample",
            "group_id",
            "target_scope",
            "cascade_scheme",
            "adaptive_filter",
            "window_idx",
            "time_s",
            "segment_label",
            "ref_hr_bpm",
            "baseline_ppg_hr_bpm",
            "adaptive_hr_bpm",
            "adaptive_stages_json",
            "lms_stages_json",
            "is_filtered_segment",
            "baseline_abs_err_bpm",
            "adaptive_abs_err_bpm",
            "baseline_aae_bpm",
            "baseline_acc_pct",
            "adaptive_aae_bpm",
            "adaptive_acc_pct",
        ]
    )


def _jsonify(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_jsonify(v) for v in obj.tolist()]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, Path):
        return str(obj)
    return obj
