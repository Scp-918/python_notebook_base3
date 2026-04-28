"""CSV, JSON, and PNG outputs for the batch adaptive protocol.

中文说明：
本模块集中管理协议输出文件，避免 Notebook 里散落保存逻辑。输出分三类：
1. QC/summary CSV；
2. 每个样本的最优参数 JSON 与长表 CSV；
3. 每个样本的运动段信号图、贝叶斯收敛图、自适应滤波结果图。
"""

from __future__ import annotations

import csv
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..params import CascadeScheme, TargetScope
from .batch_pairing import UnpairedSample
from .preprocess_protocol import ProtocolDataset
from .protocol_optimizer import ProtocolModeResult
from .qc import QcResult

__all__ = [
    "SampleOutputPaths",
    "write_batch_summary",
    "write_metric_matrix_tables",
    "write_qc_tables",
    "write_sample_outputs",
]

QC_TABLE_COLUMNS = [
    "group_id",
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
    """Paths written for one sample."""

    result_csv: Path
    report_json: Path
    motion_png: Path
    motion_recovery_png: Path
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

    # 中文注释：好采样/坏采样名单单独保存，Notebook 会直接 display 这两张表。
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
            {"metric": "good_samples", "value": len(good)},
            {"metric": "bad_samples", "value": len(bad)},
            {"metric": "unpaired_samples", "value": len(unpaired)},
        ]
    ).to_csv(paths["summary"], index=False, encoding="utf-8-sig")
    return paths


def write_sample_outputs(
    sample_stem: str,
    results: list[ProtocolModeResult],
    *,
    csv_out_dir: str | Path,
    report_out_dir: str | Path,
    fig_out_dir: str | Path | None = None,
    dataset: ProtocolDataset | None = None,
) -> SampleOutputPaths:
    """Write the long result CSV, JSON report, and all protocol PNG figures."""

    csv_dir = Path(csv_out_dir)
    report_dir = Path(report_out_dir)
    fig_dir = Path(fig_out_dir) if fig_out_dir is not None else report_dir
    csv_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 中文注释：长表 CSV 合并 14 个模式的逐窗 HR、误差和标签。
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

    # 中文注释：JSON 保存每个模式的 best params、AAE、accuracy、trial history 和重要性。
    report_json = report_dir / f"Best_Params_Result_{group_id}.json"
    payload = {r.mode_key: _mode_payload(r) for r in results}
    with report_json.open("w", encoding="utf-8") as f:
        json.dump(_jsonify(payload), f, ensure_ascii=False, indent=2)

    signal_png = fig_dir / "filtered_motion_signals" / f"filtered_motion_13ch_{group_id}.png"
    bayes_motion_png = fig_dir / "bayes_training_curves" / f"bayes_curve_motion_only_{group_id}.png"
    bayes_motion_recovery_png = (
        fig_dir / "bayes_training_curves" / f"bayes_curve_motion_recovery_{group_id}.png"
    )
    motion_png = fig_dir / "hr_compare" / f"hr_compare_motion_only_{group_id}.png"
    motion_recovery_png = fig_dir / "hr_compare" / f"hr_compare_motion_recovery_{group_id}.png"

    if dataset is not None:
        _safe_plot(_plot_motion_segment_signals, dataset, results, signal_png)
    _safe_plot(
        _plot_bayes_convergence,
        results,
        TargetScope.MOTION_ONLY,
        bayes_motion_png,
        title=f"{sample_stem} 运动段贝叶斯优化 AAE 收敛曲线",
    )
    _safe_plot(
        _plot_bayes_convergence,
        results,
        TargetScope.MOTION_AND_RECOVERY,
        bayes_motion_recovery_png,
        title=f"{sample_stem} 运动加恢复段贝叶斯优化 AAE 收敛曲线",
    )
    _safe_plot(
        _plot_scope,
        results,
        TargetScope.MOTION_ONLY,
        motion_png,
        title=f"{sample_stem} 运动段自适应滤波信号段选择",
    )
    _safe_plot(
        _plot_scope,
        results,
        TargetScope.MOTION_AND_RECOVERY,
        motion_recovery_png,
        title=f"{sample_stem} 运动加恢复段自适应滤波信号段选择",
    )
    return SampleOutputPaths(
        result_csv=result_csv,
        report_json=report_json,
        motion_png=motion_png,
        motion_recovery_png=motion_recovery_png,
        signal_png=signal_png,
        bayes_motion_png=bayes_motion_png,
        bayes_motion_recovery_png=bayes_motion_recovery_png,
    )


def write_batch_summary(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    """Write one row per sample summarising batch-level status."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample",
        "status",
        "reason",
        "result_csv",
        "report_json",
        "signal_png",
        "bayes_motion_png",
        "bayes_motion_recovery_png",
        "motion_png",
        "motion_recovery_png",
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
    """Write four cross-sample metric matrices for filtered segments only.

    输出 4 张 7 行 x n 列的表：行是级联滤波方案，列是好采样运动 group_id。
    AAE/accuracy 取每个样本、每个目标段、每个方案最终最优 repeat 的结果。
    """

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    specs = {
        "motion_only_aae": (
            TargetScope.MOTION_ONLY,
            "adaptive_aae_bpm",
            "filtered_motion_aae_bpm.csv",
        ),
        "motion_only_accuracy": (
            TargetScope.MOTION_ONLY,
            "adaptive_acc_pct",
            "filtered_motion_accuracy_pct.csv",
        ),
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


def _mode_payload(result: ProtocolModeResult) -> dict[str, Any]:
    run = result.best_run
    return {
        "target_scope": result.target_scope.value,
        "cascade_scheme": result.cascade_scheme.value,
        "runtime_config": {
            "DEBUG_MODE": bool(result.debug_mode),
            "n_trials": int(result.n_trials),
            "n_repeats": int(result.n_repeats),
            "best_repeat_idx": int(result.best_repeat_idx),
        },
        "best_params": result.best_params.to_dict(),
        "best_aae_bpm": result.best_aae_bpm,
        "baseline_aae_bpm": result.baseline_aae_bpm,
        "adaptive_aae_bpm": result.adaptive_aae_bpm,
        "baseline_acc_pct": result.baseline_acc_pct,
        "adaptive_acc_pct": result.adaptive_acc_pct,
        "param_importance": result.param_importance,
        "trial_history": result.trial_history,
        "lms_stage_summary": _lms_stage_summary(run.frame),
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


def _safe_plot(func: Any, *args: Any, **kwargs: Any) -> None:
    try:
        func(*args, **kwargs)
    except ModuleNotFoundError as exc:
        warnings.warn(f"skip plot because dependency is missing: {exc}", RuntimeWarning, stacklevel=2)
    except Exception as exc:
        warnings.warn(f"skip plot because plotting failed: {exc}", RuntimeWarning, stacklevel=2)


def _plot_motion_segment_signals(
    dataset: ProtocolDataset,
    results: list[ProtocolModeResult],
    out_path: Path,
) -> None:
    """Plot 13 band-pass-filtered channels inside the detected motion segment."""

    plt = _prepare_matplotlib(out_path)
    segment = _first_valid_segment(results)
    if segment is None:
        start_s = float(dataset.time_s[0])
        end_s = float(dataset.time_s[-1])
        title_suffix = "未检测到运动段，显示全段"
    else:
        start_s = float(segment.motion_start_s)
        end_s = float(segment.motion_end_s)
        title_suffix = f"运动段 {start_s:.1f}-{end_s:.1f}s"

    mask = (dataset.time_s >= start_s) & (dataset.time_s <= end_s)
    if mask.sum() < 2:
        mask = np.ones_like(dataset.time_s, dtype=bool)
    x = dataset.time_s[mask]
    x = x - x[0]

    groups = [
        ("两路 HF 热膜信号", "幅值/mV", [("HF1", dataset.hf1), ("HF2", dataset.hf2)]),
        ("两路 CF 冷膜信号", "幅值/ratio", [("CF1", dataset.cf1), ("CF2", dataset.cf2)]),
        (
            "三路 PPG 信号",
            "幅值/a.u.",
            [("Green", dataset.ppg_green), ("Red", dataset.ppg_red), ("IR", dataset.ppg_ir)],
        ),
        (
            "三轴加速度计信号",
            "幅值/g",
            [("AccX", dataset.accx), ("AccY", dataset.accy), ("AccZ", dataset.accz)],
        ),
        (
            "三轴陀螺仪信号",
            "幅值/dps",
            [("GyroX", dataset.gyrox), ("GyroY", dataset.gyroy), ("GyroZ", dataset.gyroz)],
        ),
    ]

    fig, axes = plt.subplots(5, 1, figsize=(13, 14), sharex=True)
    for ax, (subtitle, ylabel, series) in zip(axes, groups, strict=True):
        for label, values in series:
            ax.plot(x, np.asarray(values, dtype=float)[mask], linewidth=1.0, label=label)
        ax.set_title(subtitle)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", ncol=min(3, len(series)), fontsize=8)
    axes[-1].set_xlabel("运动段相对时间/s")
    fig.suptitle(f"{dataset.sample_stem} 带通滤波后 13 路运动段信号（{title_suffix}）", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_bayes_convergence(
    results: list[ProtocolModeResult],
    target_scope: TargetScope,
    out_path: Path,
    *,
    title: str,
) -> None:
    """Plot AAE versus Optuna trial index for seven cascade schemes."""

    plt = _prepare_matplotlib(out_path)
    by_scheme = {r.cascade_scheme: r for r in results if r.target_scope == target_scope}
    fig, axes = plt.subplots(7, 1, figsize=(12, 18), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])

    for ax, scheme in zip(axes, list(CascadeScheme), strict=True):
        result = by_scheme.get(scheme)
        ax.set_ylabel("AAE/bpm")
        ax.grid(True, alpha=0.25)
        if result is None or not result.trial_history:
            ax.set_title(f"{scheme.value} 无 trial 记录")
            ax.text(0.5, 0.5, "无 trial 记录", ha="center", va="center", transform=ax.transAxes)
            continue
        history = sorted(
            (
                item
                for item in result.trial_history
                if int(item.get("repeat_idx", 0)) == int(result.best_repeat_idx)
            ),
            key=lambda item: int(item.get("trial_idx", 0)),
        )
        if not history:
            history = sorted(result.trial_history, key=lambda item: int(item.get("trial_idx", 0)))
        x = np.asarray([int(item.get("trial_idx", 0)) + 1 for item in history], dtype=float)
        y = np.asarray([float(item.get("value", np.nan)) for item in history], dtype=float)
        y[~np.isfinite(y)] = np.nan
        best = np.minimum.accumulate(np.nan_to_num(y, nan=np.inf))
        best[~np.isfinite(best)] = np.nan
        ax.set_title(
            f"{scheme.value} best AAE={result.best_aae_bpm:.2f}, "
            f"best repeat={result.best_repeat_idx}"
        )
        ax.plot(x, y, color="0.55", marker="o", markersize=3, linewidth=1.0, label="当前 trial AAE")
        ax.plot(x, best, color="#1f77b4", linewidth=1.6, label="当前最优 AAE")
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("训练轮次/trial")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_scope(
    results: list[ProtocolModeResult],
    target_scope: TargetScope,
    out_path: Path,
    *,
    title: str,
) -> None:
    """Plot reference, raw-PPG baseline, and adaptive-filter HR curves."""

    plt = _prepare_matplotlib(out_path)
    by_scheme = {r.cascade_scheme: r for r in results if r.target_scope == target_scope}
    fig, axes = plt.subplots(7, 1, figsize=(12, 18), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])

    for ax, scheme in zip(axes, list(CascadeScheme), strict=True):
        result = by_scheme.get(scheme)
        ax.set_title(scheme.value)
        ax.set_ylabel("心率/bpm")
        ax.grid(True, alpha=0.25)
        if result is None or result.best_run.frame.empty:
            ax.text(0.5, 0.5, "无结果", ha="center", va="center", transform=ax.transAxes)
            continue
        frame = _scope_frame(result.best_run.frame, target_scope)
        if frame.empty:
            ax.text(0.5, 0.5, "目标段无结果", ha="center", va="center", transform=ax.transAxes)
            continue
        x = frame["time_s"].to_numpy(dtype=float)
        ax.plot(x, frame["ref_hr_bpm"], color="black", linewidth=1.6, label="真实 HR")
        ax.plot(
            x,
            frame["baseline_ppg_hr_bpm"],
            color="0.55",
            linestyle="--",
            linewidth=1.2,
            label="未去伪影 PPG baseline",
        )
        label = (
            f"自适应滤波 HR "
            f"AAE={result.adaptive_aae_bpm:.2f} "
            f"accuracy={result.adaptive_acc_pct:.1f}%"
        )
        ax.plot(x, frame["adaptive_hr_bpm"], color="#1f77b4", linewidth=1.4, label=label)
        ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("时间/s")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _first_valid_segment(results: list[ProtocolModeResult]) -> Any | None:
    for result in results:
        segment = result.best_run.segment_info
        if segment is not None and segment.is_valid:
            return segment
    return None


def _scope_frame(frame: pd.DataFrame, target_scope: TargetScope) -> pd.DataFrame:
    if "segment_label" not in frame.columns:
        return frame
    if target_scope == TargetScope.MOTION_ONLY:
        return frame.loc[frame["segment_label"].astype(str) == "motion"].copy()
    return frame.loc[frame["segment_label"].astype(str).isin(["motion", "recovery"])].copy()


def _lms_stage_summary(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame is None or frame.empty or "lms_stages_json" not in frame.columns:
        return []
    out: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for text in frame["lms_stages_json"].dropna().astype(str):
        try:
            stages = json.loads(text)
        except json.JSONDecodeError:
            continue
        for stage in stages:
            key = (
                stage.get("sensor_type"),
                stage.get("channel"),
                stage.get("D_opt_samples"),
                stage.get("M"),
                stage.get("K"),
                round(float(stage.get("mu", 0.0)), 10),
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
    mpl_cache = out_path.parent / ".matplotlib"
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
            "window_idx",
            "time_s",
            "segment_label",
            "ref_hr_bpm",
            "baseline_ppg_hr_bpm",
            "adaptive_hr_bpm",
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
