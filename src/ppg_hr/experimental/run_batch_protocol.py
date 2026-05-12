"""Top-level batch runner for grouped adaptive protocol experiments.

中文说明：本模块是 Notebook 和脚本调用的总调度器。它按顺序完成文件配对、QC、
预处理、信号图、按运动类型划分 train/val/test 或 all_train、逐模式 Optuna 优化、
贝叶斯训练曲线和跨运动类型汇总输出。
"""

from __future__ import annotations

import ast
import gc
import hashlib
import json
import math
import os
import shutil
import subprocess
from collections import OrderedDict
from dataclasses import dataclass, field, fields, replace
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

try:
    import optuna
    from optuna.samplers import TPESampler
except ModuleNotFoundError:  # pragma: no cover - only used in lean environments
    optuna = None
    TPESampler = None

from ..params import CascadeScheme, ProtocolParams, TargetScope
from .batch_pairing import PairDiscovery, SamplePair, discover_sample_pairs_with_unpaired
from .alignment import _window_fft_hr, search_time_bias_after
from .cascade_solver import (
    MetricArrays,
    ProtocolRunResult,
    _get_trial_base,
    aggregate_metric_arrays,
    clear_all_caches,
    clear_trial_caches,
    clear_trial_heavy_caches,
    run_protocol_trial,
)
from .preprocess_protocol import ProtocolDataset, load_and_preprocess_protocol
from .protocol_outputs import SampleOutputPaths, plot_signal_figures, write_qc_tables
from .protocol_search_space import (
    ProtocolSearchSpace,
    ProtocolTrialParams,
    decode_protocol_search_space,
    default_protocol_search_space,
)
from .qc import QcResult, quality_filter_sample
from .segmentation import detect_activity_segments

__all__ = [
    "BatchProtocolResult",
    "build_output_run_name",
    "redraw_best_param_hr_curves",
    "run_batch_adaptive_protocol",
    "safe_prepare_output_dir",
]

if optuna is not None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)

_VALID_FILTERS = ("lms", "volterra", "rff_lms")
_VALID_OBJECTIVES = ("aae", "accuracy")
_VALID_SPLIT_MODES = ("split", "all_train", "leave_one_group_out")
_DEFAULT_TRIAL_CACHE_MAX_ENTRIES = 128


@dataclass
class BatchProtocolResult:
    """Return payload from :func:`run_batch_adaptive_protocol`."""

    input_dir: Path
    output_root: Path
    csv_out_dir: Path
    report_out_dir: Path
    fig_out_dir: Path
    sample_outputs: dict[str, SampleOutputPaths] = field(default_factory=dict)
    mode_results: dict[str, list[Any]] = field(default_factory=dict)
    good_samples: list[QcResult] = field(default_factory=list)
    bad_samples: list[QcResult] = field(default_factory=list)
    pairs: list[SamplePair] = field(default_factory=list)
    unpaired_count: int = 0
    qc_tables: dict[str, Path] = field(default_factory=dict)
    batch_summary_csv: Path | None = None
    motion_type_dirs: dict[str, Path] = field(default_factory=dict)
    final_summary_tables: dict[str, Path] = field(default_factory=dict)
    signal_figures: dict[str, dict[str, Path]] = field(default_factory=dict)


@dataclass
class _ModeOptimisation:
    """Internal aggregate result for one motion_type/mode."""

    motion_type: str
    target_scope: TargetScope
    cascade_scheme: CascadeScheme
    adaptive_filter: str
    objective_mode: str
    data_split_mode: str
    best_params: ProtocolTrialParams
    best_repeat_idx: int
    best_trial_idx: int
    n_trials: int
    n_repeats: int
    train_metrics: dict[str, Any]
    val_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    per_group_rows: list[dict[str, Any]]
    history: list[dict[str, Any]]
    success: bool
    reason: str
    fold_id: int | None = None
    heldout_group_id: str = ""
    train_group_ids: list[str] = field(default_factory=list)
    test_group_id: str = ""
    fold_results: list["_ModeOptimisation"] = field(default_factory=list, repr=False)
    metric_arrays_by_split: dict[str, MetricArrays] = field(default_factory=dict, repr=False)
    result_level: str = "fold"
    aggregation: str = ""
    params_semantics: str = ""
    representative_fold_id: int | None = None
    representative_heldout_group_id: str = ""

    @property
    def mode_key(self) -> str:
        return f"{self.target_scope.value}__{self.cascade_scheme.value}__{self.adaptive_filter}"


def build_output_run_name(
    target_scopes: list[str | TargetScope],
    cascade_schemes: list[str | CascadeScheme],
    adaptive_filters: list[str],
    objective_mode: str,
    data_split_mode: str,
    tw_f_s: float = 0.0,
) -> str:
    """Build the run_name required by the Notebook output layout."""

    scopes = "-".join(TargetScope(x).value for x in target_scopes)
    schemes = "-".join(CascadeScheme(x).value for x in cascade_schemes)
    filters = "-".join(str(x) for x in adaptive_filters)
    return f"{scopes}__{schemes}__{filters}__{objective_mode}__{data_split_mode}__{_tw_f_run_label(tw_f_s)}"


def _tw_f_run_label(value: float) -> str:
    """Return a compact output-folder label for fixed TW_F seconds."""

    value_f = float(value)
    if value_f.is_integer():
        return f"TW_F{int(value_f)}s"
    text = f"{value_f:g}".replace(".", "p").replace("-", "m")
    return f"TW_F{text}s"


def safe_prepare_output_dir(
    *,
    project_root: str | Path,
    output_dir: str | Path,
    clean_outputs: bool = False,
) -> Path:
    """Create, and optionally clean, one isolated run output directory.

    中文说明：仅允许清空 ``PROJECT_ROOT / 'outputs'`` 下的本次运行目录，明确禁止
    删除 testdata、src、notebooks 或项目根目录。
    """

    root = Path(project_root).resolve()
    out = Path(output_dir).resolve()
    outputs_root = (root / "outputs").resolve()
    protected = {
        root,
        (root / "testdata").resolve(),
        (root / "src").resolve(),
        (root / "notebooks").resolve(),
    }
    if clean_outputs:
        if out in protected:
            raise ValueError(f"拒绝清空受保护目录: {out}")
        if outputs_root not in [out, *out.parents]:
            raise ValueError(f"拒绝清空 outputs 之外的目录: {out}")
        if out == outputs_root:
            raise ValueError("拒绝清空 outputs 根目录；只能清空本次 run_name 子目录")
        if out.exists():
            shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    return out


def run_batch_adaptive_protocol(
    *,
    input_dir: str | Path,
    output_root: str | Path | None = None,
    csv_out_dir: str | Path | None = None,
    report_out_dir: str | Path | None = None,
    fig_out_dir: str | Path | None = None,
    max_iterations: int = 350,
    num_repeats: int = 3,
    random_state: int = 42,
    num_seed_points: int = 10,
    fs_origin: int = 100,
    penalty_value: float = 999.0,
    parallel_repeats: int = 1,
    n_jobs: int | None = 1,
    trial_cache_max_entries: int = _DEFAULT_TRIAL_CACHE_MAX_ENTRIES,
    save_stage_json: bool = False,
    debug_mode: bool = False,
    search_space: ProtocolSearchSpace | None = None,
    trial_param_overrides: dict[str, Any] | None = None,
    target_scopes: list[TargetScope | str] | None = None,
    cascade_schemes: list[CascadeScheme | str] | None = None,
    adaptive_filters: list[str] | None = None,
    objective_mode: str = "aae",
    data_split_mode: str = "split",
    delay_estimation_mode: str = "envelope",
    val_groups_per_type: int = 1,
    test_groups_per_type: int = 1,
    cascade_train_budgets: dict[str, dict[str, int]] | None = None,
    project_root: str | Path | None = None,
    clean_outputs: bool = False,
    verbose: bool = True,
    on_log: Callable[[str], None] | None = None,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> BatchProtocolResult:
    """Run grouped protocol optimisation and output generation.

    中文说明：这里的 train/val/test 不是传统模型拟合；每个 trial 只评估一组超参数，
    自适应滤波器权重在窗口内在线更新，不跨文件保存。
    """

    objective_mode = str(objective_mode).lower()
    data_split_mode = str(data_split_mode).lower()
    delay_estimation_mode = str(delay_estimation_mode).lower()
    if objective_mode not in _VALID_OBJECTIVES:
        raise ValueError(f"objective_mode must be one of {_VALID_OBJECTIVES}")
    if data_split_mode not in _VALID_SPLIT_MODES:
        raise ValueError(f"data_split_mode must be one of {_VALID_SPLIT_MODES}")
    if delay_estimation_mode not in {"envelope", "direct"}:
        raise ValueError("delay_estimation_mode must be 'envelope' or 'direct'")

    scopes = [TargetScope(x) for x in (target_scopes or [TargetScope.MOTION_ONLY, TargetScope.MOTION_AND_RECOVERY])]
    schemes = [CascadeScheme(x) for x in (cascade_schemes or list(CascadeScheme))]
    filters = [str(x) for x in (adaptive_filters or ["lms"])]
    unsupported = [x for x in filters if x not in _VALID_FILTERS]
    if unsupported:
        raise ValueError(f"Unsupported adaptive_filters: {unsupported}")

    trial_overrides = _normalise_trial_param_overrides(trial_param_overrides)
    input_path = Path(input_dir).resolve()
    if output_root is None:
        if csv_out_dir is not None:
            root_out = Path(csv_out_dir).resolve().parent
        else:
            run_name = build_output_run_name(
                scopes,
                schemes,
                filters,
                objective_mode,
                data_split_mode,
                tw_f_s=float(trial_overrides.get("TW_F", 0.0)),
            )
            root_out = input_path.parent / "outputs" / run_name
    else:
        root_out = Path(output_root).resolve()
    inferred_project = Path(project_root).resolve() if project_root is not None else root_out.parent.parent
    root_out = safe_prepare_output_dir(
        project_root=inferred_project,
        output_dir=root_out,
        clean_outputs=clean_outputs,
    )
    csv_dir = Path(csv_out_dir).resolve() if csv_out_dir is not None else root_out / "qc"
    report_dir = Path(report_out_dir).resolve() if report_out_dir is not None else root_out / "reports"
    fig_dir = Path(fig_out_dir).resolve() if fig_out_dir is not None else root_out
    motion_root = root_out / "motion_types"
    signal_dir = root_out / "signal_figures"
    final_dir = root_out / "final_summary"
    for directory in (csv_dir, report_dir, fig_dir, motion_root, signal_dir, final_dir):
        directory.mkdir(parents=True, exist_ok=True)

    cfg = ProtocolParams(
        fs_origin=fs_origin,
        max_iterations=int(max_iterations),
        num_repeats=int(num_repeats),
        random_state=int(random_state),
        num_seed_points=int(num_seed_points),
        penalty_value=float(penalty_value),
        parallel_repeats=int(n_jobs if n_jobs is not None else parallel_repeats),
        debug_mode=bool(debug_mode),
    )
    save_stage_json = bool(save_stage_json or debug_mode)
    trial_cache_max_entries = max(1, int(trial_cache_max_entries))
    space = search_space or default_protocol_search_space()

    def _log(message: str) -> None:
        if on_log is not None:
            on_log(message)
        elif verbose:
            print(message)

    def _progress(info: dict[str, Any]) -> None:
        if on_progress is not None:
            on_progress(dict(info))
        if progress_callback is not None:
            progress_callback(dict(info))

    discovery = discover_sample_pairs_with_unpaired(input_path)
    _log(f"配对总数: {len(discovery.pairs)}")
    _log(f"未配对文件数量: {len(discovery.unpaired)}")

    good_qc: list[QcResult] = []
    bad_qc: list[QcResult] = []
    good_pairs: list[SamplePair] = []
    qc_by_stem: dict[str, QcResult] = {}
    for idx, pair in enumerate(discovery.pairs, start=1):
        _progress({"stage": "qc", "current": idx, "total": len(discovery.pairs), "sample": pair.stem})
        qc = quality_filter_sample(
            pair.sensor_csv,
            fs=fs_origin,
            group_id=pair.motion_id,
            motion_type=pair.motion_type,
            ref_csv=pair.ref_csv,
        )
        qc_by_stem[pair.stem] = qc
        if qc.is_good:
            good_qc.append(qc)
            good_pairs.append(pair)
        else:
            bad_qc.append(qc)
    _log(f"好样本数量: {len(good_qc)}")
    _log(f"坏样本数量: {len(bad_qc)}")
    qc_tables = write_qc_tables(csv_dir, good=good_qc, bad=bad_qc, unpaired=discovery.unpaired)
    motion_type_table = csv_dir / "motion_type_samples.csv"
    pd.DataFrame(
        [
            {
                "group_id": pair.motion_id,
                "motion_type": pair.motion_type,
                "motion_index": pair.motion_index,
                "data_file": pair.sensor_csv.name,
                "ref_file": pair.ref_csv.name,
            }
            for pair in discovery.pairs
        ]
    ).to_csv(motion_type_table, index=False, encoding="utf-8-sig")
    qc_tables["motion_type_samples"] = motion_type_table

    datasets: dict[str, ProtocolDataset] = {}
    pair_by_group: dict[str, SamplePair] = {}
    load_failures: list[dict[str, Any]] = []
    signal_figures: dict[str, dict[str, Path]] = {}
    for idx, pair in enumerate(good_pairs, start=1):
        _progress({"stage": "preprocess", "current": idx, "total": len(good_pairs), "sample": pair.stem})
        try:
            dataset = load_and_preprocess_protocol(pair.sensor_csv, pair.ref_csv, fs_origin=fs_origin)
            datasets[pair.motion_id] = dataset
            pair_by_group[pair.motion_id] = pair
            seg_for_plot = detect_activity_segments(dataset.accx, dataset.accy, dataset.accz, dataset.fs, TW=8)
            signal_figures[pair.motion_id] = plot_signal_figures(
                sensor_csv=pair.sensor_csv,
                dataset=dataset,
                segment_info=seg_for_plot,
                output_dir=signal_dir,
                group_id=pair.motion_id,
                motion_type=pair.motion_type,
                fs_origin=fs_origin,
            )
        except Exception as exc:
            load_failures.append(
                {
                    "motion_type": pair.motion_type,
                    "sample": pair.stem,
                    "status": "failed_preprocess",
                    "reason": str(exc),
                }
            )

    grouped_pairs: dict[str, list[SamplePair]] = {}
    for pair in good_pairs:
        if pair.motion_id in datasets:
            grouped_pairs.setdefault(pair.motion_type, []).append(pair)
    motion_types = sorted(grouped_pairs)
    _log(f"当前识别到的运动类型列表: {motion_types}")
    for motion_type in motion_types:
        members = grouped_pairs[motion_type]
        detail = ", ".join(f"{p.motion_id}/{p.sensor_csv.name}" for p in members)
        _log(f"{motion_type}: {detail}")

    split_plans = _build_split_plan(
        grouped_pairs,
        data_split_mode=data_split_mode,
        val_groups_per_type=val_groups_per_type,
        test_groups_per_type=test_groups_per_type,
        random_state=random_state,
    )

    batch_rows: list[dict[str, Any]] = []
    batch_rows.extend(load_failures)
    all_mode_results: dict[str, list[_ModeOptimisation]] = {}
    bayes_tables: dict[str, Path] = {}
    motion_type_dirs: dict[str, Path] = {}
    total_modes = max(1, len(motion_types) * len(scopes) * len(schemes) * len(filters))
    mode_counter = 0

    for motion_type in motion_types:
        motion_dir = motion_root / motion_type
        motion_dir.mkdir(parents=True, exist_ok=True)
        motion_type_dirs[motion_type] = motion_dir
        folds = split_plans[motion_type]
        valid_folds = [fold for fold in folds if fold.get("status") == "ok"]
        split_rows = _split_rows(folds, pair_by_group)
        split_path = motion_dir / "split_files.csv"
        pd.DataFrame(split_rows).to_csv(split_path, index=False, encoding="utf-8-sig")
        if data_split_mode == "leave_one_group_out":
            pd.DataFrame(split_rows).to_csv(motion_dir / "fold_split_files.csv", index=False, encoding="utf-8-sig")

        all_ids = _unique_ids_from_folds(folds)

        mode_results: list[_ModeOptimisation] = []
        shared_trial_cache: OrderedDict[tuple[Any, ...], ProtocolRunResult] = OrderedDict()
        for scope in scopes:
            for scheme in schemes:
                budget = _budget_for_scheme(cascade_train_budgets, scheme, cfg)
                for adaptive_filter in filters:
                    mode_counter += 1
                    _progress(
                        {
                            "stage": "optimization_mode",
                            "mode_idx": mode_counter,
                            "mode_current": mode_counter,
                            "mode_total": total_modes,
                            "motion_type": motion_type,
                            "target_scope": scope.name,
                            "target_scope_value": scope.value,
                            "cascade_scheme": scheme.value,
                            "adaptive_filter": adaptive_filter,
                        }
                    )
                    if not valid_folds:
                        reason = str(folds[0].get("reason", "no valid folds")) if folds else "no valid folds"
                        result = _failed_mode_optimisation(
                            motion_type=motion_type,
                            scope=scope,
                            scheme=scheme,
                            adaptive_filter=adaptive_filter,
                            objective_mode=objective_mode,
                            data_split_mode=data_split_mode,
                            delay_estimation_mode=delay_estimation_mode,
                            space=space,
                            n_trials=budget["n_trials"],
                            n_repeats=budget["n_repeats"],
                            trial_param_overrides=trial_overrides,
                            reason=reason,
                        )
                    elif data_split_mode == "leave_one_group_out":
                        fold_results: list[_ModeOptimisation] = []
                        for fold in valid_folds:
                            fold_trial_cache: OrderedDict[tuple[Any, ...], ProtocolRunResult] = OrderedDict()
                            train_ids = list(fold["train"])
                            test_ids = list(fold["test"])
                            result_fold = _optimise_group_mode(
                                motion_type=motion_type,
                                train_sets={gid: datasets[gid] for gid in train_ids},
                                val_sets={},
                                test_sets={gid: datasets[gid] for gid in test_ids},
                                scope=scope,
                                scheme=scheme,
                                adaptive_filter=adaptive_filter,
                                objective_mode=objective_mode,
                                data_split_mode=data_split_mode,
                                delay_estimation_mode=delay_estimation_mode,
                                cfg=cfg,
                                space=space,
                                trial_param_overrides=trial_overrides,
                                n_trials=budget["n_trials"],
                                n_repeats=budget["n_repeats"],
                                trial_cache=fold_trial_cache,
                                penalty_value=float(penalty_value),
                                mode_idx=mode_counter,
                                mode_total=total_modes,
                                random_state=int(random_state),
                                n_jobs=int(cfg.parallel_repeats),
                                trial_cache_max_entries=trial_cache_max_entries,
                                save_stage_json=save_stage_json,
                                on_progress=_progress,
                                fold_id=int(fold["fold_id"]),
                                heldout_group_id=str(fold["heldout_group_id"]),
                                train_group_ids=train_ids,
                                test_group_id=test_ids[0] if test_ids else "",
                            )
                            fold_results.append(result_fold)
                            fold_trial_cache.clear()
                            for gid in train_ids + test_ids:
                                clear_trial_heavy_caches(datasets[gid])
                            del fold_trial_cache
                            gc.collect()
                        result = _aggregate_logo_fold_results(
                            motion_type=motion_type,
                            scope=scope,
                            scheme=scheme,
                            adaptive_filter=adaptive_filter,
                            objective_mode=objective_mode,
                            data_split_mode=data_split_mode,
                            fold_results=fold_results,
                            n_trials=budget["n_trials"],
                            n_repeats=budget["n_repeats"],
                        )
                    else:
                        fold = valid_folds[0]
                        train_ids = list(fold["train"])
                        val_ids = list(fold["val"])
                        test_ids = list(fold["test"])
                        result = _optimise_group_mode(
                            motion_type=motion_type,
                            train_sets={gid: datasets[gid] for gid in train_ids},
                            val_sets={gid: datasets[gid] for gid in val_ids},
                            test_sets={gid: datasets[gid] for gid in test_ids},
                            scope=scope,
                            scheme=scheme,
                            adaptive_filter=adaptive_filter,
                            objective_mode=objective_mode,
                            data_split_mode=data_split_mode,
                            delay_estimation_mode=delay_estimation_mode,
                            cfg=cfg,
                            space=space,
                            trial_param_overrides=trial_overrides,
                            n_trials=budget["n_trials"],
                            n_repeats=budget["n_repeats"],
                            trial_cache=shared_trial_cache,
                            penalty_value=float(penalty_value),
                            mode_idx=mode_counter,
                            mode_total=total_modes,
                            random_state=int(random_state),
                            n_jobs=int(cfg.parallel_repeats),
                            trial_cache_max_entries=trial_cache_max_entries,
                            save_stage_json=save_stage_json,
                            on_progress=_progress,
                            fold_id=0,
                            heldout_group_id="",
                            train_group_ids=train_ids,
                            test_group_id=test_ids[0] if len(test_ids) == 1 else "",
                        )
                    mode_results.append(result)
                    gc.collect()
        all_mode_results[motion_type] = mode_results
        _write_motion_type_outputs(motion_dir, motion_type, mode_results)
        bayes_path = _plot_bayes_curves(motion_dir, motion_type, mode_results, objective_mode)
        bayes_tables[motion_type] = bayes_path
        shared_trial_cache.clear()
        for gid in all_ids:
            clear_all_caches(datasets[gid])
        batch_rows.append(
            {
                "motion_type": motion_type,
                "sample": ",".join(all_ids),
                "status": "ok" if valid_folds else "failed",
                "reason": "" if valid_folds else str(folds[0].get("reason", "no valid folds")),
                "split": str(split_path),
                "result_csv": str(motion_dir / "mode_summary_aae.csv"),
                "report_json": str(motion_dir / "best_params_all.json"),
            }
        )
        gc.collect()

    batch_summary_csv = _write_batch_summary(root_out / "batch_summary.csv", batch_rows)
    final_tables = _write_final_summary(final_dir, all_mode_results, scopes, objective_mode, data_split_mode)

    return BatchProtocolResult(
        input_dir=input_path,
        output_root=root_out,
        csv_out_dir=csv_dir,
        report_out_dir=report_dir,
        fig_out_dir=fig_dir,
        good_samples=good_qc,
        bad_samples=bad_qc,
        pairs=discovery.pairs,
        unpaired_count=len(discovery.unpaired),
        qc_tables=qc_tables,
        batch_summary_csv=batch_summary_csv,
        motion_type_dirs=motion_type_dirs,
        final_summary_tables=final_tables,
        signal_figures=signal_figures,
        mode_results=all_mode_results,
    )


def _build_split_plan(
    grouped_pairs: dict[str, list[SamplePair]],
    *,
    data_split_mode: str,
    val_groups_per_type: int,
    test_groups_per_type: int,
    random_state: int,
) -> dict[str, list[dict[str, Any]]]:
    """Build one or many folds per motion type.

    中文说明：``split`` 和 ``all_train`` 仍然只返回一个 fold；LOGO 模式为每个
    held-out group 返回一个独立 fold。如果某个 motion_type 少于 2 个好样本，
    返回失败 fold，主流程会写 reason 并继续处理其他运动类型。
    """

    plans: dict[str, list[dict[str, Any]]] = {}
    for motion_type, pairs in grouped_pairs.items():
        pairs_sorted = sorted(pairs, key=lambda p: (p.motion_index, p.motion_id))
        ids = [p.motion_id for p in pairs_sorted]
        if data_split_mode == "all_train":
            plans[motion_type] = [
                {
                    "fold_id": 0,
                    "heldout_group_id": "",
                    "train": ids,
                    "val": [],
                    "test": ids,
                    "status": "ok",
                    "reason": "",
                }
            ]
            continue
        if data_split_mode == "leave_one_group_out":
            if len(ids) < 2:
                plans[motion_type] = [
                    {
                        "fold_id": 0,
                        "heldout_group_id": "",
                        "train": [],
                        "val": [],
                        "test": [],
                        "status": "failed",
                        "reason": f"运动类型 {motion_type} 好样本数量少于 2，无法执行 leave-one-group-out。",
                    }
                ]
                continue
            plans[motion_type] = [
                {
                    "fold_id": fold_id,
                    "heldout_group_id": heldout,
                    "train": [gid for gid in ids if gid != heldout],
                    "val": [],
                    "test": [heldout],
                    "status": "ok",
                    "reason": "",
                }
                for fold_id, heldout in enumerate(ids)
            ]
            continue
        m = int(val_groups_per_type)
        k = int(test_groups_per_type)
        if len(ids) - m - k < 1:
            raise ValueError(
                f"运动类型 {motion_type} 好样本数量不足，无法按 m={m}, k={k} 划分 train/val/test。"
            )
        rng = np.random.default_rng(_stable_int_hash({"motion_type": motion_type, "random_state": random_state}))
        shuffled = list(ids)
        rng.shuffle(shuffled)
        test = shuffled[:k]
        val = shuffled[k : k + m]
        train = shuffled[k + m :]
        plans[motion_type] = [
            {
                "fold_id": 0,
                "heldout_group_id": "",
                "train": train,
                "val": val,
                "test": test,
                "status": "ok",
                "reason": "",
            }
        ]
    return plans


def _split_rows(folds: list[dict[str, Any]], pair_by_group: dict[str, SamplePair]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold in folds:
        if fold.get("status") != "ok":
            rows.append(
                {
                    "fold_id": int(fold.get("fold_id", 0)),
                    "heldout_group_id": str(fold.get("heldout_group_id", "")),
                    "split": "failed",
                    "group_id": "",
                    "motion_type": "",
                    "data_file": "",
                    "ref_file": "",
                    "reason": str(fold.get("reason", "")),
                }
            )
            continue
        for split_name in ("train", "val", "test"):
            for gid in fold.get(split_name, []):
                pair = pair_by_group.get(gid)
                rows.append(
                    {
                        "fold_id": int(fold.get("fold_id", 0)),
                        "heldout_group_id": str(fold.get("heldout_group_id", "")),
                        "split": split_name,
                        "group_id": gid,
                        "motion_type": pair.motion_type if pair is not None else "",
                        "data_file": str(pair.sensor_csv) if pair is not None else "",
                        "ref_file": str(pair.ref_csv) if pair is not None else "",
                        "reason": "",
                    }
                )
    return rows


def _unique_ids_from_folds(folds: list[dict[str, Any]]) -> list[str]:
    """Return stable unique group ids appearing anywhere in a split plan.

    中文说明：LOGO 中同一 group 会在多个 fold 的 train/test 中重复出现；这里按
    首次出现顺序去重，只用于 batch_summary 的样本列表展示。
    """

    seen: set[str] = set()
    out: list[str] = []
    for fold in folds:
        for split_name in ("train", "val", "test"):
            for gid in fold.get(split_name, []):
                if gid not in seen:
                    seen.add(gid)
                    out.append(gid)
    return out


def _budget_for_scheme(
    budgets: dict[str, dict[str, int]] | None,
    scheme: CascadeScheme,
    cfg: ProtocolParams,
) -> dict[str, int]:
    """Return per-cascade n_trials/n_repeats with config fallback."""

    item = (budgets or {}).get(scheme.value, {})
    return {
        "n_trials": int(item.get("n_trials", cfg.max_iterations)),
        "n_repeats": int(item.get("n_repeats", cfg.num_repeats)),
    }


def _run_one_optuna_repeat(
    *,
    repeat_idx: int,
    motion_type: str,
    objective_sets: dict[str, ProtocolDataset],
    objective_split: str,
    scope: TargetScope,
    scheme: CascadeScheme,
    adaptive_filter: str,
    objective_mode: str,
    delay_estimation_mode: str,
    space: ProtocolSearchSpace,
    trial_param_overrides: dict[str, Any] | None,
    n_trials: int,
    num_seed_points: int,
    penalty_value: float,
    random_state: int,
    fold_id: int | None,
    heldout_group_id: str,
    trial_cache_max_entries: int,
) -> dict[str, Any]:
    """Run one independent Optuna/random repeat for repeat-level parallelism."""

    trial_cache: OrderedDict[tuple[Any, ...], ProtocolRunResult] = OrderedDict()
    best_value = float("inf")
    best_params = _default_params_for_filter(
        space,
        adaptive_filter,
        objective_mode,
        delay_estimation_mode=delay_estimation_mode,
        trial_param_overrides=trial_param_overrides,
    )
    best_trial_idx = 0
    history: list[dict[str, Any]] = []
    best_so_far = float("inf")

    def _evaluate(params: ProtocolTrialParams) -> dict[str, Any]:
        metrics, _, _ = _evaluate_dataset_map(
            objective_sets,
            scope,
            scheme,
            params,
            objective_split,
            trial_cache,
            eval_mode="light",
            save_stage_json=False,
            trial_cache_max_entries=trial_cache_max_entries,
            fold_id=fold_id,
            heldout_group_id=heldout_group_id,
        )
        return metrics

    try:
        if optuna is not None and TPESampler is not None:
            sampler = TPESampler(
                seed=int(random_state) + int(repeat_idx),
                n_startup_trials=min(int(num_seed_points), int(n_trials)),
            )
            study = optuna.create_study(direction="minimize", sampler=sampler)

            def _objective(trial: optuna.trial.Trial) -> float:
                nonlocal best_value, best_params, best_trial_idx, best_so_far

                idx_map = {
                    name: trial.suggest_int(name, 0, len(space.options(name)) - 1)
                    for name in space.names_for_filter(adaptive_filter)
                }
                params = _decode_with_seed(
                    space,
                    idx_map,
                    adaptive_filter=adaptive_filter,
                    objective_mode=objective_mode,
                    delay_estimation_mode=delay_estimation_mode,
                    mode_key=f"{scope.value}__{scheme.value}__{adaptive_filter}",
                    repeat_idx=repeat_idx,
                    random_state=random_state,
                    trial_param_overrides=trial_param_overrides,
                )
                metrics = _evaluate(params)
                value = _objective_value(metrics, objective_mode, penalty_value)
                best_so_far = min(best_so_far, value)
                if value < best_value:
                    best_value = value
                    best_params = params
                    best_trial_idx = int(trial.number)
                _append_history(
                    history,
                    motion_type,
                    scope,
                    scheme,
                    adaptive_filter,
                    params,
                    repeat_idx,
                    int(trial.number),
                    value,
                    metrics,
                    best_so_far,
                    fold_id=fold_id,
                    heldout_group_id=heldout_group_id,
                )
                return value

            study.optimize(_objective, n_trials=int(n_trials), show_progress_bar=False)
        else:
            rng = np.random.default_rng(int(random_state) + int(repeat_idx))
            for trial_idx in range(int(n_trials)):
                idx_map = {
                    name: int(rng.integers(0, len(space.options(name))))
                    for name in space.names_for_filter(adaptive_filter)
                }
                params = _decode_with_seed(
                    space,
                    idx_map,
                    adaptive_filter=adaptive_filter,
                    objective_mode=objective_mode,
                    delay_estimation_mode=delay_estimation_mode,
                    mode_key=f"{scope.value}__{scheme.value}__{adaptive_filter}",
                    repeat_idx=repeat_idx,
                    random_state=random_state,
                    trial_param_overrides=trial_param_overrides,
                )
                metrics = _evaluate(params)
                value = _objective_value(metrics, objective_mode, penalty_value)
                best_so_far = min(best_so_far, value)
                if value < best_value:
                    best_value = value
                    best_params = params
                    best_trial_idx = int(trial_idx)
                _append_history(
                    history,
                    motion_type,
                    scope,
                    scheme,
                    adaptive_filter,
                    params,
                    repeat_idx,
                    int(trial_idx),
                    value,
                    metrics,
                    best_so_far,
                    fold_id=fold_id,
                    heldout_group_id=heldout_group_id,
                )
    finally:
        trial_cache.clear()
        for dataset in objective_sets.values():
            clear_trial_heavy_caches(dataset)

    return {
        "repeat_idx": int(repeat_idx),
        "best_value": float(best_value),
        "best_params": best_params,
        "best_trial_idx": int(best_trial_idx),
        "history": history,
    }


def _recompute_history_best_so_far(history: list[dict[str, Any]]) -> None:
    best = float("inf")
    for row in history:
        value = float(row.get("objective_value", float("inf")))
        best = min(best, value)
        row["best_so_far"] = float(best)


def _optimise_group_mode(
    *,
    motion_type: str,
    train_sets: dict[str, ProtocolDataset],
    val_sets: dict[str, ProtocolDataset],
    test_sets: dict[str, ProtocolDataset],
    scope: TargetScope,
    scheme: CascadeScheme,
    adaptive_filter: str,
    objective_mode: str,
    data_split_mode: str,
    delay_estimation_mode: str,
    cfg: ProtocolParams,
    space: ProtocolSearchSpace,
    trial_param_overrides: dict[str, Any] | None,
    n_trials: int,
    n_repeats: int,
    trial_cache: OrderedDict[tuple[Any, ...], ProtocolRunResult] | dict[tuple[Any, ...], ProtocolRunResult],
    penalty_value: float,
    mode_idx: int,
    mode_total: int,
    random_state: int,
    on_progress: Callable[[dict[str, Any]], None],
    n_jobs: int = 1,
    trial_cache_max_entries: int = _DEFAULT_TRIAL_CACHE_MAX_ENTRIES,
    save_stage_json: bool = False,
    fold_id: int | None = None,
    heldout_group_id: str = "",
    train_group_ids: list[str] | None = None,
    test_group_id: str = "",
) -> _ModeOptimisation:
    """Optimise one motion_type/mode over train/val/test datasets."""

    best_value = float("inf")
    best_params = _default_params_for_filter(
        space,
        adaptive_filter,
        objective_mode,
        delay_estimation_mode=delay_estimation_mode,
        trial_param_overrides=trial_param_overrides,
    )
    best_repeat_idx = 0
    best_trial_idx = 0
    history: list[dict[str, Any]] = []
    best_so_far = float("inf")
    objective_sets = val_sets if data_split_mode == "split" else train_sets
    objective_split = "val" if data_split_mode == "split" else "train"
    actual_n_jobs = min(max(1, int(n_jobs or 1)), int(n_repeats))

    def _evaluate_objective_params(params: ProtocolTrialParams) -> dict[str, Any]:
        metrics, _, _ = _evaluate_dataset_map(
            objective_sets,
            scope,
            scheme,
            params,
            objective_split,
            trial_cache,
            eval_mode="light",
            save_stage_json=False,
            trial_cache_max_entries=trial_cache_max_entries,
            fold_id=fold_id,
            heldout_group_id=heldout_group_id,
        )
        return metrics

    def _evaluate_params_full(
        params: ProtocolTrialParams,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, MetricArrays]]:
        train_metrics, train_rows, train_arrays = _evaluate_dataset_map(
            train_sets,
            scope,
            scheme,
            params,
            "train",
            trial_cache,
            eval_mode="full",
            save_stage_json=save_stage_json,
            trial_cache_max_entries=trial_cache_max_entries,
            fold_id=fold_id,
            heldout_group_id=heldout_group_id,
        )
        if val_sets:
            val_metrics, val_rows, val_arrays = _evaluate_dataset_map(
                val_sets,
                scope,
                scheme,
                params,
                "val",
                trial_cache,
                eval_mode="full",
                save_stage_json=save_stage_json,
                trial_cache_max_entries=trial_cache_max_entries,
                fold_id=fold_id,
                heldout_group_id=heldout_group_id,
            )
        else:
            val_metrics, val_rows, val_arrays = _empty_metrics("val"), [], {}
        test_metrics, test_rows, test_arrays = _evaluate_dataset_map(
            test_sets,
            scope,
            scheme,
            params,
            "test",
            trial_cache,
            eval_mode="full",
            save_stage_json=save_stage_json,
            trial_cache_max_entries=trial_cache_max_entries,
            fold_id=fold_id,
            heldout_group_id=heldout_group_id,
        )
        arrays_by_split = {"train": train_arrays, "val": val_arrays, "test": test_arrays}
        return train_metrics, val_metrics, test_metrics, [*train_rows, *val_rows, *test_rows], arrays_by_split

    if actual_n_jobs > 1 and int(n_repeats) > 1:
        on_progress(
            {
                "stage": "optimization_parallel_repeats_start",
                "parallel_level": "repeat",
                "motion_type": motion_type,
                "mode_idx": mode_idx,
                "mode_total": mode_total,
                "target_scope": scope.name,
                "target_scope_value": scope.value,
                "cascade_scheme": scheme.value,
                "adaptive_filter": adaptive_filter,
                "fold_id": "" if fold_id is None else int(fold_id),
                "heldout_group_id": heldout_group_id,
                "repeat_total": int(n_repeats),
                "repeat_done": 0,
                "n_jobs": int(actual_n_jobs),
            }
        )
        futures = []
        with ProcessPoolExecutor(max_workers=int(actual_n_jobs)) as executor:
            for repeat_idx in range(int(n_repeats)):
                futures.append(
                    executor.submit(
                        _run_one_optuna_repeat,
                        repeat_idx=repeat_idx,
                        motion_type=motion_type,
                        objective_sets=objective_sets,
                        objective_split=objective_split,
                        scope=scope,
                        scheme=scheme,
                        adaptive_filter=adaptive_filter,
                        objective_mode=objective_mode,
                        delay_estimation_mode=delay_estimation_mode,
                        space=space,
                        trial_param_overrides=trial_param_overrides,
                        n_trials=int(n_trials),
                        num_seed_points=int(cfg.num_seed_points),
                        penalty_value=float(penalty_value),
                        random_state=int(random_state),
                        fold_id=fold_id,
                        heldout_group_id=heldout_group_id,
                        trial_cache_max_entries=int(trial_cache_max_entries),
                    )
                )
            done_count = 0
            for future in as_completed(futures):
                item = future.result()
                done_count += 1
                history.extend(item["history"])
                value = float(item["best_value"])
                if value < best_value:
                    best_value = value
                    best_params = item["best_params"]
                    best_repeat_idx = int(item["repeat_idx"])
                    best_trial_idx = int(item["best_trial_idx"])
                on_progress(
                    {
                        "stage": "optimization_parallel_repeat_done",
                        "parallel_level": "repeat",
                        "motion_type": motion_type,
                        "mode_idx": mode_idx,
                        "mode_total": mode_total,
                        "target_scope": scope.name,
                        "target_scope_value": scope.value,
                        "cascade_scheme": scheme.value,
                        "adaptive_filter": adaptive_filter,
                        "fold_id": "" if fold_id is None else int(fold_id),
                        "heldout_group_id": heldout_group_id,
                        "repeat_current": int(item["repeat_idx"]) + 1,
                        "repeat_total": int(n_repeats),
                        "repeat_done": int(done_count),
                        "n_jobs": int(actual_n_jobs),
                        "best_value": float(value),
                        "global_best_value": float(best_value),
                    }
                )
        history.sort(key=lambda row: (int(row.get("repeat_idx", 0)), int(row.get("trial_idx", 0))))
        _recompute_history_best_so_far(history)

    for repeat_idx in ([] if actual_n_jobs > 1 and int(n_repeats) > 1 else range(int(n_repeats))):
        if optuna is not None and TPESampler is not None:
            sampler = TPESampler(
                seed=int(random_state) + repeat_idx,
                n_startup_trials=min(int(cfg.num_seed_points), int(n_trials)),
            )
            study = optuna.create_study(direction="minimize", sampler=sampler)

            def _objective(trial: optuna.trial.Trial) -> float:
                nonlocal best_value, best_params, best_repeat_idx, best_trial_idx, best_so_far

                idx_map = {
                    name: trial.suggest_int(name, 0, len(space.options(name)) - 1)
                    for name in space.names_for_filter(adaptive_filter)
                }
                params = _decode_with_seed(
                    space,
                    idx_map,
                    adaptive_filter=adaptive_filter,
                    objective_mode=objective_mode,
                    delay_estimation_mode=delay_estimation_mode,
                    mode_key=f"{scope.value}__{scheme.value}__{adaptive_filter}",
                    repeat_idx=repeat_idx,
                    random_state=random_state,
                    trial_param_overrides=trial_param_overrides,
                )
                objective_metrics = _evaluate_objective_params(params)
                value = _objective_value(objective_metrics, objective_mode, penalty_value)
                best_so_far = min(best_so_far, value)
                if value < best_value:
                    best_value = value
                    best_params = params
                    best_repeat_idx = repeat_idx
                    best_trial_idx = int(trial.number)
                _append_history(
                    history,
                    motion_type,
                    scope,
                    scheme,
                    adaptive_filter,
                    params,
                    repeat_idx,
                    int(trial.number),
                    value,
                    objective_metrics,
                    best_so_far,
                    fold_id=fold_id,
                    heldout_group_id=heldout_group_id,
                )
                _emit_trial_progress(
                    on_progress,
                    cfg,
                    motion_type,
                    scope,
                    scheme,
                    adaptive_filter,
                    objective_mode,
                    value,
                    objective_metrics,
                    best_so_far,
                    repeat_idx,
                    n_repeats,
                    int(trial.number),
                    n_trials,
                    mode_idx,
                    mode_total,
                )
                return value

            study.optimize(_objective, n_trials=int(n_trials), show_progress_bar=False)
        else:
            rng = np.random.default_rng(int(random_state) + repeat_idx)
            for trial_idx in range(int(n_trials)):
                idx_map = {
                    name: int(rng.integers(0, len(space.options(name))))
                    for name in space.names_for_filter(adaptive_filter)
                }
                params = _decode_with_seed(
                    space,
                    idx_map,
                    adaptive_filter=adaptive_filter,
                    objective_mode=objective_mode,
                    delay_estimation_mode=delay_estimation_mode,
                    mode_key=f"{scope.value}__{scheme.value}__{adaptive_filter}",
                    repeat_idx=repeat_idx,
                    random_state=random_state,
                    trial_param_overrides=trial_param_overrides,
                )
                objective_metrics = _evaluate_objective_params(params)
                value = _objective_value(objective_metrics, objective_mode, penalty_value)
                best_so_far = min(best_so_far, value)
                if value < best_value:
                    best_value = value
                    best_params = params
                    best_repeat_idx = repeat_idx
                    best_trial_idx = trial_idx
                _append_history(
                    history,
                    motion_type,
                    scope,
                    scheme,
                    adaptive_filter,
                    params,
                    repeat_idx,
                    trial_idx,
                    value,
                    objective_metrics,
                    best_so_far,
                    fold_id=fold_id,
                    heldout_group_id=heldout_group_id,
                )
                _emit_trial_progress(
                    on_progress,
                    cfg,
                    motion_type,
                    scope,
                    scheme,
                    adaptive_filter,
                    objective_mode,
                    value,
                    objective_metrics,
                    best_so_far,
                    repeat_idx,
                    n_repeats,
                    trial_idx,
                    n_trials,
                    mode_idx,
                    mode_total,
                )

    train_metrics, val_metrics, test_metrics, per_group_rows, arrays_by_split = _evaluate_params_full(best_params)
    success = bool((val_metrics if data_split_mode == "split" else train_metrics).get("success", False))
    reason = str((val_metrics if data_split_mode == "split" else train_metrics).get("reason", ""))
    return _ModeOptimisation(
        motion_type=motion_type,
        target_scope=scope,
        cascade_scheme=scheme,
        adaptive_filter=adaptive_filter,
        objective_mode=objective_mode,
        data_split_mode=data_split_mode,
        best_params=best_params,
        best_repeat_idx=best_repeat_idx,
        best_trial_idx=best_trial_idx,
        n_trials=int(n_trials),
        n_repeats=int(n_repeats),
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        per_group_rows=per_group_rows,
        history=history,
        success=success,
        reason=reason,
        fold_id=fold_id,
        heldout_group_id=heldout_group_id,
        train_group_ids=list(train_group_ids or train_sets.keys()),
        test_group_id=str(test_group_id),
        metric_arrays_by_split=arrays_by_split,
        result_level="fold" if data_split_mode == "leave_one_group_out" else "single_split",
        params_semantics="fold_best_params" if data_split_mode == "leave_one_group_out" else "best_params_for_this_split",
    )


def _normalise_trial_param_overrides(overrides: dict[str, Any] | None) -> dict[str, Any]:
    """Validate fixed ProtocolTrialParams overrides that are not sampled by Optuna.

    中文说明：Notebook 可以把静息段 HR 提取参数放在第一个初始化代码块里统一维护。
    这些参数会覆盖每个 trial 的默认值，但不会进入 Optuna 搜索空间；未知字段直接报错，
    避免因为拼写错误导致配置看似生效、实际被忽略。
    """

    if not overrides:
        return {}
    valid_names = {item.name for item in fields(ProtocolTrialParams)}
    unknown = sorted(set(overrides) - valid_names)
    if unknown:
        raise ValueError(f"Unknown ProtocolTrialParams overrides: {unknown}")
    return dict(overrides)


def _default_params_for_filter(
    space: ProtocolSearchSpace,
    adaptive_filter: str,
    objective_mode: str,
    *,
    delay_estimation_mode: str = "envelope",
    trial_param_overrides: dict[str, Any] | None = None,
) -> ProtocolTrialParams:
    values = {}
    for name in space.names_for_filter(adaptive_filter):
        value = space.options(name)[0]
        if name == "RFF_LMS_Mu_Base":
            values["LMS_Mu_Base"] = value
        else:
            values[name] = value
    values["adaptive_filter"] = adaptive_filter
    values["objective_mode"] = objective_mode
    values["delay_estimation_mode"] = str(delay_estimation_mode)
    values.update(_normalise_trial_param_overrides(trial_param_overrides))
    return ProtocolTrialParams(**values)


def _decode_with_seed(
    space: ProtocolSearchSpace,
    idx_map: dict[str, int],
    *,
    adaptive_filter: str,
    objective_mode: str,
    delay_estimation_mode: str,
    mode_key: str,
    repeat_idx: int,
    random_state: int,
    trial_param_overrides: dict[str, Any] | None = None,
) -> ProtocolTrialParams:
    params = decode_protocol_search_space(
        space,
        idx_map,
        adaptive_filter=adaptive_filter,
        objective_mode=objective_mode,
        rff_seed=0,
    )
    params = replace(params, delay_estimation_mode=str(delay_estimation_mode))
    overrides = _normalise_trial_param_overrides(trial_param_overrides)
    if overrides:
        params = replace(params, **overrides)
    if adaptive_filter == "rff_lms":
        seed_payload = {
            **{k: v for k, v in params.to_dict().items() if k != "rff_seed"},
            "mode_key": mode_key,
            "repeat_idx": int(repeat_idx),
            "random_state": int(random_state),
        }
        params = replace(params, rff_seed=_stable_int_hash(seed_payload))
    return params


def _failed_mode_optimisation(
    *,
    motion_type: str,
    scope: TargetScope,
    scheme: CascadeScheme,
    adaptive_filter: str,
    objective_mode: str,
    data_split_mode: str,
    delay_estimation_mode: str,
    space: ProtocolSearchSpace,
    n_trials: int,
    n_repeats: int,
    reason: str,
    trial_param_overrides: dict[str, Any] | None = None,
) -> _ModeOptimisation:
    """Build a stable failed mode result without stopping the batch.

    中文说明：当某个 motion_type 无法执行 LOGO 时，仍为每个 mode 写出失败指标和
    reason，避免整个批处理因为单个运动类型样本不足而中断。
    """

    metrics = _failed_metrics("test", reason)
    return _ModeOptimisation(
        motion_type=motion_type,
        target_scope=scope,
        cascade_scheme=scheme,
        adaptive_filter=adaptive_filter,
        objective_mode=objective_mode,
        data_split_mode=data_split_mode,
        best_params=_default_params_for_filter(
            space,
            adaptive_filter,
            objective_mode,
            delay_estimation_mode=delay_estimation_mode,
            trial_param_overrides=trial_param_overrides,
        ),
        best_repeat_idx=0,
        best_trial_idx=0,
        n_trials=int(n_trials),
        n_repeats=int(n_repeats),
        train_metrics=_failed_metrics("train", reason),
        val_metrics=_empty_metrics("val"),
        test_metrics=metrics,
        per_group_rows=[],
        history=[],
        success=False,
        reason=reason,
        result_level="aggregate" if data_split_mode == "leave_one_group_out" else "single_split",
        aggregation="logo_window_concat" if data_split_mode == "leave_one_group_out" else "",
        params_semantics=(
            "representative_fold_best_params_not_global"
            if data_split_mode == "leave_one_group_out"
            else "best_params_for_this_split"
        ),
    )


def _aggregate_logo_fold_results(
    *,
    motion_type: str,
    scope: TargetScope,
    scheme: CascadeScheme,
    adaptive_filter: str,
    objective_mode: str,
    data_split_mode: str,
    fold_results: list[_ModeOptimisation],
    n_trials: int,
    n_repeats: int,
) -> _ModeOptimisation:
    """Aggregate LOGO folds into one motion_type/mode result.

    中文说明：正式泛化指标在这里把所有 held-out test 窗口的 metric_arrays 拼接后
    统一计算 AAE/accuracy，绝不使用每个 fold 指标的简单平均。
    """

    if not fold_results:
        reason = "leave_one_group_out has no fold results"
        return _ModeOptimisation(
            motion_type=motion_type,
            target_scope=scope,
            cascade_scheme=scheme,
            adaptive_filter=adaptive_filter,
            objective_mode=objective_mode,
            data_split_mode=data_split_mode,
            best_params=ProtocolTrialParams(adaptive_filter=adaptive_filter, objective_mode=objective_mode),
            best_repeat_idx=0,
            best_trial_idx=0,
            n_trials=int(n_trials),
            n_repeats=int(n_repeats),
            train_metrics=_failed_metrics("train", reason),
            val_metrics=_empty_metrics("val"),
            test_metrics=_failed_metrics("test", reason),
            per_group_rows=[],
            history=[],
            success=False,
            reason=reason,
            result_level="aggregate",
            aggregation="logo_window_concat",
            params_semantics="representative_fold_best_params_not_global",
        )

    test_arrays = _concat_metric_arrays([r.metric_arrays_by_split.get("test", {}) for r in fold_results])
    train_arrays = _concat_metric_arrays([r.metric_arrays_by_split.get("train", {}) for r in fold_results])
    test_failures = [str(r.test_metrics.get("reason", "")) for r in fold_results if not r.test_metrics.get("success", False)]
    train_failures = [str(r.train_metrics.get("reason", "")) for r in fold_results if not r.train_metrics.get("success", False)]
    test_metrics = _metrics_from_arrays_with_status(test_arrays, "test", test_failures)
    train_metrics = _metrics_from_arrays_with_status(train_arrays, "train", train_failures)

    def _fold_score(result: _ModeOptimisation) -> float:
        return _objective_value(result.test_metrics, objective_mode, float("inf"))

    best_fold = min(fold_results, key=_fold_score)
    history = [row for fold in fold_results for row in fold.history]
    per_group_rows = [row for fold in fold_results for row in fold.per_group_rows]
    reason = "; ".join([r for r in test_failures[:3] if r])
    success = bool(test_metrics.get("success", False))

    # 中文注释：汇总指标算完后清掉 fold 内大数组，只保留 CSV/JSON 需要的小指标。
    for fold in fold_results:
        fold.metric_arrays_by_split = {}

    return _ModeOptimisation(
        motion_type=motion_type,
        target_scope=scope,
        cascade_scheme=scheme,
        adaptive_filter=adaptive_filter,
        objective_mode=objective_mode,
        data_split_mode=data_split_mode,
        best_params=best_fold.best_params,
        best_repeat_idx=best_fold.best_repeat_idx,
        best_trial_idx=best_fold.best_trial_idx,
        n_trials=int(n_trials),
        n_repeats=int(n_repeats),
        train_metrics=train_metrics,
        val_metrics=_empty_metrics("val"),
        test_metrics=test_metrics,
        per_group_rows=per_group_rows,
        history=history,
        success=success,
        reason=reason,
        fold_results=fold_results,
        result_level="aggregate",
        aggregation="logo_window_concat",
        params_semantics="representative_fold_best_params_not_global",
        representative_fold_id=best_fold.fold_id,
        representative_heldout_group_id=best_fold.heldout_group_id,
    )


def _evaluate_dataset_map(
    datasets: dict[str, ProtocolDataset],
    scope: TargetScope,
    scheme: CascadeScheme,
    params: ProtocolTrialParams,
    split_name: str,
    trial_cache: OrderedDict[tuple[Any, ...], ProtocolRunResult] | dict[tuple[Any, ...], ProtocolRunResult],
    *,
    eval_mode: str = "full",
    save_stage_json: bool = False,
    trial_cache_max_entries: int = _DEFAULT_TRIAL_CACHE_MAX_ENTRIES,
    fold_id: int | None = None,
    heldout_group_id: str = "",
) -> tuple[dict[str, Any], list[dict[str, Any]], MetricArrays]:
    """Run one trial params on all datasets in one split and aggregate metrics.

    中文说明：``eval_mode='light'`` 只用于 Optuna objective，不生成 DataFrame 或
    stage JSON；``eval_mode='full'`` 只在 best_params 复评估时使用。缓存键包含
    eval_mode，避免 light 结果被误当作 full 结果复用。
    """

    eval_mode = str(eval_mode).lower()
    if eval_mode not in {"light", "full"}:
        raise ValueError("eval_mode must be 'light' or 'full'")
    runs: list[ProtocolRunResult] = []
    rows: list[dict[str, Any]] = []
    for group_id, dataset in datasets.items():
        cache_key = (
            eval_mode,
            group_id,
            int(params.Fs_Target),
            int(params.TW),
            scope.value,
            scheme.value,
            str(params.adaptive_filter),
            params.cache_key(),
        )
        run = trial_cache.get(cache_key)
        if run is not None and isinstance(trial_cache, OrderedDict):
            trial_cache.move_to_end(cache_key)
        if run is None:
            run = run_protocol_trial(
                dataset,
                scheme,
                scope,
                params,
                collect_frame=eval_mode == "full",
                collect_stages=bool(save_stage_json and eval_mode == "full"),
            )
            _store_trial_cache_entry(trial_cache, cache_key, run, trial_cache_max_entries)
        runs.append(run)
        if eval_mode == "full":
            rows.append(
                _per_group_row(
                    group_id,
                    split_name,
                    run,
                    fold_id=fold_id,
                    heldout_group_id=heldout_group_id,
                )
            )
    arrays = _concat_metric_arrays([r.metric_arrays for r in runs if r.success])
    return _aggregate_runs(runs, split_name, arrays), rows, arrays


def _store_trial_cache_entry(
    trial_cache: OrderedDict[tuple[Any, ...], ProtocolRunResult] | dict[tuple[Any, ...], ProtocolRunResult],
    key: tuple[Any, ...],
    run: ProtocolRunResult,
    max_entries: int,
) -> None:
    """Store one run in a bounded insertion-ordered cache."""

    if isinstance(trial_cache, OrderedDict):
        trial_cache[key] = run
        trial_cache.move_to_end(key)
        while len(trial_cache) > int(max_entries):
            trial_cache.popitem(last=False)
        return
    trial_cache[key] = run
    while len(trial_cache) > int(max_entries):
        trial_cache.pop(next(iter(trial_cache)))


def _aggregate_runs(
    runs: list[ProtocolRunResult],
    split_name: str,
    arrays: MetricArrays | None = None,
) -> dict[str, Any]:
    """Aggregate window-level metrics across a split."""

    failures = [r.reason for r in runs if not r.success]
    arrays = arrays if arrays is not None else _concat_metric_arrays([r.metric_arrays for r in runs if r.success])
    if not arrays or arrays.get("ref_hr_bpm", np.asarray([], dtype=float)).size == 0:
        return {
            "split": split_name,
            "success": False,
            "reason": "; ".join(failures) if failures else "no successful runs",
            "baseline_aae_bpm": float("nan"),
            "adaptive_aae_bpm": float("nan"),
            "final_aae_bpm": float("nan"),
            "baseline_acc_pct": float("nan"),
            "adaptive_acc_pct": float("nan"),
            "final_acc_pct": float("nan"),
            "num_windows": 0,
            **_empty_posthoc_metrics(),
        }
    metrics = aggregate_metric_arrays(arrays, split_name=split_name)
    return {
        "split": split_name,
        "success": len(failures) == 0,
        "reason": "; ".join(failures[:3]),
        "baseline_aae_bpm": metrics["baseline_aae_bpm"],
        "adaptive_aae_bpm": metrics["adaptive_aae_bpm"],
        "final_aae_bpm": metrics["final_aae_bpm"],
        "baseline_acc_pct": metrics["baseline_acc_pct"],
        "adaptive_acc_pct": metrics["adaptive_acc_pct"],
        "final_acc_pct": metrics["final_acc_pct"],
        "num_windows": metrics["num_windows"],
        **_posthoc_metrics_from(metrics),
    }


def _concat_metric_arrays(items: list[MetricArrays]) -> MetricArrays:
    """Concatenate compact metric arrays from multiple successful runs.

    中文说明：split/all_train/LOGO 的指标都从窗口级数组拼接后统一计算；LOGO 的
    held-out test 指标尤其不能用各 fold AAE 的简单平均替代。
    """

    valid = [item for item in items if item and item.get("ref_hr_bpm", np.asarray([], dtype=float)).size]
    if not valid:
        return {}
    keys = set().union(*(item.keys() for item in valid))
    out: MetricArrays = {}
    for key in keys:
        arrays = [np.asarray(item[key]) for item in valid if key in item]
        if arrays:
            out[key] = np.concatenate(arrays)
    return out


def _per_group_row(
    group_id: str,
    split_name: str,
    run: ProtocolRunResult,
    *,
    fold_id: int | None = None,
    heldout_group_id: str = "",
) -> dict[str, Any]:
    """Build one per-group output row with optional LOGO fold metadata."""

    return {
        "fold_id": "" if fold_id is None else int(fold_id),
        "heldout_group_id": str(heldout_group_id),
        "result_level": "fold" if str(heldout_group_id) else "single_split",
        "aggregation": "",
        "params_semantics": "fold_best_params" if str(heldout_group_id) else "best_params_for_this_split",
        "split": split_name,
        "group_id": group_id,
        "target_scope": run.target_scope.value,
        "cascade_scheme": run.cascade_scheme.value,
        "adaptive_filter": run.adaptive_filter,
        "success": bool(run.success),
        "reason": run.reason,
        "baseline_aae_bpm": run.baseline_aae_bpm,
        "adaptive_aae_bpm": run.adaptive_aae_bpm,
        "final_aae_bpm": run.final_aae_bpm,
        "baseline_acc_pct": run.baseline_acc_pct,
        "adaptive_acc_pct": run.adaptive_acc_pct,
        "final_acc_pct": run.final_acc_pct,
        "time_bias_after_s": (
            float(run.time_bias_after.time_bias_after_s) if run.time_bias_after is not None else float("nan")
        ),
        "time_bias_after_mode": (
            str(run.time_bias_after.mode) if run.time_bias_after is not None else ""
        ),
        "time_bias_after_range_s": (
            f"({run.time_bias_after.search_range_s[0]:.1f}, {run.time_bias_after.search_range_s[1]:.1f})"
            if run.time_bias_after is not None
            else ""
        ),
        "time_bias_after_step_s": (
            float(run.time_bias_after.search_step_s) if run.time_bias_after is not None else float("nan")
        ),
        "time_bias_after_n_valid": (
            int(run.time_bias_after.n_valid) if run.time_bias_after is not None else 0
        ),
        "posthoc_adaptive_aae_bpm": run.posthoc_adaptive_aae_bpm,
        "posthoc_adaptive_acc_pct": run.posthoc_adaptive_acc_pct,
        "posthoc_adaptive_hit_rate_5bpm": run.posthoc_adaptive_hit_rate_5bpm,
        "posthoc_final_aae_bpm": run.posthoc_final_aae_bpm,
        "posthoc_final_acc_pct": run.posthoc_final_acc_pct,
        "posthoc_baseline_aae_bpm": run.posthoc_baseline_aae_bpm,
        "posthoc_baseline_acc_pct": run.posthoc_baseline_acc_pct,
        "posthoc_n_valid_windows": run.posthoc_n_valid_windows,
    }


def _objective_value(metrics: dict[str, Any], objective_mode: str, penalty_value: float) -> float:
    if objective_mode == "accuracy":
        acc = float(metrics.get("final_acc_pct", metrics.get("adaptive_acc_pct", float("nan"))))
        value = 100.0 - acc
    else:
        value = float(metrics.get("final_aae_bpm", metrics.get("adaptive_aae_bpm", float("nan"))))
    return value if np.isfinite(value) else float(penalty_value)


def _append_history(
    history: list[dict[str, Any]],
    motion_type: str,
    scope: TargetScope,
    scheme: CascadeScheme,
    adaptive_filter: str,
    params: ProtocolTrialParams,
    repeat_idx: int,
    trial_idx: int,
    objective_value: float,
    metrics: dict[str, Any],
    best_so_far: float,
    fold_id: int | None = None,
    heldout_group_id: str = "",
) -> None:
    history.append(
        {
            "fold_id": "" if fold_id is None else int(fold_id),
            "heldout_group_id": str(heldout_group_id),
            "motion_type": motion_type,
            "target_scope": scope.value,
            "cascade_scheme": scheme.value,
            "adaptive_filter": adaptive_filter,
            "repeat_idx": int(repeat_idx),
            "trial_idx": int(trial_idx),
            "objective_value": float(objective_value),
            "aae_bpm": float(metrics.get("adaptive_aae_bpm", float("nan"))),
            "accuracy_pct": float(metrics.get("adaptive_acc_pct", float("nan"))),
            "final_aae_bpm": float(metrics.get("final_aae_bpm", float("nan"))),
            "final_acc_pct": float(metrics.get("final_acc_pct", float("nan"))),
            "best_so_far": float(best_so_far),
            "success": bool(metrics.get("success", False)),
            "reason": str(metrics.get("reason", "")),
            "rff_seed": int(getattr(params, "rff_seed", 0)),
            "params": params.to_dict(),
        }
    )


def _emit_trial_progress(
    on_progress: Callable[[dict[str, Any]], None],
    cfg: ProtocolParams,
    motion_type: str,
    scope: TargetScope,
    scheme: CascadeScheme,
    adaptive_filter: str,
    objective_mode: str,
    objective_value: float,
    metrics: dict[str, Any],
    best_so_far: float,
    repeat_idx: int,
    n_repeats: int,
    trial_idx: int,
    n_trials: int,
    mode_idx: int,
    mode_total: int,
) -> None:
    on_progress(
        {
            "stage": "optimization",
            "motion_type": motion_type,
            "mode_idx": mode_idx,
            "mode_total": mode_total,
            "target_scope": scope.name,
            "target_scope_value": scope.value,
            "cascade_scheme": scheme.value,
            "adaptive_filter": adaptive_filter,
            "objective_mode": objective_mode,
            "objective_value": float(objective_value),
            "aae_bpm": metrics.get("adaptive_aae_bpm"),
            "accuracy_pct": metrics.get("adaptive_acc_pct"),
            "final_aae_bpm": metrics.get("final_aae_bpm"),
            "final_acc_pct": metrics.get("final_acc_pct"),
            "best_so_far": float(best_so_far),
            "repeat_idx": int(repeat_idx) + 1,
            "repeat_total": int(n_repeats),
            "trial_idx": int(trial_idx) + 1,
            "trial_total": int(n_trials),
            "debug_mode": bool(cfg.debug_mode),
        }
    )


def _write_motion_type_outputs(
    motion_dir: Path,
    motion_type: str,
    results: list[_ModeOptimisation],
) -> None:
    """Write all required per-motion_type CSV/JSON files."""

    summary_rows = [_summary_row(r) for r in results]
    summary_df = pd.DataFrame(summary_rows, columns=_summary_columns())
    summary_df.to_csv(motion_dir / "mode_summary_aae.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(motion_dir / "mode_summary_accuracy.csv", index=False, encoding="utf-8-sig")

    fold_summary = pd.DataFrame(
        [_summary_row(fold) for r in results for fold in r.fold_results],
        columns=_summary_columns(),
    )
    fold_summary.to_csv(motion_dir / "fold_summary_aae.csv", index=False, encoding="utf-8-sig")
    fold_summary.to_csv(motion_dir / "fold_summary_accuracy.csv", index=False, encoding="utf-8-sig")

    per_group = pd.DataFrame([row for r in results for row in r.per_group_rows])
    per_group.to_csv(motion_dir / "per_group_aae.csv", index=False, encoding="utf-8-sig")
    per_group.to_csv(motion_dir / "per_group_accuracy.csv", index=False, encoding="utf-8-sig")

    for adaptive_filter in _VALID_FILTERS:
        rows = [
            _best_param_row(item)
            for r in results
            if r.adaptive_filter == adaptive_filter
            for item in (r.fold_results if r.fold_results else [r])
        ]
        pd.DataFrame(rows, columns=_best_param_columns()).to_csv(
            motion_dir / f"best_params_{adaptive_filter}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    payload = {
        r.mode_key: {
            "motion_type": motion_type,
            "target_scope": r.target_scope.value,
            "cascade_scheme": r.cascade_scheme.value,
            "adaptive_filter": r.adaptive_filter,
            "objective_mode": r.objective_mode,
            "data_split_mode": r.data_split_mode,
            "result_level": r.result_level,
            "aggregation": r.aggregation,
            "params_semantics": r.params_semantics,
            "representative_fold_id": r.representative_fold_id,
            "representative_heldout_group_id": r.representative_heldout_group_id,
            "best_repeat_idx": r.best_repeat_idx,
            "best_trial_idx": r.best_trial_idx,
            "n_trials": r.n_trials,
            "n_repeats": r.n_repeats,
            "best_params": r.best_params.to_dict(),
            "representative_best_params": (
                r.best_params.to_dict() if r.result_level == "aggregate" else None
            ),
            "train_metrics": r.train_metrics,
            "val_metrics": r.val_metrics,
            "test_metrics": r.test_metrics,
            "fold_results": [
                {
                    "fold_id": fold.fold_id,
                    "heldout_group_id": fold.heldout_group_id,
                    "result_level": fold.result_level,
                    "aggregation": fold.aggregation,
                    "params_semantics": fold.params_semantics,
                    "train_group_ids": fold.train_group_ids,
                    "test_group_id": fold.test_group_id,
                    "best_repeat_idx": fold.best_repeat_idx,
                    "best_trial_idx": fold.best_trial_idx,
                    "n_trials": fold.n_trials,
                    "n_repeats": fold.n_repeats,
                    "best_params": fold.best_params.to_dict(),
                    "train_metrics": fold.train_metrics,
                    "val_metrics": fold.val_metrics,
                    "test_metrics": fold.test_metrics,
                    "success": fold.success,
                    "reason": fold.reason,
                    "trial_history": fold.history,
                }
                for fold in r.fold_results
            ],
            "success": r.success,
            "reason": r.reason,
            "trial_history": r.history,
        }
        for r in results
    }
    (motion_dir / "best_params_all.json").write_text(
        json.dumps(_jsonify(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    bayes_df = pd.DataFrame([row for r in results for row in r.history])
    bayes_df.to_csv(motion_dir / "bayes_curve_data.csv", index=False, encoding="utf-8-sig")
    _write_stage6_record_files(motion_dir, motion_type, results)


def _write_stage6_record_files(
    motion_dir: Path,
    motion_type: str,
    results: list[_ModeOptimisation],
) -> None:
    """Write compact deployment-oriented result records.

    中文说明：这些文件是新 replay、诊断图和跨运动汇总的稳定读取入口；旧的
    best_params_lms.csv / mode_summary_*.csv 暂时保留，避免破坏已有 Notebook。
    """

    records = [item for result in results for item in (result.fold_results if result.fold_results else [result])]
    pd.DataFrame(
        [_best_params_alignment_record(item) for item in records],
        columns=_best_params_alignment_columns(),
    ).to_csv(motion_dir / "best_params_and_alignment.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [_best_metrics_record(item) for item in records],
        columns=_best_metrics_columns(),
    ).to_csv(motion_dir / "best_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [_motion_frequency_params_record(item) for item in records],
        columns=_motion_frequency_params_columns(),
    ).to_csv(motion_dir / "motion_frequency_and_params.csv", index=False, encoding="utf-8-sig")
    report = {
        "motion_type": motion_type,
        "config": {
            "result_files": [
                "best_params_and_alignment.csv",
                "best_metrics.csv",
                "motion_frequency_and_params.csv",
            ],
        },
        "search_space": {},
        "best_trials": [_jsonify(_best_params_alignment_record(item)) for item in records],
        "split_metrics": [_jsonify(_best_metrics_record(item)) for item in records],
        "alignment": [
            {
                "motion_type": item.motion_type,
                "target_scope": item.target_scope.value,
                "best_tdelay_s": item.test_metrics.get("best_tdelay_s", float("nan")),
                "time_bias_after_s": item.test_metrics.get("time_bias_after_s", float("nan")),
            }
            for item in records
        ],
        "posthoc_time_bias_after": [
            {
                "target_scope": item.target_scope.value,
                "time_bias_after_s": item.test_metrics.get("time_bias_after_s", float("nan")),
                "mode": item.test_metrics.get("time_bias_after_mode", ""),
            }
            for item in records
        ],
        "qc_summary": {
            "n_qc_fallback": int(sum(int(item.test_metrics.get("n_qc_fallback", 0) or 0) for item in records)),
        },
        "fusion_source_distribution": {},
        "version_info": {"package": "ppg_hr"},
        "run_command": "",
        "git_commit_hash": _git_commit_hash(),
        "failed_samples": [item.reason for item in records if not item.success],
    }
    (motion_dir / "full_report.json").write_text(
        json.dumps(_jsonify(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _best_params_alignment_record(result: _ModeOptimisation) -> dict[str, Any]:
    params = result.best_params
    metrics = result.test_metrics
    return {
        "motion_type": result.motion_type,
        "split": _result_split_name(result),
        "mode": result.data_split_mode,
        "target_scope": result.target_scope.value,
        "objective_mode": result.objective_mode,
        "cascade_scheme": result.cascade_scheme.value,
        "adaptive_filter": result.adaptive_filter,
        "adaptive_data_type": result.cascade_scheme.value,
        "TW": float(params.TW),
        "TW_F": float(getattr(params, "TW_F", 0.0)),
        "Fs_Target": int(params.Fs_Target),
        "normalization_mode": str(getattr(params, "normalization_mode", "minmax")),
        "best_params_json": json.dumps(_jsonify(params.to_dict()), ensure_ascii=False, sort_keys=True),
        "best_tdelay_s": metrics.get("best_tdelay_s", float("nan")),
        "time_bias_after_s": metrics.get("time_bias_after_s", float("nan")),
        "alignment_score_mode": getattr(params, "Rest_Alignment_Score_Mode", "aae"),
        "pre_align_metric_json": "{}",
        "no_posthoc_final_aae_bpm": metrics.get("final_aae_bpm"),
        "no_posthoc_final_acc_pct": metrics.get("final_acc_pct"),
        "posthoc_final_aae_bpm": metrics.get("posthoc_final_aae_bpm"),
        "posthoc_final_acc_pct": metrics.get("posthoc_final_acc_pct"),
        "baseline_aae_bpm": metrics.get("baseline_aae_bpm"),
        "baseline_acc_pct": metrics.get("baseline_acc_pct"),
        "adaptive_aae_bpm": metrics.get("adaptive_aae_bpm"),
        "adaptive_acc_pct": metrics.get("adaptive_acc_pct"),
    }


def _best_metrics_record(result: _ModeOptimisation) -> dict[str, Any]:
    params = result.best_params
    metrics = result.test_metrics
    return {
        "motion_type": result.motion_type,
        "split": _result_split_name(result),
        "mode": result.data_split_mode,
        "target_scope": result.target_scope.value,
        "filter_type": result.adaptive_filter,
        "adaptive_data_type": result.cascade_scheme.value,
        "TW": float(params.TW),
        "TW_F": float(getattr(params, "TW_F", 0.0)),
        "baseline_aae_bpm": metrics.get("baseline_aae_bpm"),
        "adaptive_aae_bpm": metrics.get("adaptive_aae_bpm"),
        "final_aae_bpm": metrics.get("final_aae_bpm"),
        "baseline_acc_pct": metrics.get("baseline_acc_pct"),
        "adaptive_acc_pct": metrics.get("adaptive_acc_pct"),
        "final_acc_pct": metrics.get("final_acc_pct"),
        "posthoc_baseline_aae_bpm": metrics.get("posthoc_baseline_aae_bpm"),
        "posthoc_adaptive_aae_bpm": metrics.get("posthoc_adaptive_aae_bpm"),
        "posthoc_final_aae_bpm": metrics.get("posthoc_final_aae_bpm"),
        "posthoc_baseline_acc_pct": metrics.get("posthoc_baseline_acc_pct"),
        "posthoc_adaptive_acc_pct": metrics.get("posthoc_adaptive_acc_pct"),
        "posthoc_final_acc_pct": metrics.get("posthoc_final_acc_pct"),
        "n_windows": int(metrics.get("num_windows", 0) or 0),
        "n_valid_windows": int(metrics.get("num_windows", 0) or 0),
        "n_qc_fallback": int(metrics.get("n_qc_fallback", 0) or 0),
        "n_recovery_fallback": int(metrics.get("n_recovery_fallback", 0) or 0),
    }


def _motion_frequency_params_record(result: _ModeOptimisation) -> dict[str, Any]:
    params = result.best_params
    return {
        "motion_type": result.motion_type,
        "split": _result_split_name(result),
        "mode": result.data_split_mode,
        "motion_frequency_hz": result.test_metrics.get("motion_frequency_hz", float("nan")),
        "penalty_ref_channel": result.test_metrics.get("penalty_ref_channel", ""),
        "cascade_scheme": result.cascade_scheme.value,
        "adaptive_filter": result.adaptive_filter,
        "adaptive_data_type": result.cascade_scheme.value,
        "TW": float(params.TW),
        "TW_F": float(getattr(params, "TW_F", 0.0)),
        "max_order": int(params.max_order),
        "M_base": int(params.M_base),
        "C_scale": float(params.C_scale),
        "K_max": int(params.K_max),
        "Spec_Penalty_Width": float(params.Spec_Penalty_Width),
        "Spec_Penalty_Weight": float(params.Spec_Penalty_Weight),
        "reference_channel_ranking_summary": result.test_metrics.get("reference_channel_ranking_summary", ""),
    }


def _best_params_alignment_columns() -> list[str]:
    return [
        "motion_type", "split", "mode", "target_scope", "objective_mode",
        "cascade_scheme", "adaptive_filter", "adaptive_data_type", "TW", "TW_F",
        "Fs_Target", "normalization_mode", "best_params_json", "best_tdelay_s",
        "time_bias_after_s", "alignment_score_mode", "pre_align_metric_json",
        "no_posthoc_final_aae_bpm", "no_posthoc_final_acc_pct",
        "posthoc_final_aae_bpm", "posthoc_final_acc_pct",
        "baseline_aae_bpm", "baseline_acc_pct", "adaptive_aae_bpm", "adaptive_acc_pct",
    ]


def _best_metrics_columns() -> list[str]:
    return [
        "motion_type", "split", "mode", "target_scope", "filter_type", "adaptive_data_type",
        "TW", "TW_F", "baseline_aae_bpm", "adaptive_aae_bpm", "final_aae_bpm",
        "baseline_acc_pct", "adaptive_acc_pct", "final_acc_pct",
        "posthoc_baseline_aae_bpm", "posthoc_adaptive_aae_bpm", "posthoc_final_aae_bpm",
        "posthoc_baseline_acc_pct", "posthoc_adaptive_acc_pct", "posthoc_final_acc_pct",
        "n_windows", "n_valid_windows", "n_qc_fallback", "n_recovery_fallback",
    ]


def _motion_frequency_params_columns() -> list[str]:
    return [
        "motion_type", "split", "mode", "motion_frequency_hz", "penalty_ref_channel",
        "cascade_scheme", "adaptive_filter", "adaptive_data_type", "TW", "TW_F",
        "max_order", "M_base", "C_scale", "K_max",
        "Spec_Penalty_Width", "Spec_Penalty_Weight", "reference_channel_ranking_summary",
    ]


def _result_split_name(result: _ModeOptimisation) -> str:
    if result.fold_id is not None:
        return f"fold_{int(result.fold_id)}"
    return "test"


def _git_commit_hash() -> str:
    try:
        root = Path(__file__).resolve().parents[3]
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        return completed.stdout.strip()
    except Exception:
        return ""


def _summary_row(result: _ModeOptimisation) -> dict[str, Any]:
    row = {
        "motion_type": result.motion_type,
        "target_scope": result.target_scope.value,
        "cascade_scheme": result.cascade_scheme.value,
        "adaptive_filter": result.adaptive_filter,
        "objective_mode": result.objective_mode,
        "data_split_mode": result.data_split_mode,
        "result_level": result.result_level,
        "aggregation": result.aggregation,
        "params_semantics": result.params_semantics,
        "representative_fold_id": "" if result.representative_fold_id is None else result.representative_fold_id,
        "representative_heldout_group_id": result.representative_heldout_group_id,
        "fold_id": "" if result.fold_id is None else result.fold_id,
        "heldout_group_id": result.heldout_group_id,
        "train_group_ids": ",".join(result.train_group_ids),
        "test_group_id": result.test_group_id,
        "n_trials": result.n_trials,
        "n_repeats": result.n_repeats,
        "best_repeat_idx": result.best_repeat_idx,
        "best_trial_idx": result.best_trial_idx,
        "success": result.success,
        "reason": result.reason,
    }
    for prefix, metrics in (
        ("train", result.train_metrics),
        ("val", result.val_metrics),
        ("test", result.test_metrics),
    ):
        row[f"{prefix}_aae_bpm"] = metrics.get("adaptive_aae_bpm")
        row[f"{prefix}_accuracy_pct"] = metrics.get("adaptive_acc_pct")
        row[f"{prefix}_baseline_aae_bpm"] = metrics.get("baseline_aae_bpm")
        row[f"{prefix}_baseline_accuracy_pct"] = metrics.get("baseline_acc_pct")
        row[f"{prefix}_adaptive_aae_bpm"] = metrics.get("adaptive_aae_bpm")
        row[f"{prefix}_adaptive_accuracy_pct"] = metrics.get("adaptive_acc_pct")
        row[f"{prefix}_final_aae_bpm"] = metrics.get("final_aae_bpm")
        row[f"{prefix}_final_accuracy_pct"] = metrics.get("final_acc_pct")
        row[f"{prefix}_num_windows"] = metrics.get("num_windows")
        row[f"{prefix}_time_bias_after_s"] = metrics.get("time_bias_after_s")
        row[f"{prefix}_time_bias_after_mode"] = metrics.get("time_bias_after_mode")
        row[f"{prefix}_time_bias_after_range_s"] = metrics.get("time_bias_after_range_s")
        row[f"{prefix}_time_bias_after_step_s"] = metrics.get("time_bias_after_step_s")
        row[f"{prefix}_time_bias_after_n_valid"] = metrics.get("time_bias_after_n_valid")
        row[f"{prefix}_posthoc_adaptive_aae_bpm"] = metrics.get("posthoc_adaptive_aae_bpm")
        row[f"{prefix}_posthoc_adaptive_acc_pct"] = metrics.get("posthoc_adaptive_acc_pct")
        row[f"{prefix}_posthoc_adaptive_hit_rate_5bpm"] = metrics.get("posthoc_adaptive_hit_rate_5bpm")
        row[f"{prefix}_posthoc_final_aae_bpm"] = metrics.get("posthoc_final_aae_bpm")
        row[f"{prefix}_posthoc_final_acc_pct"] = metrics.get("posthoc_final_acc_pct")
        row[f"{prefix}_posthoc_baseline_aae_bpm"] = metrics.get("posthoc_baseline_aae_bpm")
        row[f"{prefix}_posthoc_baseline_acc_pct"] = metrics.get("posthoc_baseline_acc_pct")
        row[f"{prefix}_posthoc_n_valid_windows"] = metrics.get("posthoc_n_valid_windows")
    # 中文说明：不带 split 前缀的字段用于 best_params CSV 和手动重画优先读取；
    # 对多样本聚合若不存在唯一 bias，聚合函数会保留 NaN，避免伪装成全局泛化参数。
    for key, value in _posthoc_metrics_from(result.test_metrics).items():
        row[key] = value
    return row


def _best_param_row(result: _ModeOptimisation) -> dict[str, Any]:
    return {**_summary_row(result), **result.best_params.to_dict()}


def _best_param_columns() -> list[str]:
    """Return stable best-params CSV columns, including headers for empty files."""

    summary_cols = _summary_columns()
    param_cols = [item.name for item in fields(ProtocolTrialParams) if item.name not in summary_cols]
    return [*summary_cols, *param_cols]


def _summary_columns() -> list[str]:
    """Return stable summary CSV columns, including headers for empty LOGO files.

    中文说明：非 LOGO 模式也会输出空的 fold_summary_*.csv；固定表头可以让
    Notebook 检查单元格安全读取空 fold summary。
    """

    return [
        "motion_type",
        "target_scope",
        "cascade_scheme",
        "adaptive_filter",
        "objective_mode",
        "data_split_mode",
        "result_level",
        "aggregation",
        "params_semantics",
        "representative_fold_id",
        "representative_heldout_group_id",
        "fold_id",
        "heldout_group_id",
        "train_group_ids",
        "test_group_id",
        "n_trials",
        "n_repeats",
        "best_repeat_idx",
        "best_trial_idx",
        "success",
        "reason",
        "train_aae_bpm",
        "train_accuracy_pct",
        "train_baseline_aae_bpm",
        "train_baseline_accuracy_pct",
        "train_adaptive_aae_bpm",
        "train_adaptive_accuracy_pct",
        "train_final_aae_bpm",
        "train_final_accuracy_pct",
        "train_num_windows",
        "val_aae_bpm",
        "val_accuracy_pct",
        "val_baseline_aae_bpm",
        "val_baseline_accuracy_pct",
        "val_adaptive_aae_bpm",
        "val_adaptive_accuracy_pct",
        "val_final_aae_bpm",
        "val_final_accuracy_pct",
        "val_num_windows",
        "test_aae_bpm",
        "test_accuracy_pct",
        "test_baseline_aae_bpm",
        "test_baseline_accuracy_pct",
        "test_adaptive_aae_bpm",
        "test_adaptive_accuracy_pct",
        "test_final_aae_bpm",
        "test_final_accuracy_pct",
        "test_num_windows",
        "time_bias_after_s",
        "time_bias_after_mode",
        "time_bias_after_range_s",
        "time_bias_after_step_s",
        "time_bias_after_n_valid",
        "posthoc_adaptive_aae_bpm",
        "posthoc_adaptive_acc_pct",
        "posthoc_adaptive_hit_rate_5bpm",
        "posthoc_final_aae_bpm",
        "posthoc_final_acc_pct",
        "posthoc_baseline_aae_bpm",
        "posthoc_baseline_acc_pct",
        "posthoc_n_valid_windows",
        "train_time_bias_after_s",
        "train_time_bias_after_mode",
        "train_time_bias_after_range_s",
        "train_time_bias_after_step_s",
        "train_time_bias_after_n_valid",
        "train_posthoc_adaptive_aae_bpm",
        "train_posthoc_adaptive_acc_pct",
        "train_posthoc_adaptive_hit_rate_5bpm",
        "train_posthoc_final_aae_bpm",
        "train_posthoc_final_acc_pct",
        "train_posthoc_baseline_aae_bpm",
        "train_posthoc_baseline_acc_pct",
        "train_posthoc_n_valid_windows",
        "val_time_bias_after_s",
        "val_time_bias_after_mode",
        "val_time_bias_after_range_s",
        "val_time_bias_after_step_s",
        "val_time_bias_after_n_valid",
        "val_posthoc_adaptive_aae_bpm",
        "val_posthoc_adaptive_acc_pct",
        "val_posthoc_adaptive_hit_rate_5bpm",
        "val_posthoc_final_aae_bpm",
        "val_posthoc_final_acc_pct",
        "val_posthoc_baseline_aae_bpm",
        "val_posthoc_baseline_acc_pct",
        "val_posthoc_n_valid_windows",
        "test_time_bias_after_s",
        "test_time_bias_after_mode",
        "test_time_bias_after_range_s",
        "test_time_bias_after_step_s",
        "test_time_bias_after_n_valid",
        "test_posthoc_adaptive_aae_bpm",
        "test_posthoc_adaptive_acc_pct",
        "test_posthoc_adaptive_hit_rate_5bpm",
        "test_posthoc_final_aae_bpm",
        "test_posthoc_final_acc_pct",
        "test_posthoc_baseline_aae_bpm",
        "test_posthoc_baseline_acc_pct",
        "test_posthoc_n_valid_windows",
    ]


def _plot_bayes_curves(
    motion_dir: Path,
    motion_type: str,
    results: list[_ModeOptimisation],
    objective_mode: str,
) -> Path:
    """Plot dynamic Bayesian training curves for one motion type."""

    out_path = motion_dir / "bayes_curve.png"
    plt = _prepare_matplotlib(motion_dir)
    n = max(1, len(results))
    cols = min(3, n)
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 3.4 * rows), squeeze=False)
    flat = axes.ravel()
    for ax in flat[n:]:
        ax.axis("off")
    for ax, result in zip(flat, results, strict=False):
        hist = result.history
        ax.grid(True, alpha=0.25)
        ax.set_title(f"{result.target_scope.value} / {result.cascade_scheme.value} / {result.adaptive_filter}")
        ax.set_xlabel("trial index")
        ax.set_ylabel("objective value")
        if not hist:
            ax.text(0.5, 0.5, "no trial history", ha="center", va="center", transform=ax.transAxes)
            continue
        x = np.arange(len(hist), dtype=float)
        y = np.asarray([float(item.get("objective_value", np.nan)) for item in hist], dtype=float)
        best = np.asarray([float(item.get("best_so_far", np.nan)) for item in hist], dtype=float)
        ax.plot(x, y, color="0.55", marker="o", markersize=3, linewidth=1.0, label="trial")
        ax.plot(x, best, color="#1f77b4", linewidth=1.6, label="best so far")
        ax.legend(loc="upper right", fontsize=8)
        text = (
            f"best AAE={result.test_metrics.get('adaptive_aae_bpm', np.nan):.2f}\n"
            f"best accuracy={result.test_metrics.get('adaptive_acc_pct', np.nan):.1f}%\n"
            f"objective={objective_mode}"
        )
        ax.text(0.02, 0.98, text, ha="left", va="top", transform=ax.transAxes, fontsize=8)
    fig.suptitle(f"{motion_type} Bayesian training curves", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return motion_dir / "bayes_curve_data.csv"


def _write_final_summary(
    final_dir: Path,
    all_results: dict[str, list[_ModeOptimisation]],
    active_scopes: list[TargetScope],
    objective_mode: str,
    data_split_mode: str,
) -> dict[str, Path]:
    """Write strict and post-hoc cross-motion-type summary CSV files."""

    header = [
        "motion_type",
        "target_scope",
        "cascade_scheme",
        "adaptive_filter",
        "result_level",
        "aggregation",
        "params_semantics",
        "metric_value",
        "baseline_metric_value",
        "adaptive_metric_value",
        "final_metric_value",
        "objective_mode",
        "data_split_mode",
        "n_trials",
        "n_repeats",
        "success",
        "reason",
    ]
    paths: dict[str, Path] = {}
    for scope in [
        TargetScope.MOTION_ONLY,
        TargetScope.MOTION_AND_RECOVERY,
        TargetScope.MOTION_POST10,
        TargetScope.GLOBAL,
    ]:
        for metric_name, columns, filename in (
            (
                "aae",
                ("baseline_aae_bpm", "adaptive_aae_bpm", "final_aae_bpm"),
                f"{scope.value}_test_aae.csv",
            ),
            (
                "accuracy",
                ("baseline_acc_pct", "adaptive_acc_pct", "final_acc_pct"),
                f"{scope.value}_test_accuracy.csv",
            ),
            (
                "posthoc_aae",
                ("posthoc_baseline_aae_bpm", "posthoc_adaptive_aae_bpm", "posthoc_final_aae_bpm"),
                f"{scope.value}_test_posthoc_aae.csv",
            ),
            (
                "posthoc_accuracy",
                ("posthoc_baseline_acc_pct", "posthoc_adaptive_acc_pct", "posthoc_final_acc_pct"),
                f"{scope.value}_test_posthoc_accuracy.csv",
            ),
        ):
            baseline_col, adaptive_col, final_col = columns
            rows: list[dict[str, Any]] = []
            if scope in active_scopes:
                for motion_type, results in all_results.items():
                    for result in results:
                        if result.target_scope != scope:
                            continue
                        rows.append(
                            {
                                "motion_type": motion_type,
                                "target_scope": scope.value,
                                "cascade_scheme": result.cascade_scheme.value,
                                "adaptive_filter": result.adaptive_filter,
                                "result_level": result.result_level,
                                "aggregation": result.aggregation,
                                "params_semantics": result.params_semantics,
                                "metric_value": result.test_metrics.get(final_col),
                                "baseline_metric_value": result.test_metrics.get(baseline_col),
                                "adaptive_metric_value": result.test_metrics.get(adaptive_col),
                                "final_metric_value": result.test_metrics.get(final_col),
                                "objective_mode": objective_mode,
                                "data_split_mode": data_split_mode,
                                "n_trials": result.n_trials,
                                "n_repeats": result.n_repeats,
                                "success": result.test_metrics.get("success"),
                                "reason": result.test_metrics.get("reason"),
                            }
                        )
            path = final_dir / filename
            pd.DataFrame(rows, columns=header).to_csv(path, index=False, encoding="utf-8-sig")
            paths[f"{scope.value}_{metric_name}"] = path
    return paths


def _write_batch_summary(path: Path, rows: list[dict[str, Any]]) -> Path:
    fieldnames = ["motion_type", "sample", "status", "reason", "split", "result_csv", "report_json"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = pd.DataFrame(rows, columns=fieldnames)
        writer.to_csv(f, index=False)
    return path


def redraw_best_param_hr_curves(
    *,
    sensor_csv_path: str | Path,
    ref_csv_path: str | Path,
    target_scope: str,
    cascade_scheme: str,
    adaptive_filter: str,
    best_param_csv_path: str | Path,
    output_dir: str | Path,
    fs_origin: int = 100,
    fold_id: int | None = None,
    heldout_group_id: str | None = None,
    enable_time_bias_after: bool | None = None,
    time_bias_after_range_s: tuple[float, float] | None = None,
    time_bias_after_step_s: float | None = None,
    time_bias_after_mode: str | None = None,
) -> dict[str, Path]:
    """Re-run one sample with one best-param row and draw HR curves.

    中文说明：这是 Notebook 手动重画单元格使用的函数。主训练流程不会调用它。
    图 A 只画训练段；图 B 画全局曲线，非目标段使用普通 PPG FFT HR，目标段使用
    adaptive HR，并只保留这一条合并后的 ``ppg_hr_bpm`` 与后对齐参考曲线。
    """

    scope = TargetScope(target_scope)
    scheme = CascadeScheme(cascade_scheme)
    adaptive_filter = str(adaptive_filter)
    best_row = _best_param_row_from_csv(
        best_param_csv_path,
        scope,
        scheme,
        adaptive_filter,
        fold_id=fold_id,
        heldout_group_id=heldout_group_id,
    )
    params = _params_from_best_csv(
        best_param_csv_path,
        scope,
        scheme,
        adaptive_filter,
        fold_id=fold_id,
        heldout_group_id=heldout_group_id,
    )
    dataset = load_and_preprocess_protocol(sensor_csv_path, ref_csv_path, fs_origin=fs_origin)
    run = run_protocol_trial(dataset, scheme, scope, params)
    if not run.success or run.frame.empty:
        raise RuntimeError(f"手动重画失败: {run.reason}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plt = _prepare_matplotlib(out)
    group_id = Path(sensor_csv_path).stem.removeprefix("multi_")
    train_path = out / f"manual_training_hr_{group_id}_{scope.value}_{scheme.value}_{adaptive_filter}.png"
    global_path = out / f"manual_global_hr_{group_id}_{scope.value}_{scheme.value}_{adaptive_filter}.png"
    global_csv_path = out / f"manual_global_hr_{group_id}_{scope.value}_{scheme.value}_{adaptive_filter}.csv"

    frame = run.frame.copy()
    train_frame = frame.loc[frame["is_filtered_segment"].astype(bool)].copy()
    _plot_hr_frame(
        plt,
        train_frame,
        train_path,
        title=f"{group_id} training-scope HR ({scope.value} / {scheme.value} / {adaptive_filter})",
        adaptive_label=(
            f"adaptive HR AAE={run.adaptive_aae_bpm:.2f} bpm, "
            f"accuracy={run.adaptive_acc_pct:.1f}%"
        ),
    )

    combined = _global_combined_hr(dataset, params, frame)
    bias, bias_source, bias_mode = _time_bias_after_for_redraw(
        best_row,
        dataset,
        frame,
        combined,
        params,
        enable_time_bias_after=enable_time_bias_after,
        time_bias_after_range_s=time_bias_after_range_s,
        time_bias_after_step_s=time_bias_after_step_s,
        time_bias_after_mode=time_bias_after_mode,
    )
    time_s = frame["time_s"].to_numpy(dtype=float)
    ref_after = np.interp(
        time_s + float(bias),
        np.asarray(dataset.ref_time_s, dtype=float),
        np.asarray(dataset.ref_hr_bpm, dtype=float),
        left=np.nan,
        right=np.nan,
    )
    valid = np.isfinite(combined) & np.isfinite(ref_after)
    global_frame = frame.loc[valid].copy()
    global_frame["ppg_hr_bpm"] = np.asarray(combined, dtype=float)[valid]
    global_frame["ref_hr_after_bpm"] = ref_after[valid]
    global_frame["time_bias_after_s"] = float(bias)
    global_frame["time_bias_after_source"] = bias_source
    global_frame["time_bias_after_mode"] = bias_mode
    global_frame["abs_err_after_bpm"] = np.abs(
        global_frame["ppg_hr_bpm"].to_numpy(dtype=float) - global_frame["ref_hr_after_bpm"].to_numpy(dtype=float)
    )
    filtered_valid = global_frame["is_filtered_segment"].astype(bool).to_numpy()
    target_abs_err = global_frame["abs_err_after_bpm"].to_numpy(dtype=float)[filtered_valid]
    posthoc_aae = _nanmean(target_abs_err)
    posthoc_acc = _accuracy_from_abs_err(target_abs_err)
    tdelay = float(run.alignment_info.best_tdelay_s) if run.alignment_info is not None else float("nan")
    _plot_manual_global_hr_frame(
        plt,
        global_frame,
        global_path,
        title=(
            f"{group_id} global HR ({scope.value} / {scheme.value} / {adaptive_filter}) | "
            f"Tdelay={tdelay:.2f}s, time_bias_after={float(bias):.2f}s, "
            f"posthoc AAE={posthoc_aae:.2f} bpm, hit-rate={posthoc_acc:.1f}%"
        ),
    )
    _write_manual_global_hr_csv(global_frame, global_csv_path)
    return {"training_scope": train_path, "global": global_path, "global_csv": global_csv_path}


def _time_bias_after_for_redraw(
    best_row: pd.Series,
    dataset: ProtocolDataset,
    frame: pd.DataFrame,
    combined_hr_bpm: np.ndarray,
    params: ProtocolTrialParams,
    *,
    enable_time_bias_after: bool | None,
    time_bias_after_range_s: tuple[float, float] | None,
    time_bias_after_step_s: float | None,
    time_bias_after_mode: str | None,
) -> tuple[float, str, str]:
    """Return the post-hoc bias used by manual redraw.

    中文说明：手动重画优先复用 best_params CSV 中已保存的
    ``time_bias_after_s``。若旧 CSV 没有该列且启用了后对齐，才在本次重画中
    对已经生成的 HR 曲线重新做 post-hoc 搜索，并标记来源为
    ``computed_during_redraw``。
    """

    mode = str(time_bias_after_mode or getattr(params, "Time_Bias_After_Mode", "posthoc_oracle_alignment"))
    if "time_bias_after_s" in best_row and pd.notna(best_row["time_bias_after_s"]):
        try:
            value = float(best_row["time_bias_after_s"])
            if np.isfinite(value):
                saved_mode = str(best_row.get("time_bias_after_mode", mode) or mode)
                return value, "saved_best_params", saved_mode
        except (TypeError, ValueError):
            pass

    enabled = bool(getattr(params, "Enable_Time_Bias_After", True)) if enable_time_bias_after is None else bool(
        enable_time_bias_after
    )
    if not enabled:
        return 0.0, "disabled", mode

    search_range = (
        tuple(float(x) for x in time_bias_after_range_s)
        if time_bias_after_range_s is not None
        else tuple(float(x) for x in getattr(params, "Time_Bias_After_Range_S", (-5.0, 5.0)))
    )
    if len(search_range) != 2:
        raise ValueError("time_bias_after_range_s must contain exactly two values")
    search_step = (
        float(time_bias_after_step_s)
        if time_bias_after_step_s is not None
        else float(getattr(params, "Time_Bias_After_Step_S", 1.0))
    )
    mask = frame["is_filtered_segment"].astype(bool).to_numpy() if "is_filtered_segment" in frame else np.ones(
        len(frame),
        dtype=bool,
    )
    adaptive = (
        frame["adaptive_hr_bpm"].to_numpy(dtype=float)
        if "adaptive_hr_bpm" in frame
        else np.asarray(combined_hr_bpm, dtype=float)
    )
    result = search_time_bias_after(
        frame["time_s"].to_numpy(dtype=float)[mask],
        adaptive[mask],
        np.asarray(dataset.ref_time_s, dtype=float),
        np.asarray(dataset.ref_hr_bpm, dtype=float),
        search_range_s=(float(search_range[0]), float(search_range[1])),
        search_step_s=search_step,
        min_valid=2,
        mode=mode,
    )
    return float(result.time_bias_after_s), "computed_during_redraw", str(result.mode)


def _write_manual_global_hr_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write the redraw global HR curve with stable post-hoc columns."""

    columns = [
        "time_s",
        "segment_label",
        "is_filtered_segment",
        "ppg_hr_bpm",
        "ref_hr_after_bpm",
        "time_bias_after_s",
        "abs_err_after_bpm",
        "time_bias_after_source",
        "time_bias_after_mode",
    ]
    frame.loc[:, [c for c in columns if c in frame.columns]].to_csv(path, index=False, encoding="utf-8-sig")


def _plot_manual_global_hr_frame(
    plt: Any,
    frame: pd.DataFrame,
    out_path: Path,
    *,
    title: str,
) -> None:
    """Plot one merged PPG-derived HR curve against the shifted reference HR."""

    fig, ax = plt.subplots(1, 1, figsize=(12, 4.8))
    x = frame["time_s"].to_numpy(dtype=float)
    ax.plot(x, frame["ref_hr_after_bpm"].to_numpy(dtype=float), color="black", linewidth=1.6, label="ref HR after")
    ax.plot(x, frame["ppg_hr_bpm"].to_numpy(dtype=float), color="#1f77b4", linewidth=1.4, label="PPG HR")
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("HR (bpm)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _params_from_best_csv(
    best_param_csv_path: str | Path,
    scope: TargetScope,
    scheme: CascadeScheme,
    adaptive_filter: str,
    *,
    fold_id: int | None = None,
    heldout_group_id: str | None = None,
) -> ProtocolTrialParams:
    row = _best_param_row_from_csv(
        best_param_csv_path,
        scope,
        scheme,
        adaptive_filter,
        fold_id=fold_id,
        heldout_group_id=heldout_group_id,
    )

    values: dict[str, Any] = {}
    for item in fields(ProtocolTrialParams):
        if item.name not in row or pd.isna(row[item.name]):
            continue
        default = item.default
        value = row[item.name]
        if isinstance(default, bool):
            values[item.name] = _parse_bool_param(value)
        elif isinstance(default, int):
            values[item.name] = int(value)
        elif isinstance(default, float):
            values[item.name] = float(value)
        elif isinstance(default, tuple):
            values[item.name] = _parse_tuple_param(value, default)
        else:
            values[item.name] = str(value)
    values["adaptive_filter"] = adaptive_filter
    return ProtocolTrialParams(**values)


def _parse_bool_param(value: Any) -> bool:
    """Parse bool-ish CSV values without treating the string 'False' as True."""

    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes", "y"}:
            return True
        if text in {"false", "0", "no", "n", ""}:
            return False
    return bool(value)


def _best_param_row_from_csv(
    best_param_csv_path: str | Path,
    scope: TargetScope,
    scheme: CascadeScheme,
    adaptive_filter: str,
    *,
    fold_id: int | None = None,
    heldout_group_id: str | None = None,
) -> pd.Series:
    """Select the best-params row used by manual redraw and params decoding."""

    df = pd.read_csv(best_param_csv_path)
    mask = pd.Series(True, index=df.index)
    for column, value in (
        ("target_scope", scope.value),
        ("cascade_scheme", scheme.value),
        ("adaptive_filter", adaptive_filter),
    ):
        if column in df.columns:
            mask &= df[column].astype(str) == str(value)
    if fold_id is not None and "fold_id" in df.columns:
        mask &= pd.to_numeric(df["fold_id"], errors="coerce") == int(fold_id)
    if heldout_group_id is not None and "heldout_group_id" in df.columns:
        mask &= df["heldout_group_id"].astype(str) == str(heldout_group_id)
    if mask.any():
        candidates = df.loc[mask].copy()
        row = _select_best_param_row_for_redraw(candidates, fold_id, heldout_group_id)
    elif not df.empty:
        row = df.iloc[0]
    else:
        raise ValueError(f"best param CSV is empty: {best_param_csv_path}")
    return row


def _parse_tuple_param(value: Any, default: tuple[Any, ...]) -> tuple[Any, ...]:
    """从 best_params CSV 中恢复 tuple 参数，保证手动重画复用同一套配置。

    CSV 会把 ``(40.0, 180.0)`` 这类参数写成字符串；这里用安全的
    ``ast.literal_eval`` 解析，解析失败时回退到默认值，避免旧 CSV 或手工编辑
    的文件破坏后续 Tdelay 缓存 key。
    """

    try:
        parsed = ast.literal_eval(value) if isinstance(value, str) else value
        if not isinstance(parsed, (list, tuple)):
            return default
        return tuple(type(default[idx])(item) if idx < len(default) else item for idx, item in enumerate(parsed))
    except (SyntaxError, ValueError, TypeError):
        return default


def _select_best_param_row_for_redraw(
    candidates: pd.DataFrame,
    fold_id: int | None,
    heldout_group_id: str | None,
) -> pd.Series:
    """Choose one best-param row for manual redraw, including LOGO CSVs.

    中文说明：LOGO 的 best_params CSV 每个 fold 一行；若用户未指定 fold_id 或
    heldout_group_id，则默认选择该 mode 下 test 指标最优的一行并打印提示。
    """

    if len(candidates) <= 1:
        return candidates.iloc[0]
    objective = str(candidates.get("objective_mode", pd.Series(["aae"])).iloc[0]).lower()
    if objective == "accuracy" and "test_accuracy_pct" in candidates.columns:
        score = pd.to_numeric(candidates["test_accuracy_pct"], errors="coerce")
        idx = score.idxmax() if score.notna().any() else candidates.index[0]
    elif "test_aae_bpm" in candidates.columns:
        score = pd.to_numeric(candidates["test_aae_bpm"], errors="coerce")
        idx = score.idxmin() if score.notna().any() else candidates.index[0]
    else:
        idx = candidates.index[0]
    row = candidates.loc[idx]
    if fold_id is None and heldout_group_id is None and "fold_id" in candidates.columns:
        print(
            "[redraw] best_params CSV 含多行 LOGO fold；未指定 fold_id/heldout_group_id，"
            f"默认使用 fold_id={row.get('fold_id', '')}, heldout_group_id={row.get('heldout_group_id', '')}。"
        )
    return row


def _global_combined_hr(
    dataset: ProtocolDataset,
    params: ProtocolTrialParams,
    frame: pd.DataFrame,
) -> np.ndarray:
    """Use adaptive HR inside scope and 0.5-2 Hz Hamming FFT outside scope."""

    base = _get_trial_base(dataset, params)
    if base.aligned is None:
        return frame["adaptive_hr_bpm"].to_numpy(dtype=float)
    ds = base.aligned.dataset
    fs = int(ds.fs)
    win_len = int(round(float(params.TW) * fs))
    fft_hr: list[float] = []
    for idx in frame["window_idx"].to_numpy(dtype=int):
        if idx >= len(base.aligned.window_starts_s):
            fft_hr.append(float("nan"))
            continue
        start = int(round(float(base.aligned.window_starts_s[idx]) * fs))
        segment = ds.ppg_green[start : start + win_len]
        fft_hr.append(_window_fft_hr(segment, fs, 0.5, 2.0))
    fft_arr = np.asarray(fft_hr, dtype=float)
    adaptive = frame["adaptive_hr_bpm"].to_numpy(dtype=float)
    mask = frame["is_filtered_segment"].to_numpy(dtype=bool)
    return np.where(mask, adaptive, fft_arr)


def _plot_hr_frame(
    plt: Any,
    frame: pd.DataFrame,
    out_path: Path,
    *,
    title: str,
    adaptive_label: str,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(12, 4.8))
    x = frame["time_s"].to_numpy(dtype=float)
    ax.plot(x, frame["ref_hr_bpm"].to_numpy(dtype=float), color="black", linewidth=1.6, label="true HR")
    ax.plot(
        x,
        frame["baseline_ppg_hr_bpm"].to_numpy(dtype=float),
        color="0.55",
        linestyle="--",
        linewidth=1.2,
        label="PPG baseline HR",
    )
    ax.plot(x, frame["adaptive_hr_bpm"].to_numpy(dtype=float), color="#1f77b4", linewidth=1.4, label=adaptive_label)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("HR (bpm)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _empty_metrics(split_name: str) -> dict[str, Any]:
    return {
        "split": split_name,
        "success": True,
        "reason": "",
        "baseline_aae_bpm": float("nan"),
        "adaptive_aae_bpm": float("nan"),
        "final_aae_bpm": float("nan"),
        "baseline_acc_pct": float("nan"),
        "adaptive_acc_pct": float("nan"),
        "final_acc_pct": float("nan"),
        "num_windows": 0,
        **_empty_posthoc_metrics(),
    }


def _failed_metrics(split_name: str, reason: str) -> dict[str, Any]:
    return {
        "split": split_name,
        "success": False,
        "reason": str(reason),
        "baseline_aae_bpm": float("nan"),
        "adaptive_aae_bpm": float("nan"),
        "final_aae_bpm": float("nan"),
        "baseline_acc_pct": float("nan"),
        "adaptive_acc_pct": float("nan"),
        "final_acc_pct": float("nan"),
        "num_windows": 0,
        **_empty_posthoc_metrics(),
    }


def _empty_posthoc_metrics() -> dict[str, Any]:
    """Return stable empty fields for time_bias_after/posthoc CSV output."""

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


def _posthoc_metrics_from(metrics: dict[str, Any]) -> dict[str, Any]:
    """Pick only the explicit post-hoc fields from an aggregate metric dict."""

    out = _empty_posthoc_metrics()
    for key in out:
        if key in metrics:
            out[key] = metrics[key]
    return out


def _metrics_from_arrays_with_status(
    arrays: MetricArrays,
    split_name: str,
    failures: list[str],
) -> dict[str, Any]:
    """Aggregate concatenated arrays and attach split success/reason fields."""

    if not arrays or arrays.get("ref_hr_bpm", np.asarray([], dtype=float)).size == 0:
        reason = "; ".join([x for x in failures if x]) or "no successful runs"
        return _failed_metrics(split_name, reason)
    metrics = aggregate_metric_arrays(arrays, split_name=split_name)
    return {
        **metrics,
        "success": len([x for x in failures if x]) == 0,
        "reason": "; ".join([x for x in failures[:3] if x]),
    }


def _nanmean(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def _accuracy_from_abs_err(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr <= 5.0) * 100.0)


def _stable_int_hash(payload: Any) -> int:
    text = json.dumps(_jsonify(payload), ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)


def _prepare_matplotlib(out_path: Path) -> Any:
    mpl_cache = out_path / ".matplotlib" if out_path.suffix == "" else out_path.parent / ".matplotlib"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


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
