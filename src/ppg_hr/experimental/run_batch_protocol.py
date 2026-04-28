"""Top-level batch runner for grouped adaptive protocol experiments.

中文说明：本模块是 Notebook 和脚本调用的总调度器。它按顺序完成文件配对、QC、
预处理、信号图、按运动类型划分 train/val/test 或 all_train、逐模式 Optuna 优化、
贝叶斯训练曲线和跨运动类型汇总输出。
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import shutil
from dataclasses import dataclass, field, fields, replace
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
from .alignment import _window_fft_hr
from .cascade_solver import ProtocolRunResult, _get_trial_base, run_protocol_trial
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
_VALID_SPLIT_MODES = ("split", "all_train")


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

    @property
    def mode_key(self) -> str:
        return f"{self.target_scope.value}__{self.cascade_scheme.value}__{self.adaptive_filter}"


def build_output_run_name(
    target_scopes: list[str | TargetScope],
    cascade_schemes: list[str | CascadeScheme],
    adaptive_filters: list[str],
    objective_mode: str,
    data_split_mode: str,
) -> str:
    """Build the run_name required by the Notebook output layout."""

    scopes = "-".join(TargetScope(x).value for x in target_scopes)
    schemes = "-".join(CascadeScheme(x).value for x in cascade_schemes)
    filters = "-".join(str(x) for x in adaptive_filters)
    return f"{scopes}__{schemes}__{filters}__{objective_mode}__{data_split_mode}"


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
    debug_mode: bool = False,
    search_space: ProtocolSearchSpace | None = None,
    target_scopes: list[TargetScope | str] | None = None,
    cascade_schemes: list[CascadeScheme | str] | None = None,
    adaptive_filters: list[str] | None = None,
    objective_mode: str = "aae",
    data_split_mode: str = "split",
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
    if objective_mode not in _VALID_OBJECTIVES:
        raise ValueError(f"objective_mode must be one of {_VALID_OBJECTIVES}")
    if data_split_mode not in _VALID_SPLIT_MODES:
        raise ValueError(f"data_split_mode must be one of {_VALID_SPLIT_MODES}")

    scopes = [TargetScope(x) for x in (target_scopes or [TargetScope.MOTION_ONLY, TargetScope.MOTION_AND_RECOVERY])]
    schemes = [CascadeScheme(x) for x in (cascade_schemes or list(CascadeScheme))]
    filters = [str(x) for x in (adaptive_filters or ["lms"])]
    unsupported = [x for x in filters if x not in _VALID_FILTERS]
    if unsupported:
        raise ValueError(f"Unsupported adaptive_filters: {unsupported}")

    input_path = Path(input_dir).resolve()
    if output_root is None:
        if csv_out_dir is not None:
            root_out = Path(csv_out_dir).resolve().parent
        else:
            run_name = build_output_run_name(scopes, schemes, filters, objective_mode, data_split_mode)
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

    splits = _build_splits(
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
        split_rows = _split_rows(splits[motion_type], pair_by_group)
        split_path = motion_dir / "split_files.csv"
        pd.DataFrame(split_rows).to_csv(split_path, index=False, encoding="utf-8-sig")

        train_ids = splits[motion_type]["train"]
        val_ids = splits[motion_type]["val"]
        test_ids = splits[motion_type]["test"]
        train_sets = {gid: datasets[gid] for gid in train_ids}
        val_sets = {gid: datasets[gid] for gid in val_ids}
        test_sets = {gid: datasets[gid] for gid in test_ids}
        if data_split_mode == "all_train":
            val_sets = {}

        mode_results: list[_ModeOptimisation] = []
        trial_cache: dict[tuple[Any, ...], ProtocolRunResult] = {}
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
                    result = _optimise_group_mode(
                        motion_type=motion_type,
                        train_sets=train_sets,
                        val_sets=val_sets,
                        test_sets=test_sets,
                        scope=scope,
                        scheme=scheme,
                        adaptive_filter=adaptive_filter,
                        objective_mode=objective_mode,
                        data_split_mode=data_split_mode,
                        cfg=cfg,
                        space=space,
                        n_trials=budget["n_trials"],
                        n_repeats=budget["n_repeats"],
                        trial_cache=trial_cache,
                        penalty_value=float(penalty_value),
                        mode_idx=mode_counter,
                        mode_total=total_modes,
                        random_state=int(random_state),
                        on_progress=_progress,
                    )
                    mode_results.append(result)
                    gc.collect()
        all_mode_results[motion_type] = mode_results
        _write_motion_type_outputs(motion_dir, motion_type, mode_results)
        bayes_path = _plot_bayes_curves(motion_dir, motion_type, mode_results, objective_mode)
        bayes_tables[motion_type] = bayes_path
        batch_rows.append(
            {
                "motion_type": motion_type,
                "sample": ",".join(train_ids + val_ids + test_ids),
                "status": "ok",
                "reason": "",
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


def _build_splits(
    grouped_pairs: dict[str, list[SamplePair]],
    *,
    data_split_mode: str,
    val_groups_per_type: int,
    test_groups_per_type: int,
    random_state: int,
) -> dict[str, dict[str, list[str]]]:
    """Build fixed train/val/test or all_train splits per motion type."""

    splits: dict[str, dict[str, list[str]]] = {}
    for motion_type, pairs in grouped_pairs.items():
        pairs_sorted = sorted(pairs, key=lambda p: (p.motion_index, p.motion_id))
        ids = [p.motion_id for p in pairs_sorted]
        if data_split_mode == "all_train":
            splits[motion_type] = {"train": ids, "val": [], "test": ids}
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
        splits[motion_type] = {"train": train, "val": val, "test": test}
    return splits


def _split_rows(split: dict[str, list[str]], pair_by_group: dict[str, SamplePair]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name, ids in split.items():
        for gid in ids:
            pair = pair_by_group.get(gid)
            rows.append(
                {
                    "split": split_name,
                    "group_id": gid,
                    "motion_type": pair.motion_type if pair is not None else "",
                    "data_file": str(pair.sensor_csv) if pair is not None else "",
                    "ref_file": str(pair.ref_csv) if pair is not None else "",
                }
            )
    return rows


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
    cfg: ProtocolParams,
    space: ProtocolSearchSpace,
    n_trials: int,
    n_repeats: int,
    trial_cache: dict[tuple[Any, ...], ProtocolRunResult],
    penalty_value: float,
    mode_idx: int,
    mode_total: int,
    random_state: int,
    on_progress: Callable[[dict[str, Any]], None],
) -> _ModeOptimisation:
    """Optimise one motion_type/mode over train/val/test datasets."""

    best_value = float("inf")
    best_params = _default_params_for_filter(space, adaptive_filter, objective_mode)
    best_repeat_idx = 0
    best_trial_idx = 0
    history: list[dict[str, Any]] = []
    best_so_far = float("inf")
    objective_sets = val_sets if data_split_mode == "split" else train_sets

    def _evaluate_params(params: ProtocolTrialParams) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
        train_metrics, train_rows = _evaluate_dataset_map(
            train_sets, scope, scheme, params, "train", trial_cache
        )
        val_metrics, val_rows = _evaluate_dataset_map(
            val_sets, scope, scheme, params, "val", trial_cache
        ) if val_sets else (_empty_metrics("val"), [])
        test_metrics, test_rows = _evaluate_dataset_map(
            test_sets, scope, scheme, params, "test", trial_cache
        )
        return train_metrics, val_metrics, test_metrics, [*train_rows, *val_rows, *test_rows]

    for repeat_idx in range(int(n_repeats)):
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
                    mode_key=f"{scope.value}__{scheme.value}__{adaptive_filter}",
                    repeat_idx=repeat_idx,
                    random_state=random_state,
                )
                train_metrics, val_metrics, test_metrics, _ = _evaluate_params(params)
                objective_metrics = val_metrics if data_split_mode == "split" else train_metrics
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
                    mode_key=f"{scope.value}__{scheme.value}__{adaptive_filter}",
                    repeat_idx=repeat_idx,
                    random_state=random_state,
                )
                train_metrics, val_metrics, test_metrics, _ = _evaluate_params(params)
                objective_metrics = val_metrics if data_split_mode == "split" else train_metrics
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

    train_metrics, val_metrics, test_metrics, per_group_rows = _evaluate_params(best_params)
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
    )


def _default_params_for_filter(
    space: ProtocolSearchSpace,
    adaptive_filter: str,
    objective_mode: str,
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
    return ProtocolTrialParams(**values)


def _decode_with_seed(
    space: ProtocolSearchSpace,
    idx_map: dict[str, int],
    *,
    adaptive_filter: str,
    objective_mode: str,
    mode_key: str,
    repeat_idx: int,
    random_state: int,
) -> ProtocolTrialParams:
    params = decode_protocol_search_space(
        space,
        idx_map,
        adaptive_filter=adaptive_filter,
        objective_mode=objective_mode,
        rff_seed=0,
    )
    if adaptive_filter == "rff_lms":
        seed_payload = {
            **{k: v for k, v in params.to_dict().items() if k != "rff_seed"},
            "mode_key": mode_key,
            "repeat_idx": int(repeat_idx),
            "random_state": int(random_state),
        }
        params = replace(params, rff_seed=_stable_int_hash(seed_payload))
    return params


def _evaluate_dataset_map(
    datasets: dict[str, ProtocolDataset],
    scope: TargetScope,
    scheme: CascadeScheme,
    params: ProtocolTrialParams,
    split_name: str,
    trial_cache: dict[tuple[Any, ...], ProtocolRunResult],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run one trial params on all datasets in one split and aggregate metrics."""

    runs: list[ProtocolRunResult] = []
    rows: list[dict[str, Any]] = []
    for group_id, dataset in datasets.items():
        cache_key = (
            group_id,
            int(params.Fs_Target),
            int(params.TW),
            scope.value,
            scheme.value,
            str(params.adaptive_filter),
            params.cache_key(),
        )
        run = trial_cache.get(cache_key)
        if run is None:
            run = run_protocol_trial(dataset, scheme, scope, params)
            trial_cache[cache_key] = run
        runs.append(run)
        rows.append(_per_group_row(group_id, split_name, run))
    return _aggregate_runs(runs, split_name), rows


def _aggregate_runs(runs: list[ProtocolRunResult], split_name: str) -> dict[str, Any]:
    """Aggregate window-level metrics across a split."""

    frames = [r.frame for r in runs if r.success and r.frame is not None and not r.frame.empty]
    failures = [r.reason for r in runs if not r.success]
    if not frames:
        return {
            "split": split_name,
            "success": False,
            "reason": "; ".join(failures) if failures else "no successful runs",
            "baseline_aae_bpm": float("nan"),
            "adaptive_aae_bpm": float("nan"),
            "baseline_acc_pct": float("nan"),
            "adaptive_acc_pct": float("nan"),
            "num_windows": 0,
        }
    frame = pd.concat(frames, ignore_index=True)
    mask = frame.get("is_filtered_segment", pd.Series(True, index=frame.index)).to_numpy(dtype=bool)
    baseline_err = frame.loc[mask, "baseline_abs_err_bpm"].to_numpy(dtype=float)
    adaptive_err = frame.loc[mask, "adaptive_abs_err_bpm"].to_numpy(dtype=float)
    return {
        "split": split_name,
        "success": len(failures) == 0,
        "reason": "; ".join(failures[:3]),
        "baseline_aae_bpm": _nanmean(baseline_err),
        "adaptive_aae_bpm": _nanmean(adaptive_err),
        "baseline_acc_pct": _accuracy_from_abs_err(baseline_err),
        "adaptive_acc_pct": _accuracy_from_abs_err(adaptive_err),
        "num_windows": int(np.isfinite(adaptive_err).sum()),
    }


def _per_group_row(group_id: str, split_name: str, run: ProtocolRunResult) -> dict[str, Any]:
    return {
        "split": split_name,
        "group_id": group_id,
        "target_scope": run.target_scope.value,
        "cascade_scheme": run.cascade_scheme.value,
        "adaptive_filter": run.adaptive_filter,
        "success": bool(run.success),
        "reason": run.reason,
        "baseline_aae_bpm": run.baseline_aae_bpm,
        "adaptive_aae_bpm": run.adaptive_aae_bpm,
        "baseline_acc_pct": run.baseline_acc_pct,
        "adaptive_acc_pct": run.adaptive_acc_pct,
    }


def _objective_value(metrics: dict[str, Any], objective_mode: str, penalty_value: float) -> float:
    if objective_mode == "accuracy":
        acc = float(metrics.get("adaptive_acc_pct", float("nan")))
        value = 100.0 - acc
    else:
        value = float(metrics.get("adaptive_aae_bpm", float("nan")))
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
) -> None:
    history.append(
        {
            "motion_type": motion_type,
            "target_scope": scope.value,
            "cascade_scheme": scheme.value,
            "adaptive_filter": adaptive_filter,
            "repeat_idx": int(repeat_idx),
            "trial_idx": int(trial_idx),
            "objective_value": float(objective_value),
            "aae_bpm": float(metrics.get("adaptive_aae_bpm", float("nan"))),
            "accuracy_pct": float(metrics.get("adaptive_acc_pct", float("nan"))),
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
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(motion_dir / "mode_summary_aae.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(motion_dir / "mode_summary_accuracy.csv", index=False, encoding="utf-8-sig")

    per_group = pd.DataFrame([row for r in results for row in r.per_group_rows])
    per_group.to_csv(motion_dir / "per_group_aae.csv", index=False, encoding="utf-8-sig")
    per_group.to_csv(motion_dir / "per_group_accuracy.csv", index=False, encoding="utf-8-sig")

    for adaptive_filter in _VALID_FILTERS:
        rows = [_best_param_row(r) for r in results if r.adaptive_filter == adaptive_filter]
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
            "best_repeat_idx": r.best_repeat_idx,
            "best_trial_idx": r.best_trial_idx,
            "n_trials": r.n_trials,
            "n_repeats": r.n_repeats,
            "best_params": r.best_params.to_dict(),
            "train_metrics": r.train_metrics,
            "val_metrics": r.val_metrics,
            "test_metrics": r.test_metrics,
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


def _summary_row(result: _ModeOptimisation) -> dict[str, Any]:
    row = {
        "motion_type": result.motion_type,
        "target_scope": result.target_scope.value,
        "cascade_scheme": result.cascade_scheme.value,
        "adaptive_filter": result.adaptive_filter,
        "objective_mode": result.objective_mode,
        "data_split_mode": result.data_split_mode,
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
        row[f"{prefix}_num_windows"] = metrics.get("num_windows")
    return row


def _best_param_row(result: _ModeOptimisation) -> dict[str, Any]:
    return {**_summary_row(result), **result.best_params.to_dict()}


def _best_param_columns() -> list[str]:
    """Return stable best-params CSV columns, including headers for empty files."""

    summary_cols = [
        "motion_type",
        "target_scope",
        "cascade_scheme",
        "adaptive_filter",
        "objective_mode",
        "data_split_mode",
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
        "train_num_windows",
        "val_aae_bpm",
        "val_accuracy_pct",
        "val_baseline_aae_bpm",
        "val_baseline_accuracy_pct",
        "val_num_windows",
        "test_aae_bpm",
        "test_accuracy_pct",
        "test_baseline_aae_bpm",
        "test_baseline_accuracy_pct",
        "test_num_windows",
    ]
    param_cols = [item.name for item in fields(ProtocolTrialParams) if item.name not in summary_cols]
    return [*summary_cols, *param_cols]


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
    """Write six final cross-motion-type summary CSV files."""

    header = [
        "motion_type",
        "target_scope",
        "cascade_scheme",
        "adaptive_filter",
        "metric_value",
        "objective_mode",
        "data_split_mode",
        "n_trials",
        "n_repeats",
        "success",
        "reason",
    ]
    paths: dict[str, Path] = {}
    for scope in [TargetScope.MOTION_ONLY, TargetScope.MOTION_AND_RECOVERY, TargetScope.MOTION_POST10]:
        for metric_name, column, filename in (
            ("aae", "adaptive_aae_bpm", f"{scope.value}_test_aae.csv"),
            ("accuracy", "adaptive_acc_pct", f"{scope.value}_test_accuracy.csv"),
        ):
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
                                "metric_value": result.test_metrics.get(column),
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
) -> dict[str, Path]:
    """Re-run one sample with one best-param row and draw HR curves.

    中文说明：这是 Notebook 手动重画单元格使用的函数。主训练流程不会调用它。
    图 A 只画训练段；图 B 画全局曲线，非训练段使用 Hamming + FFT、0.5-2 Hz 主频。
    """

    scope = TargetScope(target_scope)
    scheme = CascadeScheme(cascade_scheme)
    adaptive_filter = str(adaptive_filter)
    params = _params_from_best_csv(best_param_csv_path, scope, scheme, adaptive_filter)
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
    frame["manual_global_hr_bpm"] = combined
    global_abs_err = np.abs(frame["manual_global_hr_bpm"].to_numpy(dtype=float) - frame["ref_hr_bpm"].to_numpy(dtype=float))
    global_aae = _nanmean(global_abs_err)
    global_acc = _accuracy_from_abs_err(global_abs_err)
    _plot_hr_frame(
        plt,
        frame.rename(columns={"manual_global_hr_bpm": "adaptive_hr_bpm"}),
        global_path,
        title=f"{group_id} global HR ({scope.value} / {scheme.value} / {adaptive_filter})",
        adaptive_label=f"adaptive + FFT HR AAE={global_aae:.2f} bpm, accuracy={global_acc:.1f}%",
    )
    return {"training_scope": train_path, "global": global_path}


def _params_from_best_csv(
    best_param_csv_path: str | Path,
    scope: TargetScope,
    scheme: CascadeScheme,
    adaptive_filter: str,
) -> ProtocolTrialParams:
    df = pd.read_csv(best_param_csv_path)
    mask = pd.Series(True, index=df.index)
    for column, value in (
        ("target_scope", scope.value),
        ("cascade_scheme", scheme.value),
        ("adaptive_filter", adaptive_filter),
    ):
        if column in df.columns:
            mask &= df[column].astype(str) == str(value)
    if mask.any():
        row = df.loc[mask].iloc[0]
    elif not df.empty:
        row = df.iloc[0]
    else:
        raise ValueError(f"best param CSV is empty: {best_param_csv_path}")

    values: dict[str, Any] = {}
    for item in fields(ProtocolTrialParams):
        if item.name not in row or pd.isna(row[item.name]):
            continue
        default = item.default
        value = row[item.name]
        if isinstance(default, bool):
            values[item.name] = bool(value)
        elif isinstance(default, int):
            values[item.name] = int(value)
        elif isinstance(default, float):
            values[item.name] = float(value)
        else:
            values[item.name] = str(value)
    values["adaptive_filter"] = adaptive_filter
    return ProtocolTrialParams(**values)


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
        "baseline_acc_pct": float("nan"),
        "adaptive_acc_pct": float("nan"),
        "num_windows": 0,
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
