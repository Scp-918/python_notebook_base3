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
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field, fields, replace
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

from ..params import (DEFAULT_REFERENCE_CONFIGS, RESULT_SCHEMA_VERSION, ReferenceConfig, MotionType, ProtocolParams, TargetScope,
                      resolve_core_config, core_preset_metadata, CORE_SEARCH_FIELDS,
                      CORE_EXPERIMENT_FIELDS, CORE_EVALUATION_FIELDS)
from ..preprocess.calibration import load_subject_calibration
from .penalty_policy import PenaltyPolicy, spectrum_peaks
from .batch_pairing import SamplePair, discover_sample_pairs_with_unpaired
from .alignment import _window_fft_hr
from .cascade_solver import (
    MetricArrays,
    ProtocolRunResult,
    _get_trial_base,
    aggregate_metric_arrays,
    clear_all_caches,
    clear_trial_heavy_caches,
    run_protocol_trial,
)
from .preprocess_protocol import (
    ProtocolDataset,
    _validate_ud_calculation_config,
    load_and_preprocess_protocol,
    resample_protocol_dataset,
    apply_ppg_input_transform,
)
from .protocol_outputs import SampleOutputPaths, plot_signal_figures, write_qc_tables
from .protocol_search_space import (
    ProtocolSearchSpace,
    ProtocolTrialParams,
    decode_protocol_search_space,
    default_protocol_search_space,
)
from .qc import QcResult, quality_filter_sample
from .segmentation import apply_motion_protocol, detect_source_imu_segments
from .spectral_utils import compute_power_spectrum

__all__ = [
    "BatchProtocolResult",
    "build_cross_motion_summary_table",
    "flatten_params_for_record",
    "build_output_run_name",
    "build_subject_output_dir",
    "plot_window_diagnostics_from_records",
    "protocol_params_from_record",
    "replay_best_record_hr_curves",
    "redraw_best_param_hr_curves",
    "redraw_subject_target_hr_curves",
    "run_batch_reference_compare",
    "run_batch_adaptive_protocol",
    "safe_prepare_output_dir",
]

if optuna is not None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)

_VALID_FILTERS = ("lms", "volterra", "rff_lms", "klms")
_VALID_OBJECTIVES = ("aae", "accuracy", "posthoc_aae")
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
    reference_config: ReferenceConfig
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
    training_elapsed_s: float = 0.0
    ud_calculation_mode: str = "smoothed"
    ud_smoothing_window_s: float = 0.1
    ud_smoothing_window_samples: int = 10
    fold_id: int | None = None
    heldout_group_id: str = ""
    train_group_ids: list[str] = field(default_factory=list)
    test_group_id: str = ""
    history_path: str = ""
    per_group_path: str = ""
    mode_manifest_path: str = ""
    fold_results: list["_ModeOptimisation"] = field(default_factory=list, repr=False)
    metric_arrays_by_split: dict[str, MetricArrays] = field(default_factory=dict, repr=False)
    acc_compare_metrics: dict[str, Any] = field(default_factory=dict)
    acc_compare_arrays: MetricArrays = field(default_factory=dict, repr=False)
    result_level: str = "fold"
    aggregation: str = ""
    params_semantics: str = ""
    representative_fold_id: int | None = None
    representative_heldout_group_id: str = ""
    run_fingerprint: str = ""

    @property
    def mode_key(self) -> str:
        return _mode_key_for(
            self.motion_type,
            self.target_scope,
            self.reference_config,
            self.adaptive_filter,
        )


def build_output_run_name(
    target_scopes: list[str | TargetScope],
    reference_group_orders: list[tuple[str, ...] | ReferenceConfig],
    adaptive_filters: list[str],
    objective_mode: str,
    data_split_mode: str,
    prewarm_s: float = 0.0,
    postprocess_method: str = "fft",
) -> str:
    """Build the run_name required by the Notebook output layout."""

    scopes = "-".join(TargetScope(x).value for x in target_scopes)
    schemes = "-".join(ReferenceConfig(x).reference_order_key for x in reference_group_orders)
    filters = "-".join(str(x) for x in adaptive_filters)
    return (
        f"{scopes}__{schemes}__{filters}__{objective_mode}__{data_split_mode}"
        f"__{_prewarm_run_label(prewarm_s)}__{_postprocess_run_label(postprocess_method)}"
    )


def build_subject_output_dir(
    *,
    subject_dir: str | Path,
    target_scopes: list[str | TargetScope],
    reference_group_orders: list[tuple[str, ...] | ReferenceConfig],
    adaptive_filters: list[str],
    objective_mode: str,
    data_split_mode: str,
    prewarm_s: float = 0.0,
    postprocess_method: str = "fft",
) -> Path:
    """Return ``subject_dir.parent/outputs/{subject}__{run_signature}``."""

    subject = Path(subject_dir).resolve()
    run_signature = build_output_run_name(
        target_scopes,
        reference_group_orders,
        adaptive_filters,
        objective_mode,
        data_split_mode,
        prewarm_s=prewarm_s,
        postprocess_method=postprocess_method,
    )
    return subject.parent / "outputs" / f"{subject.name}__{run_signature}"


def _prewarm_run_label(value: float) -> str:
    """Return a compact output-folder label for fixed prewarm_s seconds."""

    value_f = float(value)
    if value_f.is_integer():
        return f"prewarm_s{int(value_f)}s"
    text = f"{value_f:g}".replace(".", "p").replace("-", "m")
    return f"prewarm_s{text}s"


def _postprocess_run_label(value: str) -> str:
    """Return a path-safe output-folder label for HR postprocess method."""

    text = str(value or "fft").strip().lower()
    safe = "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")
    return f"hr_{safe or 'fft'}"


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
    subject_dir: str | Path,
    output_root: str | Path | None = None,
    csv_out_dir: str | Path | None = None,
    report_out_dir: str | Path | None = None,
    fig_out_dir: str | Path | None = None,
    max_iterations: int = 200,
    num_repeats: int = 1,
    random_state: int = 42,
    num_seed_points: int = 10,
    fs_origin: int = 100,
    ud_calculation_mode: str = "smoothed",
    ud_smoothing_window_s: float = 0.1,
    penalty_value: float = 999.0,
    parallel_repeats: int = 1,
    n_jobs: int | None = 1,
    trial_cache_max_entries: int = _DEFAULT_TRIAL_CACHE_MAX_ENTRIES,
    save_stage_json: bool = False,
    debug_mode: bool = False,
    search_space: ProtocolSearchSpace | None = None,
    trial_param_overrides: dict[str, Any] | None = None,
    target_scopes: list[TargetScope | str] | None = None,
    reference_group_orders: list[ReferenceConfig | tuple[str, ...]] | None = None,
    adaptive_filters: list[str] | None = None,
    objective_mode: str = "posthoc_aae",
    data_split_mode: str = "all_train",
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
    ud_calculation_mode, ud_smoothing_window_samples = _validate_ud_calculation_config(
        ud_calculation_mode,
        ud_smoothing_window_s,
        int(fs_origin),
    )

    scopes = [TargetScope(x) for x in (target_scopes or [TargetScope.MOTION_POST10])]
    schemes = [ReferenceConfig(x) for x in (reference_group_orders or list(DEFAULT_REFERENCE_CONFIGS))]
    filters = [str(x) for x in (adaptive_filters or ["lms"])]
    unsupported = [x for x in filters if x not in _VALID_FILTERS]
    if unsupported:
        raise ValueError(f"Unsupported adaptive_filters: {unsupported}")

    trial_overrides = _normalise_trial_param_overrides(trial_param_overrides)
    input_path = Path(subject_dir).resolve()
    if output_root is None:
        if csv_out_dir is not None:
            root_out = Path(csv_out_dir).resolve().parent
        else:
            root_out = build_subject_output_dir(
                subject_dir=input_path,
                target_scopes=scopes,
                reference_group_orders=schemes,
                adaptive_filters=filters,
                objective_mode=objective_mode,
                data_split_mode=data_split_mode,
                prewarm_s=float(trial_overrides.get("prewarm_s", 0.0)),
                postprocess_method=str(trial_overrides.get("postprocess_method", "fft")),
            )
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

    calibration = load_subject_calibration(input_path, on_log=_log)
    _log(
        "UD 计算配置: "
        f"mode={ud_calculation_mode}, smoothing=centered_moving_median, "
        f"window={float(ud_smoothing_window_s):g}s/{ud_smoothing_window_samples} samples"
    )
    discovery = discover_sample_pairs_with_unpaired(input_path)
    _log(f"配对总数: {len(discovery.pairs)}")
    _log(f"未配对文件数量: {len(discovery.unpaired)}")

    good_qc: list[QcResult] = []
    bad_qc: list[QcResult] = []
    accepted_pairs: list[SamplePair] = []
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
        # QC is metadata, not a structural admission filter. Strict parsing above
        # decides whether a pair is loadable; bad samples continue into calculation.
        accepted_pairs.append(pair)
        if qc.is_good:
            good_qc.append(qc)
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
    for idx, pair in enumerate(accepted_pairs, start=1):
        _progress({"stage": "preprocess", "current": idx, "total": len(accepted_pairs), "sample": pair.stem})
        try:
            dataset = load_and_preprocess_protocol(
                pair.sensor_csv,
                pair.ref_csv,
                fs_origin=fs_origin,
                calibration=calibration,
                ud_calculation_mode=ud_calculation_mode,
                ud_smoothing_window_s=ud_smoothing_window_s,
            )
            datasets[pair.motion_id] = dataset
            pair_by_group[pair.motion_id] = pair
            seg_for_plot = apply_motion_protocol(detect_source_imu_segments(dataset.source_imu), trial_overrides.get("motion_policy", "single_bout"))
            signal_figures[pair.motion_id] = plot_signal_figures(
                sensor_csv=pair.sensor_csv,
                dataset=dataset,
                segment_info=seg_for_plot,
                output_dir=signal_dir,
                group_id=pair.motion_id,
                motion_type=pair.motion_type,
                fs_origin=fs_origin,
                calibration=calibration,
                ud_calculation_mode=ud_calculation_mode,
                ud_smoothing_window_s=ud_smoothing_window_s,
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
    for pair in accepted_pairs:
        if pair.motion_id in datasets:
            grouped_pairs.setdefault(pair.motion_type, []).append(pair)
    motion_types = _motion_types_in_canonical_order(grouped_pairs)
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

        restored_results = _load_completed_mode_results(motion_dir)
        restored_by_key = {result.mode_key: result for result in restored_results}
        input_signatures = {
            pair.motion_id: f"{_file_content_signature(pair.sensor_csv)}:{_file_content_signature(pair.ref_csv)}"
            for pair in members
        }
        calibration_signature = str(getattr(calibration, "source_sha256", ""))
        search_space_signature = hashlib.sha256(
            json.dumps(_jsonify(asdict(space)), ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
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
                            "reference_order_key": scheme.reference_order_key,
                            "adaptive_filter": adaptive_filter,
                        }
                    )
                    mode_key = _mode_key_for(motion_type, scope, scheme, adaptive_filter)
                    expected_fingerprint = _build_mode_fingerprint(
                        motion_type=motion_type,
                        scope=scope,
                        scheme=scheme,
                        adaptive_filter=adaptive_filter,
                        objective_mode=objective_mode,
                        data_split_mode=data_split_mode,
                        input_signatures=input_signatures,
                        calibration_signature=calibration_signature,
                        search_space_signature=search_space_signature,
                        n_trials=budget["n_trials"],
                        n_repeats=budget["n_repeats"],
                        fixed_config={
                            "fs_origin": int(fs_origin),
                            "random_state": int(random_state),
                            "num_seed_points": int(num_seed_points),
                            "penalty_value": float(penalty_value),
                            "delay_estimation_mode": delay_estimation_mode,
                            "trial_param_overrides": trial_overrides,
                            **_ud_preprocessing_fingerprint_fields(
                                scheme,
                                mode=ud_calculation_mode,
                                window_s=ud_smoothing_window_s,
                                window_samples=ud_smoothing_window_samples,
                            ),
                        },
                    )
                    resumed = restored_by_key.get(mode_key)
                    if resumed is not None and resumed.run_fingerprint == expected_fingerprint:
                        mode_results.append(resumed)
                        _progress(
                            _mode_completion_progress_payload(
                                resumed,
                                mode_idx=mode_counter,
                                mode_total=total_modes,
                                resumed=True,
                            )
                        )
                        gc.collect()
                        continue
                    mode_started_at = time.perf_counter()
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
                    result.training_elapsed_s = max(0.0, time.perf_counter() - mode_started_at)
                    result.ud_calculation_mode = ud_calculation_mode
                    result.ud_smoothing_window_s = float(ud_smoothing_window_s)
                    result.ud_smoothing_window_samples = int(ud_smoothing_window_samples)
                    result.run_fingerprint = expected_fingerprint
                    mode_results.append(result)
                    _finalize_completed_mode(
                        motion_dir,
                        motion_type,
                        mode_results,
                        result,
                        write_best_params_all=False,
                    )
                    _progress(
                        _mode_completion_progress_payload(
                            result,
                            mode_idx=mode_counter,
                            mode_total=total_modes,
                            resumed=False,
                        )
                    )
                    gc.collect()
        all_mode_results[motion_type] = mode_results
        _write_motion_type_outputs(motion_dir, motion_type, mode_results, write_best_params_all=True)
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


def _motion_types_in_canonical_order(grouped: dict[str, Any]) -> list[str]:
    """Return present motion types in the one shared protocol order."""

    return [motion.value for motion in MotionType if motion.value in grouped]


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
    scheme: ReferenceConfig,
    cfg: ProtocolParams,
) -> dict[str, int]:
    """Return per-cascade n_trials/n_repeats with config fallback."""

    item = (budgets or {}).get(scheme.reference_order_key, {})
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
    scheme: ReferenceConfig,
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
            mode_seed = _derive_mode_seed(
                random_state=random_state,
                motion_type=motion_type,
                scope=scope,
                scheme=scheme,
                adaptive_filter=adaptive_filter,
                repeat_idx=repeat_idx,
            )
            sampler = TPESampler(
                seed=mode_seed,
                n_startup_trials=min(int(num_seed_points), int(n_trials)),
            )
            study = optuna.create_study(direction="minimize", sampler=sampler)

            def _objective(trial: optuna.trial.Trial) -> float:
                nonlocal best_value, best_params, best_trial_idx, best_so_far

                idx_map = {
                    name: trial.suggest_int(name, 0, len(space.options(name, adaptive_filter=adaptive_filter)) - 1)
                    for name in space.names_for_filter(adaptive_filter)
                }
                params = _decode_with_seed(
                    space,
                    idx_map,
                    adaptive_filter=adaptive_filter,
                    objective_mode=objective_mode,
                    delay_estimation_mode=delay_estimation_mode,
                    mode_key=_mode_key_for(motion_type, scope, scheme, adaptive_filter),
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
            rng = np.random.default_rng(
                _derive_mode_seed(
                    random_state=random_state,
                    motion_type=motion_type,
                    scope=scope,
                    scheme=scheme,
                    adaptive_filter=adaptive_filter,
                    repeat_idx=repeat_idx,
                )
            )
            for trial_idx in range(int(n_trials)):
                idx_map = {
                    name: int(rng.integers(0, len(space.options(name, adaptive_filter=adaptive_filter))))
                    for name in space.names_for_filter(adaptive_filter)
                }
                params = _decode_with_seed(
                    space,
                    idx_map,
                    adaptive_filter=adaptive_filter,
                    objective_mode=objective_mode,
                    delay_estimation_mode=delay_estimation_mode,
                    mode_key=_mode_key_for(motion_type, scope, scheme, adaptive_filter),
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
    scheme: ReferenceConfig,
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
                "reference_order_key": scheme.reference_order_key,
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
                        "reference_order_key": scheme.reference_order_key,
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
            mode_seed = _derive_mode_seed(
                random_state=random_state,
                motion_type=motion_type,
                scope=scope,
                scheme=scheme,
                adaptive_filter=adaptive_filter,
                repeat_idx=repeat_idx,
            )
            sampler = TPESampler(
                seed=mode_seed,
                n_startup_trials=min(int(cfg.num_seed_points), int(n_trials)),
            )
            study = optuna.create_study(direction="minimize", sampler=sampler)

            def _objective(trial: optuna.trial.Trial) -> float:
                nonlocal best_value, best_params, best_repeat_idx, best_trial_idx, best_so_far

                idx_map = {
                    name: trial.suggest_int(name, 0, len(space.options(name, adaptive_filter=adaptive_filter)) - 1)
                    for name in space.names_for_filter(adaptive_filter)
                }
                params = _decode_with_seed(
                    space,
                    idx_map,
                    adaptive_filter=adaptive_filter,
                    objective_mode=objective_mode,
                    delay_estimation_mode=delay_estimation_mode,
                    mode_key=_mode_key_for(motion_type, scope, scheme, adaptive_filter),
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
            rng = np.random.default_rng(
                _derive_mode_seed(
                    random_state=random_state,
                    motion_type=motion_type,
                    scope=scope,
                    scheme=scheme,
                    adaptive_filter=adaptive_filter,
                    repeat_idx=repeat_idx,
                )
            )
            for trial_idx in range(int(n_trials)):
                idx_map = {
                    name: int(rng.integers(0, len(space.options(name, adaptive_filter=adaptive_filter))))
                    for name in space.names_for_filter(adaptive_filter)
                }
                params = _decode_with_seed(
                    space,
                    idx_map,
                    adaptive_filter=adaptive_filter,
                    objective_mode=objective_mode,
                    delay_estimation_mode=delay_estimation_mode,
                    mode_key=_mode_key_for(motion_type, scope, scheme, adaptive_filter),
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
    result = _ModeOptimisation(
        motion_type=motion_type,
        target_scope=scope,
        reference_config=scheme,
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
    _attach_acc_compare(
        result,
        test_sets=test_sets,
        trial_cache=trial_cache,
        trial_cache_max_entries=trial_cache_max_entries,
    )
    return result


def _normalise_trial_param_overrides(overrides: dict[str, Any] | None) -> dict[str, Any]:
    """Validate fixed ProtocolTrialParams overrides that are not sampled by Optuna.

    中文说明：Notebook 可以把静息段 HR 提取参数放在第一个初始化代码块里统一维护。
    这些参数会覆盖每个 trial 的默认值，但不会进入 Optuna 搜索空间；未知字段直接报错，
    避免因为拼写错误导致配置看似生效、实际被忽略。
    """

    if not overrides:
        return {}
    allowed = CORE_SEARCH_FIELDS | CORE_EXPERIMENT_FIELDS | CORE_EVALUATION_FIELDS
    rejected = sorted(set(overrides) - allowed)
    if rejected:
        raise ValueError(f"Frozen or unknown core_enhanced_v1 overrides: {rejected}")
    resolve_core_config(overrides)
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
        value = space.options(name, adaptive_filter=adaptive_filter)[0]
        values[name] = value
    values["adaptive_filter"] = adaptive_filter
    values["objective_mode"] = objective_mode
    values["delay_estimation_mode"] = str(delay_estimation_mode)
    values.update(_normalise_trial_param_overrides(trial_param_overrides))
    return resolve_core_config(values)


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
    scheme: ReferenceConfig,
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
    result = _ModeOptimisation(
        motion_type=motion_type,
        target_scope=scope,
        reference_config=scheme,
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
        acc_compare_metrics=_empty_acc_compare_fields("failed", reason),
        result_level="aggregate" if data_split_mode == "leave_one_group_out" else "single_split",
        aggregation="logo_window_concat" if data_split_mode == "leave_one_group_out" else "",
        params_semantics=(
            "representative_fold_best_params_not_global"
            if data_split_mode == "leave_one_group_out"
            else "best_params_for_this_split"
        ),
    )
    return result


def _aggregate_logo_fold_results(
    *,
    motion_type: str,
    scope: TargetScope,
    scheme: ReferenceConfig,
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
            reference_config=scheme,
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
            acc_compare_metrics=_empty_acc_compare_fields("failed", reason),
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

    result = _ModeOptimisation(
        motion_type=motion_type,
        target_scope=scope,
        reference_config=scheme,
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
    if scheme == ReferenceConfig(('ACC',)):
        result.acc_compare_metrics = _acc_compare_fields(test_metrics, status="same_as_original", reason="")
        result.acc_compare_arrays = test_arrays
    else:
        acc_arrays = _concat_metric_arrays([r.acc_compare_arrays for r in fold_results])
        failures = [
            str(r.acc_compare_metrics.get("acc_compare_reason", ""))
            for r in fold_results
            if str(r.acc_compare_metrics.get("acc_compare_status", "")) not in {"ok", "same_as_original"}
        ]
        acc_metrics = _metrics_from_arrays_with_status(acc_arrays, "test", failures)
        result.acc_compare_metrics = _acc_compare_fields(
            acc_metrics,
            status="ok" if bool(acc_metrics.get("success", False)) else "failed",
            reason=str(acc_metrics.get("reason", "")),
        )
        result.acc_compare_arrays = acc_arrays
    return result


def _evaluate_dataset_map(
    datasets: dict[str, ProtocolDataset],
    scope: TargetScope,
    scheme: ReferenceConfig,
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
            int(params.hr_window_s),
            scope.value,
            scheme.reference_order_key,
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


def _attach_acc_compare(
    result: _ModeOptimisation,
    *,
    test_sets: dict[str, ProtocolDataset],
    trial_cache: OrderedDict[tuple[Any, ...], ProtocolRunResult] | dict[tuple[Any, ...], ProtocolRunResult],
    trial_cache_max_entries: int,
) -> None:
    """Attach same-best-params ACC diagnostic test metrics to one completed mode."""

    if result.reference_config == ReferenceConfig(('ACC',)):
        result.acc_compare_metrics = _acc_compare_fields(
            result.test_metrics,
            status="same_as_original",
            reason="",
        )
        result.acc_compare_arrays = result.metric_arrays_by_split.get("test", {})
        return
    try:
        metrics, _, arrays = _evaluate_dataset_map(
            test_sets,
            result.target_scope,
            ReferenceConfig(('ACC',)),
            result.best_params,
            "test",
            trial_cache,
            eval_mode="light",
            trial_cache_max_entries=trial_cache_max_entries,
        )
        status = "ok" if bool(metrics.get("success", False)) else "failed"
        result.acc_compare_metrics = _acc_compare_fields(
            metrics,
            status=status,
            reason=str(metrics.get("reason", "")),
        )
        result.acc_compare_arrays = arrays
    except Exception as exc:
        result.acc_compare_metrics = _empty_acc_compare_fields("failed", str(exc))
        result.acc_compare_arrays = {}


def _acc_compare_fields(metrics: dict[str, Any], *, status: str, reason: str) -> dict[str, Any]:
    return {
        "acc_compare_aae_bpm": metrics.get("final_aae_bpm", metrics.get("adaptive_aae_bpm", float("nan"))),
        "acc_compare_accuracy_pct": metrics.get("final_acc_pct", metrics.get("adaptive_acc_pct", float("nan"))),
        "acc_compare_posthoc_aae_bpm": metrics.get("posthoc_final_aae_bpm", float("nan")),
        "acc_compare_posthoc_accuracy_pct": metrics.get("posthoc_final_acc_pct", float("nan")),
        "acc_compare_num_windows": int(metrics.get("num_windows", 0) or 0),
        "acc_compare_status": str(status),
        "acc_compare_reason": str(reason),
    }


def _empty_acc_compare_fields(status: str = "", reason: str = "") -> dict[str, Any]:
    return {
        "acc_compare_aae_bpm": float("nan"),
        "acc_compare_accuracy_pct": float("nan"),
        "acc_compare_posthoc_aae_bpm": float("nan"),
        "acc_compare_posthoc_accuracy_pct": float("nan"),
        "acc_compare_num_windows": 0,
        "acc_compare_status": str(status),
        "acc_compare_reason": str(reason),
    }


def _mode_completion_progress_payload(
    result: _ModeOptimisation,
    *,
    mode_idx: int,
    mode_total: int,
    resumed: bool,
) -> dict[str, Any]:
    """Build the single progress event emitted after one motion/mode is ready."""

    compare = _acc_compare_metrics_for(result)
    return {
        "stage": "optimization_mode_completed",
        "mode_idx": int(mode_idx),
        "mode_current": int(mode_idx),
        "mode_total": int(mode_total),
        "motion_type": result.motion_type,
        "target_scope": result.target_scope.name,
        "target_scope_value": result.target_scope.value,
        "reference_order_key": result.reference_config.reference_order_key,
        "adaptive_filter": result.adaptive_filter,
        "success": bool(result.success),
        "reason": str(result.reason),
        "resumed": bool(resumed),
        "training_elapsed_s": float(result.training_elapsed_s),
        "posthoc_final_aae_bpm": result.test_metrics.get("posthoc_final_aae_bpm", float("nan")),
        "posthoc_final_acc_pct": result.test_metrics.get("posthoc_final_acc_pct", float("nan")),
        "acc_compare_posthoc_aae_bpm": compare.get("acc_compare_posthoc_aae_bpm", float("nan")),
        "acc_compare_posthoc_accuracy_pct": compare.get(
            "acc_compare_posthoc_accuracy_pct", float("nan")
        ),
        "acc_compare_status": compare.get("acc_compare_status", ""),
        "acc_compare_reason": compare.get("acc_compare_reason", ""),
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
        "reference_order_key": run.reference_config.reference_order_key,
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
        "tracking_metadata_json": json.dumps(_jsonify(run.metadata), ensure_ascii=False, sort_keys=True),
    }


def _objective_value(metrics: dict[str, Any], objective_mode: str, penalty_value: float) -> float:
    if objective_mode == "accuracy":
        acc = float(metrics.get("final_acc_pct", metrics.get("adaptive_acc_pct", float("nan"))))
        value = 100.0 - acc
    elif objective_mode == "posthoc_aae":
        posthoc = metrics.get("posthoc_final_aae_bpm", float("nan"))
        if np.isfinite(float(posthoc)):
            value = float(posthoc)
        else:
            value = float(metrics.get("final_aae_bpm", metrics.get("adaptive_aae_bpm", float("nan"))))
    else:
        value = float(metrics.get("final_aae_bpm", metrics.get("adaptive_aae_bpm", float("nan"))))
    return value if np.isfinite(value) else float(penalty_value)


def flatten_params_for_record(params: ProtocolTrialParams, prefix: str = "param_") -> dict[str, Any]:
    """Flatten ``ProtocolTrialParams`` into stable CSV record columns.

    中文说明：新记录标准入口使用 ``param_*`` 分列；tuple/list/dict 写成 JSON 字符串，
    None 写成空字符串，读取时由 :func:`protocol_params_from_record` 还原。
    """

    out: dict[str, Any] = {"result_schema_version": RESULT_SCHEMA_VERSION,
                           "preset_metadata_json": json.dumps(core_preset_metadata(params), sort_keys=True)}
    for name, value in params.to_dict().items():
        key = f"{prefix}{name}"
        if value is None:
            out[key] = ""
        elif isinstance(value, (tuple, list, dict)):
            out[key] = json.dumps(_jsonify(value), ensure_ascii=False, sort_keys=True)
        else:
            out[key] = value
    return out


def protocol_params_from_record(row: Any) -> ProtocolTrialParams:
    """Restore complete current-schema trial parameters, preferring param columns."""
    record = _record_to_dict(row)
    _require_result_schema(record, "parameter record")
    values = _parse_stage6_params_json(record.get("best_params_json", ""))
    for item in fields(ProtocolTrialParams):
        column = f"param_{item.name}"
        if column in record:
            if _record_value_missing(record[column]):
                if item.default is None:
                    values[item.name] = None
                else:
                    raise ValueError(f"Missing parameter value: {column}")
            else:
                values[item.name] = _coerce_protocol_param_value(record[column], item)
    missing = _protocol_param_names() - values.keys()
    unknown = values.keys() - _protocol_param_names()
    if missing or unknown:
        raise ValueError(f"Invalid parameter record fields: missing={sorted(missing)}, unknown={sorted(unknown)}")
    values = {item.name: _coerce_protocol_param_value(values[item.name], item) for item in fields(ProtocolTrialParams)}
    return resolve_core_config(values)


def _require_result_schema(payload: dict[str, Any], source: str) -> None:
    """Reject artifacts from any other result schema."""
    version = payload.get("result_schema_version")
    if isinstance(version, bool) or version != RESULT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported result schema in {source}: {version!r}; expected {RESULT_SCHEMA_VERSION}")


def _require_result_table_schema(frame: pd.DataFrame, source: str) -> None:
    if "result_schema_version" not in frame.columns:
        raise ValueError(f"Missing result schema in {source}")
    for version in frame["result_schema_version"]:
        _require_result_schema({"result_schema_version": version}, source)


def _record_to_dict(row: Any) -> dict[str, Any]:
    """Convert pandas/dict-like records to a plain mapping."""

    if isinstance(row, pd.Series):
        return row.to_dict()
    if isinstance(row, dict):
        return dict(row)
    if hasattr(row, "items"):
        return dict(row.items())
    return {}


def _protocol_param_names() -> set[str]:
    return {item.name for item in fields(ProtocolTrialParams)}


def _record_value_missing(value: Any) -> bool:
    """Return whether a CSV cell should be treated as missing."""

    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _mode_artifact_dir(motion_dir: Path, result: _ModeOptimisation) -> Path:
    return motion_dir / "_modes" / result.mode_key


def _mode_key_for(
    motion_type: str,
    scope: TargetScope,
    scheme: ReferenceConfig,
    adaptive_filter: str,
) -> str:
    """Return a globally unique optimisation/checkpoint identity."""

    return f"{motion_type}__{scope.value}__{scheme.reference_order_key}__{adaptive_filter}"


def _derive_mode_seed(
    *,
    random_state: int,
    motion_type: str,
    scope: TargetScope,
    scheme: ReferenceConfig,
    adaptive_filter: str,
    repeat_idx: int,
) -> int:
    """Derive a reproducible sampler/RFF namespace including motion type."""

    return _stable_int_hash(
        {
            "random_state": int(random_state),
            "motion_type": str(motion_type),
            "target_scope": scope.value,
            "reference_order_key": scheme.reference_order_key,
            "adaptive_filter": str(adaptive_filter),
            "repeat_idx": int(repeat_idx),
        }
    )


def _build_mode_fingerprint(
    *,
    motion_type: str,
    scope: TargetScope,
    scheme: ReferenceConfig,
    adaptive_filter: str,
    objective_mode: str,
    data_split_mode: str,
    input_signatures: dict[str, str],
    calibration_signature: str,
    search_space_signature: str,
    n_trials: int,
    n_repeats: int,
    fixed_config: dict[str, Any],
) -> str:
    """Hash every input that determines whether a mode checkpoint is reusable."""

    payload = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "preset": core_preset_metadata(resolve_core_config()),
        "motion_type": str(motion_type),
        "target_scope": scope.value,
        "reference_order_key": scheme.reference_order_key,
        "adaptive_filter": str(adaptive_filter),
        "objective_mode": str(objective_mode),
        "data_split_mode": str(data_split_mode),
        "input_signatures": dict(sorted(input_signatures.items())),
        "calibration_signature": str(calibration_signature),
        "search_space_signature": str(search_space_signature),
        "n_trials": int(n_trials),
        "n_repeats": int(n_repeats),
        "fixed_config": fixed_config,
    }
    text = json.dumps(_jsonify(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _ud_preprocessing_fingerprint_fields(
    scheme: ReferenceConfig,
    *,
    mode: str,
    window_s: float,
    window_samples: int,
) -> dict[str, Any]:
    """Invalidate only modes whose numerical inputs changed with raw-formula UD."""

    if "UD" in scheme.reference_groups_order:
        return {
            "ud_preprocessing": "calibrated_formula_optional_voltage_smoothing_no_bandpass_v2",
            "ud_calculation_mode": str(mode),
            "ud_smoothing_method": "centered_moving_median",
            "ud_smoothing_window_s": float(window_s),
            "ud_smoothing_window_samples": int(window_samples),
        }
    return {}


def _mode_history_path(motion_dir: Path, result: _ModeOptimisation) -> Path:
    return _mode_artifact_dir(motion_dir, result) / "history.csv"


def _mode_per_group_path(motion_dir: Path, result: _ModeOptimisation) -> Path:
    return _mode_artifact_dir(motion_dir, result) / "per_group.csv"


def _fold_history_path(motion_dir: Path, result: _ModeOptimisation, fold: _ModeOptimisation) -> Path:
    fold_id = 0 if fold.fold_id is None else int(fold.fold_id)
    return _mode_artifact_dir(motion_dir, result) / f"fold_{fold_id}_history.csv"


def _fold_per_group_path(motion_dir: Path, result: _ModeOptimisation, fold: _ModeOptimisation) -> Path:
    fold_id = 0 if fold.fold_id is None else int(fold.fold_id)
    return _mode_artifact_dir(motion_dir, result) / f"fold_{fold_id}_per_group.csv"


def _mode_manifest_payload(result: _ModeOptimisation) -> dict[str, Any]:
    return {
        **core_preset_metadata(result.best_params),
        "reference_groups_order": result.reference_config.reference_groups_order,
        "reference_sequence": result.reference_config.reference_sequence,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "motion_type": result.motion_type,
        "target_scope": result.target_scope.value,
        "reference_order_key": result.reference_config.reference_order_key,
        "adaptive_filter": result.adaptive_filter,
        "objective_mode": result.objective_mode,
        "data_split_mode": result.data_split_mode,
        "best_params": result.best_params.to_dict(),
        "best_repeat_idx": result.best_repeat_idx,
        "best_trial_idx": result.best_trial_idx,
        "n_trials": result.n_trials,
        "n_repeats": result.n_repeats,
        "train_metrics": result.train_metrics,
        "val_metrics": result.val_metrics,
        "test_metrics": result.test_metrics,
        "success": result.success,
        "reason": result.reason,
        "training_elapsed_s": result.training_elapsed_s,
        "ud_calculation_mode": result.ud_calculation_mode,
        "ud_smoothing_window_s": result.ud_smoothing_window_s,
        "ud_smoothing_window_samples": result.ud_smoothing_window_samples,
        "fold_id": result.fold_id,
        "heldout_group_id": result.heldout_group_id,
        "train_group_ids": result.train_group_ids,
        "test_group_id": result.test_group_id,
        "history_path": result.history_path,
        "per_group_path": result.per_group_path,
        "mode_manifest_path": result.mode_manifest_path,
        "acc_compare_metrics": result.acc_compare_metrics,
        "result_level": result.result_level,
        "aggregation": result.aggregation,
        "params_semantics": result.params_semantics,
        "representative_fold_id": result.representative_fold_id,
        "representative_heldout_group_id": result.representative_heldout_group_id,
        "run_fingerprint": result.run_fingerprint,
        "fold_results": [_mode_manifest_payload(fold) for fold in result.fold_results],
    }


def _mode_result_from_manifest_payload(payload: dict[str, Any]) -> _ModeOptimisation:
    _require_result_schema(payload, "mode manifest")
    params = protocol_params_from_record({
        "result_schema_version": payload["result_schema_version"],
        "best_params_json": payload["best_params"],
    })
    return _ModeOptimisation(
        motion_type=str(payload["motion_type"]),
        target_scope=TargetScope(str(payload["target_scope"])),
        reference_config=ReferenceConfig.from_key(str(payload["reference_order_key"])),
        adaptive_filter=str(payload["adaptive_filter"]),
        objective_mode=str(payload["objective_mode"]),
        data_split_mode=str(payload["data_split_mode"]),
        best_params=params,
        best_repeat_idx=int(payload["best_repeat_idx"] or 0),
        best_trial_idx=int(payload["best_trial_idx"] or 0),
        n_trials=int(payload["n_trials"] or 0),
        n_repeats=int(payload["n_repeats"] or 0),
        train_metrics=dict(payload["train_metrics"]),
        val_metrics=dict(payload["val_metrics"]),
        test_metrics=dict(payload["test_metrics"]),
        per_group_rows=[],
        history=[],
        success=bool(payload["success"]),
        reason=str(payload["reason"]),
        training_elapsed_s=float(payload["training_elapsed_s"] or 0.0),
        ud_calculation_mode=str(payload["ud_calculation_mode"]),
        ud_smoothing_window_s=float(payload["ud_smoothing_window_s"] or 0.1),
        ud_smoothing_window_samples=int(payload["ud_smoothing_window_samples"] or 10),
        fold_id=None if payload["fold_id"] in ("", None) else int(payload["fold_id"]),
        heldout_group_id=str(payload["heldout_group_id"]),
        train_group_ids=[str(item) for item in payload["train_group_ids"]],
        test_group_id=str(payload["test_group_id"]),
        history_path=str(payload["history_path"]),
        per_group_path=str(payload["per_group_path"]),
        mode_manifest_path=str(payload["mode_manifest_path"]),
        acc_compare_metrics=dict(payload["acc_compare_metrics"]),
        fold_results=[
            _mode_result_from_manifest_payload(dict(item)) for item in payload["fold_results"]
        ],
        result_level=str(payload["result_level"]),
        aggregation=str(payload["aggregation"]),
        params_semantics=str(payload["params_semantics"]),
        representative_fold_id=(
            None if payload["representative_fold_id"] in ("", None)
            else int(payload["representative_fold_id"])
        ),
        representative_heldout_group_id=str(payload["representative_heldout_group_id"]),
        run_fingerprint=str(payload["run_fingerprint"]),
    )


def _write_mode_manifest(motion_dir: Path, result: _ModeOptimisation) -> Path:
    mode_dir = _mode_artifact_dir(motion_dir, result)
    mode_dir.mkdir(parents=True, exist_ok=True)
    history_path = _mode_history_path(motion_dir, result)
    result.history_path = str(history_path)
    manifest_path = mode_dir / "mode_result.json"
    result.mode_manifest_path = str(manifest_path)
    manifest_path.write_text(
        json.dumps(_jsonify(_mode_manifest_payload(result)), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest_path


def _read_motion_checkpoint(motion_dir: Path) -> dict[str, Any]:
    path = motion_dir / "_checkpoint.json"
    if not path.exists():
        return {"result_schema_version": RESULT_SCHEMA_VERSION, "motion_type": motion_dir.name, "modes": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid result schema checkpoint: {path}")
    _require_result_schema(payload, str(path))
    if not isinstance(payload["modes"], dict):
        raise ValueError(f"Invalid modes in checkpoint: {path}")
    return payload


def _write_motion_checkpoint(motion_dir: Path, checkpoint: dict[str, Any]) -> Path:
    _require_result_schema(checkpoint, "checkpoint")
    path = motion_dir / "_checkpoint.json"
    path.write_text(json.dumps(_jsonify(checkpoint), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _update_mode_checkpoint_entry(
    motion_dir: Path,
    result: _ModeOptimisation,
    *,
    checkpoint_status: str,
    manifest_path: Path | None = None,
    history_path: Path | None = None,
) -> Path:
    checkpoint = _read_motion_checkpoint(motion_dir)
    checkpoint["modes"][result.mode_key] = {
        "status": str(checkpoint_status),
        "run_fingerprint": result.run_fingerprint,
        "mode_manifest_path": str(manifest_path or result.mode_manifest_path or ""),
        "history_path": str(history_path or result.history_path or ""),
    }
    return _write_motion_checkpoint(motion_dir, checkpoint)


def _persist_mode_artifacts(
    motion_dir: Path,
    result: _ModeOptimisation,
    *,
    checkpoint_status: str,
) -> Path:
    mode_dir = _mode_artifact_dir(motion_dir, result)
    mode_dir.mkdir(parents=True, exist_ok=True)
    history_path = _mode_history_path(motion_dir, result)
    pd.DataFrame(result.history).to_csv(history_path, index=False, encoding="utf-8-sig")
    result.history_path = str(history_path)
    if result.per_group_rows:
        per_group_path = _mode_per_group_path(motion_dir, result)
        pd.DataFrame(result.per_group_rows).to_csv(per_group_path, index=False, encoding="utf-8-sig")
        result.per_group_path = str(per_group_path)
    for fold in result.fold_results:
        fold_history_path = _fold_history_path(motion_dir, result, fold)
        pd.DataFrame(fold.history).to_csv(fold_history_path, index=False, encoding="utf-8-sig")
        fold.history_path = str(fold_history_path)
        fold_per_group_path = _fold_per_group_path(motion_dir, result, fold)
        pd.DataFrame(fold.per_group_rows).to_csv(fold_per_group_path, index=False, encoding="utf-8-sig")
        fold.per_group_path = str(fold_per_group_path)
    manifest_path = _write_mode_manifest(motion_dir, result)
    _update_mode_checkpoint_entry(
        motion_dir,
        result,
        checkpoint_status=checkpoint_status,
        manifest_path=manifest_path,
        history_path=history_path,
    )
    return manifest_path


def _load_completed_mode_results(motion_dir: Path) -> list[_ModeOptimisation]:
    checkpoint = _read_motion_checkpoint(motion_dir)
    restored: list[_ModeOptimisation] = []
    for mode_key in sorted(checkpoint.get("modes", {})):
        item = checkpoint["modes"].get(mode_key, {})
        if not isinstance(item, dict) or str(item.get("status", "")) != "done":
            continue
        manifest_text = str(item.get("mode_manifest_path", ""))
        if not manifest_text:
            continue
        manifest_path = Path(manifest_text)
        if not manifest_path.exists():
            continue
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        restored.append(_mode_result_from_manifest_payload(payload))
    return restored


def _coerce_protocol_param_value(value: Any, item: Any) -> Any:
    """Coerce one record value according to the ProtocolTrialParams default."""

    default = item.default
    if _record_value_missing(value):
        if default is None:
            return None
        raise ValueError(f"Missing parameter value: {item.name}")
    if default is None:
        text = str(value).strip() if isinstance(value, str) else value
        if isinstance(text, str) and text.lower() in {"none", "null", "nan"}:
            return None
        try:
            return float(text)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid optional numeric parameter: {item.name}={value!r}") from exc
    if isinstance(default, bool):
        return _parse_bool_param(value)
    if isinstance(default, int) and not isinstance(default, bool):
        return int(float(value))
    if isinstance(default, float):
        return float(value)
    if isinstance(default, tuple):
        return _parse_tuple_param(value, default)
    return str(value)


def _append_history(
    history: list[dict[str, Any]],
    motion_type: str,
    scope: TargetScope,
    scheme: ReferenceConfig,
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
    row = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "fold_id": "" if fold_id is None else int(fold_id),
        "heldout_group_id": str(heldout_group_id),
        "motion_type": motion_type,
        "target_scope": scope.value,
        "reference_order_key": scheme.reference_order_key,
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
        "params": json.dumps(_jsonify(params.to_dict()), ensure_ascii=False, sort_keys=True),
    }
    row.update(flatten_params_for_record(params))
    history.append(row)


def _load_history_rows_from_path(path_text: str) -> list[dict[str, Any]]:
    if not path_text:
        return []
    path = Path(path_text)
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return []
    return df.to_dict(orient="records")


def _load_per_group_rows_from_path(path_text: str) -> list[dict[str, Any]]:
    if not path_text:
        return []
    path = Path(path_text)
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return []
    return df.to_dict(orient="records")


def _collect_mode_history_rows(results: list[_ModeOptimisation]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        if result.history:
            rows.extend(result.history)
            continue
        rows.extend(_load_history_rows_from_path(result.history_path))
    return rows


def _collect_per_group_rows(results: list[_ModeOptimisation]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        if result.per_group_rows:
            rows.extend(result.per_group_rows)
        else:
            rows.extend(_load_per_group_rows_from_path(result.per_group_path))
    return rows


def _emit_trial_progress(
    on_progress: Callable[[dict[str, Any]], None],
    cfg: ProtocolParams,
    motion_type: str,
    scope: TargetScope,
    scheme: ReferenceConfig,
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
            "reference_order_key": scheme.reference_order_key,
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
    *,
    write_best_params_all: bool = True,
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

    per_group = pd.DataFrame(_collect_per_group_rows(results))
    per_group.to_csv(motion_dir / "per_group_aae.csv", index=False, encoding="utf-8-sig")
    per_group.to_csv(motion_dir / "per_group_accuracy.csv", index=False, encoding="utf-8-sig")

    if write_best_params_all:
        payload = {
            r.mode_key: {
                "result_schema_version": RESULT_SCHEMA_VERSION,
                "motion_type": motion_type,
                "target_scope": r.target_scope.value,
                "reference_order_key": r.reference_config.reference_order_key,
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
                "acc_compare_metrics": _acc_compare_metrics_for(r),
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
                        "acc_compare_metrics": _acc_compare_metrics_for(fold),
                        "success": fold.success,
                        "reason": fold.reason,
                        "trial_history": fold.history if fold.history else _load_history_rows_from_path(fold.history_path),
                    }
                    for fold in r.fold_results
                ],
                "success": r.success,
                "reason": r.reason,
                "trial_history": r.history if r.history else _load_history_rows_from_path(r.history_path),
            }
            for r in results
        }
        (motion_dir / "best_params_all.json").write_text(
            json.dumps(_jsonify(payload), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    bayes_df = pd.DataFrame(_collect_mode_history_rows(results))
    bayes_df.to_csv(motion_dir / "bayes_curve_data.csv", index=False, encoding="utf-8-sig")
    _write_stage6_record_files(motion_dir, motion_type, results)


def _clear_mode_heavy_fields(result: _ModeOptimisation) -> None:
    result.history = []
    result.metric_arrays_by_split = {}
    if result.fold_results:
        result.per_group_rows = []
        for fold in result.fold_results:
            fold.history = []
            fold.per_group_rows = []
            fold.metric_arrays_by_split = {}


def _finalize_completed_mode(
    motion_dir: Path,
    motion_type: str,
    mode_results: list[_ModeOptimisation],
    result: _ModeOptimisation,
    *,
    write_best_params_all: bool,
) -> None:
    manifest_path = _persist_mode_artifacts(motion_dir, result, checkpoint_status="running")
    _write_motion_type_outputs(
        motion_dir,
        motion_type,
        mode_results,
        write_best_params_all=write_best_params_all,
    )
    _update_mode_checkpoint_entry(
        motion_dir,
        result,
        checkpoint_status="done",
        manifest_path=manifest_path,
        history_path=Path(result.history_path) if result.history_path else None,
    )
    _clear_mode_heavy_fields(result)


def _write_stage6_record_files(
    motion_dir: Path,
    motion_type: str,
    results: list[_ModeOptimisation],
) -> None:
    """Write current-schema deployment result records."""

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
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "preset": core_preset_metadata(resolve_core_config()),
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
    row = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "motion_type": result.motion_type,
        "split": _result_split_name(result),
        "mode": result.data_split_mode,
        "target_scope": result.target_scope.value,
        "objective_mode": result.objective_mode,
        "reference_order_key": result.reference_config.reference_order_key,
        "adaptive_filter": result.adaptive_filter,
        "adaptive_data_type": result.reference_config.reference_order_key,
        "adaptive_data_type_display": result.reference_config.reference_order_key,
        "hr_window_s": float(params.hr_window_s),
        "prewarm_s": float(getattr(params, "prewarm_s", 0.0)),
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
        "training_elapsed_s": float(result.training_elapsed_s),
        "ud_calculation_mode": result.ud_calculation_mode,
        "ud_smoothing_window_s": float(result.ud_smoothing_window_s),
        "ud_smoothing_window_samples": int(result.ud_smoothing_window_samples),
        "baseline_aae_bpm": metrics.get("baseline_aae_bpm"),
        "baseline_acc_pct": metrics.get("baseline_acc_pct"),
        "adaptive_aae_bpm": metrics.get("adaptive_aae_bpm"),
        "adaptive_acc_pct": metrics.get("adaptive_acc_pct"),
    }
    row.update(_acc_compare_metrics_for(result))
    row.update(flatten_params_for_record(params))
    return row


def _best_metrics_record(result: _ModeOptimisation) -> dict[str, Any]:
    params = result.best_params
    metrics = result.test_metrics
    return {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "motion_type": result.motion_type,
        "split": _result_split_name(result),
        "mode": result.data_split_mode,
        "target_scope": result.target_scope.value,
        "filter_type": result.adaptive_filter,
        "adaptive_data_type": result.reference_config.reference_order_key,
        "hr_window_s": float(params.hr_window_s),
        "prewarm_s": float(getattr(params, "prewarm_s", 0.0)),
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
        "training_elapsed_s": float(result.training_elapsed_s),
        "n_windows": int(metrics.get("num_windows", 0) or 0),
        "n_valid_windows": int(metrics.get("num_windows", 0) or 0),
        "n_qc_fallback": int(metrics.get("n_qc_fallback", 0) or 0),
        "n_recovery_fallback": int(metrics.get("n_recovery_fallback", 0) or 0),
    }


def _motion_frequency_params_record(result: _ModeOptimisation) -> dict[str, Any]:
    params = result.best_params
    row = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "motion_type": result.motion_type,
        "split": _result_split_name(result),
        "mode": result.data_split_mode,
        "motion_frequency_hz": result.test_metrics.get("motion_frequency_hz", float("nan")),
        "penalty_ref_channel": result.test_metrics.get("penalty_ref_channel", ""),
        "reference_order_key": result.reference_config.reference_order_key,
        "adaptive_filter": result.adaptive_filter,
        "adaptive_data_type": result.reference_config.reference_order_key,
        "hr_window_s": float(params.hr_window_s),
        "prewarm_s": float(getattr(params, "prewarm_s", 0.0)),
        "memory_ms": float(params.memory_ms),
        "history_tap_limit": params.history_tap_limit,
        "M_base": int(params.M_base),
        "C_scale": float(params.C_scale),
        "K_max": int(params.K_max),
        "Spec_Penalty_Width": float(params.Spec_Penalty_Width),
        "Spec_Penalty_Weight": float(params.Spec_Penalty_Weight),
        "reference_channel_ranking_summary": result.test_metrics.get("reference_channel_ranking_summary", ""),
    }
    row.update(flatten_params_for_record(params))
    return row


def _best_params_alignment_columns() -> list[str]:
    return [
        "result_schema_version",
        "motion_type", "split", "mode", "target_scope", "objective_mode",
        "reference_order_key", "adaptive_filter", "adaptive_data_type", "hr_window_s", "prewarm_s",
        "Fs_Target", "normalization_mode", "best_params_json", "best_tdelay_s",
        "time_bias_after_s", "alignment_score_mode", "pre_align_metric_json",
        "no_posthoc_final_aae_bpm", "no_posthoc_final_acc_pct",
        "posthoc_final_aae_bpm", "posthoc_final_acc_pct",
        "training_elapsed_s",
        "ud_calculation_mode", "ud_smoothing_window_s", "ud_smoothing_window_samples",
        "baseline_aae_bpm", "baseline_acc_pct", "adaptive_aae_bpm", "adaptive_acc_pct",
        "acc_compare_aae_bpm", "acc_compare_accuracy_pct", "acc_compare_num_windows",
        "acc_compare_posthoc_aae_bpm", "acc_compare_posthoc_accuracy_pct",
        "acc_compare_status", "acc_compare_reason",
        *_param_record_columns(),
    ]


def _best_metrics_columns() -> list[str]:
    return [
        "result_schema_version",
        "motion_type", "split", "mode", "target_scope", "filter_type", "adaptive_data_type",
        "hr_window_s", "prewarm_s", "baseline_aae_bpm", "adaptive_aae_bpm", "final_aae_bpm",
        "baseline_acc_pct", "adaptive_acc_pct", "final_acc_pct",
        "posthoc_baseline_aae_bpm", "posthoc_adaptive_aae_bpm", "posthoc_final_aae_bpm",
        "posthoc_baseline_acc_pct", "posthoc_adaptive_acc_pct", "posthoc_final_acc_pct",
        "training_elapsed_s",
        "n_windows", "n_valid_windows", "n_qc_fallback", "n_recovery_fallback",
    ]


def _motion_frequency_params_columns() -> list[str]:
    return [
        "result_schema_version",
        "motion_type", "split", "mode", "motion_frequency_hz", "penalty_ref_channel",
        "reference_order_key", "adaptive_filter", "adaptive_data_type", "hr_window_s", "prewarm_s",
        "memory_ms", "history_tap_limit", "M_base", "C_scale", "K_max",
        "Spec_Penalty_Width", "Spec_Penalty_Weight", "reference_channel_ranking_summary",
        *_param_record_columns(),
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
        "reference_order_key": result.reference_config.reference_order_key,
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
        "training_elapsed_s": float(result.training_elapsed_s),
        "ud_calculation_mode": result.ud_calculation_mode,
        "ud_smoothing_window_s": float(result.ud_smoothing_window_s),
        "ud_smoothing_window_samples": int(result.ud_smoothing_window_samples),
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
    row.update(_acc_compare_metrics_for(result))
    return row


def _acc_compare_metrics_for(result: _ModeOptimisation) -> dict[str, Any]:
    metrics = dict(getattr(result, "acc_compare_metrics", {}) or {})
    required = _empty_acc_compare_fields()
    if metrics:
        required.update({key: metrics.get(key, required[key]) for key in required})
        return required
    if result.reference_config == ReferenceConfig(('ACC',)):
        return _acc_compare_fields(result.test_metrics, status="same_as_original", reason="")
    return required




def _param_record_columns() -> list[str]:
    """Return stable ``param_*`` columns for all ProtocolTrialParams fields."""

    return [f"param_{item.name}" for item in fields(ProtocolTrialParams)]


def _summary_columns() -> list[str]:
    """Return stable summary CSV columns, including headers for empty LOGO files.

    中文说明：非 LOGO 模式也会输出空的 fold_summary_*.csv；固定表头可以让
    Notebook 检查单元格安全读取空 fold summary。
    """

    return [
        "motion_type",
        "target_scope",
        "reference_order_key",
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
        "training_elapsed_s",
        "ud_calculation_mode",
        "ud_smoothing_window_s",
        "ud_smoothing_window_samples",
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
        "acc_compare_aae_bpm",
        "acc_compare_accuracy_pct",
        "acc_compare_posthoc_aae_bpm",
        "acc_compare_posthoc_accuracy_pct",
        "acc_compare_num_windows",
        "acc_compare_status",
        "acc_compare_reason",
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
        hist = result.history if result.history else _load_history_rows_from_path(result.history_path)
        ax.grid(True, alpha=0.25)
        ax.set_title(f"{result.target_scope.value} / {result.reference_config.reference_order_key} / {result.adaptive_filter}")
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
        "reference_order_key",
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
                                "reference_order_key": result.reference_config.reference_order_key,
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
    reference_config: str,
    adaptive_filter: str,
    best_param_csv_path: str | Path,
    output_dir: str | Path,
    fs_origin: int = 100,
    ud_calculation_mode: str = "smoothed",
    ud_smoothing_window_s: float = 0.1,
    fold_id: int | None = None,
    heldout_group_id: str | None = None,
    enable_time_bias_after: bool | None = None,
    time_bias_after_range_s: tuple[float, float] | None = None,
    time_bias_after_step_s: float | None = None,
    time_bias_after_mode: str | None = None,
    postprocess_method_override: str | None = None,
) -> dict[str, Path]:
    """Re-run one sample with one best-param row and draw HR curves.

    中文说明：这是 Notebook 手动重画单元格使用的函数。主训练流程不会调用它。
    图 A 只画训练段；图 B 画全局曲线，非目标段使用普通 PPG FFT HR，目标段使用
    adaptive HR，并只保留这一条合并后的 ``ppg_hr_bpm`` 与后对齐参考曲线。
    """

    scope = TargetScope(target_scope)
    scheme = ReferenceConfig.from_key(reference_config)
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
    if postprocess_method_override:
        params = replace(params, postprocess_method=str(postprocess_method_override))
    dataset = load_and_preprocess_protocol(
        sensor_csv_path,
        ref_csv_path,
        fs_origin=fs_origin,
        ud_calculation_mode=ud_calculation_mode,
        ud_smoothing_window_s=ud_smoothing_window_s,
    )
    run = run_protocol_trial(dataset, scheme, scope, params)
    if not run.success or run.frame.empty:
        raise RuntimeError(f"手动重画失败: {run.reason}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plt = _prepare_matplotlib(out)
    group_id = Path(sensor_csv_path).stem.removeprefix("multi_")
    train_path = out / f"manual_training_hr_{group_id}_{scope.value}_{scheme.reference_order_key}_{adaptive_filter}.png"
    global_path = out / f"manual_global_hr_{group_id}_{scope.value}_{scheme.reference_order_key}_{adaptive_filter}.png"
    global_csv_path = out / f"manual_global_hr_{group_id}_{scope.value}_{scheme.reference_order_key}_{adaptive_filter}.csv"

    frame = run.frame.copy()
    train_frame = frame.loc[frame["is_filtered_segment"].astype(bool)].copy()
    _plot_hr_frame(
        plt,
        train_frame,
        train_path,
        title=f"{group_id} training-scope HR ({scope.value} / {scheme.reference_order_key} / {adaptive_filter})",
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
        current_run=run,
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
            f"{group_id} global HR ({scope.value} / {scheme.reference_order_key} / {adaptive_filter}) | "
            f"Tdelay={tdelay:.2f}s, time_bias_after={float(bias):.2f}s, "
            f"posthoc AAE={posthoc_aae:.2f} bpm, hit-rate={posthoc_acc:.1f}%"
        ),
    )
    _write_manual_global_hr_csv(global_frame, global_csv_path)
    return {"training_scope": train_path, "global": global_path, "global_csv": global_csv_path}


def redraw_subject_target_hr_curves(
    *,
    subject_dir: str | Path,
    results_root: str | Path,
    target_scope: str = "motion_post10",
    reference_config: str = "ACC",
    adaptive_filter: str = "lms",
    data_split_mode: str = "all_train",
    prewarm_s: float | None = None,
    output_dir: str | Path | None = None,
    fs_origin: int = 100,
    ud_calculation_mode: str = "smoothed",
    ud_smoothing_window_s: float = 0.1,
    postprocess_method_override: str | None = None,
) -> pd.DataFrame:
    """Redraw target-scope reference/final HR curves for every subject sample.

    This is a replay-only entry point: it restores one selected Stage-6 mode per
    motion type and never creates an Optuna study.  Post-hoc reference alignment
    is applied only after the prediction frame has been produced.
    """

    subject_path = Path(subject_dir).resolve()
    results_path = Path(results_root).resolve()
    discovery = discover_sample_pairs_with_unpaired(subject_path)
    if discovery.unpaired:
        detail = "; ".join(
            f"{item.file_name} [{item.category}]: {item.reason}" for item in discovery.unpaired
        )
        raise ValueError(f"Subject file discovery contains rejected files: {detail}")
    if not discovery.pairs:
        raise ValueError(f"No complete sensor/HR pairs found under {subject_path}")

    scope = TargetScope(target_scope)
    scheme = ReferenceConfig.from_key(reference_config)
    filter_name = str(adaptive_filter)
    motion_rank = {item.value: idx for idx, item in enumerate(MotionType)}
    pairs = sorted(
        discovery.pairs,
        key=lambda item: (motion_rank.get(item.motion_type, len(motion_rank)), int(item.motion_index)),
    )
    tw_label = "auto" if prewarm_s is None else _prewarm_run_label(float(prewarm_s))
    ud_mode, ud_window_samples = _validate_ud_calculation_config(
        ud_calculation_mode, ud_smoothing_window_s, int(fs_origin)
    )
    ud_label = f"UD_{ud_mode}_{_compact_float_label(float(ud_smoothing_window_s))}s"
    mode_signature = f"{scope.value}__{scheme.reference_order_key}__{filter_name}__{data_split_mode}__{tw_label}__{ud_label}"
    output_base = (
        Path(output_dir).resolve()
        if output_dir is not None
        else results_path / "batch_target_hr_curves"
    )
    out_root = output_base / mode_signature
    out_root.mkdir(parents=True, exist_ok=True)

    mode_cache: dict[str, tuple[pd.Series, ProtocolTrialParams]] = {}
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        base_row = {
            "subject": subject_path.name,
            "sample_id": pair.sample_id,
            "motion_type": pair.motion_type,
            "motion_index": int(pair.motion_index),
            "target_scope": scope.value,
            "reference_order_key": scheme.reference_order_key,
            "adaptive_filter": filter_name,
            "data_split_mode": data_split_mode,
            "prewarm_s": prewarm_s,
            "ud_calculation_mode": ud_mode,
            "ud_smoothing_window_s": float(ud_smoothing_window_s),
            "ud_smoothing_window_samples": int(ud_window_samples),
            "status": "failed",
            "reason": "",
            "time_bias_after_s": float("nan"),
            "time_bias_after_mode": "",
            "posthoc_target_aae_bpm": float("nan"),
            "posthoc_target_accuracy_pct": float("nan"),
            "n_target_windows": 0,
            "n_valid_windows": 0,
            "best_record_path": "",
            "plot_path": "",
            "csv_path": "",
        }
        try:
            if pair.motion_type not in mode_cache:
                motion_dir = _stage6_motion_record_dir(results_path, pair.motion_type)
                records = _read_stage6_record_tables(motion_dir)
                best_row = _select_subject_batch_best_record(
                    records["best_params"],
                    motion_type=pair.motion_type,
                    data_split_mode=data_split_mode,
                    target_scope=scope.value,
                    reference_config=scheme.reference_order_key,
                    adaptive_filter=filter_name,
                    prewarm_s=prewarm_s,
                )
                params = _params_from_stage6_record(
                    best_row,
                    adaptive_filter=filter_name,
                    prewarm_s=prewarm_s,
                )
                if postprocess_method_override:
                    params = replace(params, postprocess_method=str(postprocess_method_override))
                mode_cache[pair.motion_type] = (best_row, params)
            best_row, params = mode_cache[pair.motion_type]
            motion_dir = _stage6_motion_record_dir(results_path, pair.motion_type)
            base_row["best_record_path"] = str(motion_dir / "best_params_and_alignment.csv")

            dataset = load_and_preprocess_protocol(
                pair.sensor_csv,
                pair.ref_csv,
                fs_origin=fs_origin,
                ud_calculation_mode=ud_mode,
                ud_smoothing_window_s=ud_smoothing_window_s,
            )
            run = run_protocol_trial(
                dataset,
                scheme,
                scope,
                params,
                collect_frame=True,
                collect_stages=False,
            )
            if not run.success or run.frame.empty:
                raise RuntimeError(run.reason or "trial produced no window frame")
            detail, metrics = _build_subject_target_replay_frame(dataset, run.frame, run, best_row)
            motion_out = out_root / pair.motion_type
            motion_out.mkdir(parents=True, exist_ok=True)
            csv_path = motion_out / f"{pair.sample_id}__target_hr.csv"
            plot_path = motion_out / f"{pair.sample_id}__target_hr.png"
            detail.to_csv(csv_path, index=False, encoding="utf-8-sig")
            _plot_subject_target_hr_curve(
                plot_path,
                detail,
                title=(
                    f"{pair.sample_id} | {scope.value} / {scheme.reference_order_key} / {filter_name} | "
                    f"post-hoc AAE={metrics['aae']:.2f} bpm, accuracy={metrics['accuracy']:.1f}%"
                ),
            )
            base_row.update(
                {
                    "status": "ok",
                    "time_bias_after_s": metrics["bias"],
                    "time_bias_after_mode": metrics["bias_mode"],
                    "posthoc_target_aae_bpm": metrics["aae"],
                    "posthoc_target_accuracy_pct": metrics["accuracy"],
                    "n_target_windows": int(len(detail)),
                    "n_valid_windows": metrics["n_valid"],
                    "plot_path": str(plot_path),
                    "csv_path": str(csv_path),
                }
            )
        except Exception as exc:
            base_row["reason"] = str(exc)
        rows.append(base_row)

    manifest = pd.DataFrame(rows)
    manifest_path = out_root / "batch_target_hr_manifest.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    manifest.attrs["manifest_path"] = str(manifest_path)
    manifest.attrs["output_dir"] = str(out_root)
    return manifest


def _select_subject_batch_best_record(
    best_df: pd.DataFrame,
    *,
    motion_type: str,
    data_split_mode: str,
    target_scope: str,
    reference_config: str,
    adaptive_filter: str,
    prewarm_s: float | None,
) -> pd.Series:
    """Select exactly one Stage-6 row; batch replay never falls back silently."""

    candidates = best_df.copy()
    selectors = {
        "motion_type": motion_type,
        "mode": data_split_mode,
        "target_scope": target_scope,
        "reference_order_key": reference_config,
        "adaptive_filter": adaptive_filter,
    }
    for column, expected in selectors.items():
        if column not in candidates.columns:
            raise ValueError(f"Stage-6 best params missing selector column: {column}")
        candidates = candidates[candidates[column].astype(str) == str(expected)]
    if prewarm_s is not None:
        if "prewarm_s" not in candidates.columns:
            raise ValueError("Stage-6 best params missing selector column: prewarm_s")
        values = pd.to_numeric(candidates["prewarm_s"], errors="coerce")
        candidates = candidates[np.isclose(values, float(prewarm_s), equal_nan=False)]
    if len(candidates) != 1:
        raise ValueError(
            "Expected exactly one Stage-6 best-param row for "
            f"{motion_type}/{target_scope}/{reference_config}/{adaptive_filter}/{data_split_mode}, "
            f"found {len(candidates)}"
        )
    return candidates.iloc[0]


def _build_subject_target_replay_frame(
    dataset: ProtocolDataset,
    frame: pd.DataFrame,
    run: ProtocolRunResult,
    best_row: pd.Series,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    time_s = frame["time_s"].to_numpy(dtype=float)
    bias_result = getattr(run, "time_bias_after", None)
    if bias_result is not None and np.isfinite(float(bias_result.time_bias_after_s)):
        bias = float(bias_result.time_bias_after_s)
        bias_mode = str(getattr(bias_result, "mode", "posthoc_oracle_alignment"))
    else:
        saved_bias = pd.to_numeric(pd.Series([best_row.get("time_bias_after_s", 0.0)]), errors="coerce").iloc[0]
        bias = float(saved_bias) if np.isfinite(saved_bias) else 0.0
        bias_mode = str(best_row.get("time_bias_after_mode", "saved_or_zero"))
    reference = np.interp(
        time_s + bias,
        np.asarray(dataset.ref_time_s, dtype=float),
        np.asarray(dataset.ref_hr_bpm, dtype=float),
        left=np.nan,
        right=np.nan,
    )
    target = frame["is_filtered_segment"].astype(bool).to_numpy()
    final = frame["final_hr_bpm"].to_numpy(dtype=float)
    valid = target & np.isfinite(reference) & np.isfinite(final)
    baseline = frame["baseline_hr_bpm"]
    detail = pd.DataFrame(
        {
            "time_s": time_s,
            "reference_hr_after_bpm": reference,
            "baseline_hr_bpm": baseline,
            "adaptive_hr_bpm": frame.get("adaptive_hr_bpm", np.nan),
            "final_hr_bpm": final,
            "final_source": frame.get("final_source", ""),
            "segment_label": frame.get("segment_label", ""),
            "is_valid": valid,
            "abs_err_after_bpm": np.where(valid, np.abs(final - reference), np.nan),
        }
    ).loc[target].reset_index(drop=True)
    errors = detail.loc[detail["is_valid"], "abs_err_after_bpm"].to_numpy(dtype=float)
    return detail, {
        "bias": bias,
        "bias_mode": bias_mode,
        "aae": _nanmean(errors),
        "accuracy": _accuracy_from_abs_err(errors),
        "n_valid": int(np.isfinite(errors).sum()),
    }


def _plot_subject_target_hr_curve(out_path: Path, frame: pd.DataFrame, *, title: str) -> None:
    """Plot only post-hoc reference and final HR for one target segment."""

    plt = _prepare_matplotlib(out_path.parent)
    fig, ax = plt.subplots(figsize=(10.0, 4.5))
    ax.plot(frame["time_s"], frame["reference_hr_after_bpm"], label="Reference HR (post-hoc)", linewidth=2)
    ax.plot(frame["time_s"], frame["final_hr_bpm"], label="Final predicted HR", linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("HR (bpm)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_batch_reference_compare(
    *,
    motion_type: str,
    best_param_source_reference_config: str,
    actual_reference_reference_config: str,
    target_scope: str,
    adaptive_filter: str,
    best_param_csv_path: str | Path,
    output_dir: str | Path,
    group_ids: list[str] | None = None,
    fs_origin: int = 100,
    ud_calculation_mode: str = "smoothed",
    ud_smoothing_window_s: float = 0.1,
) -> dict[str, Path]:
    """Re-evaluate one motion type with source and actual reference schemes.

    中文说明：该函数只读取已有 best_params 与 split_files.csv，不启动 Optuna。
    同一套 best_params 会分别在来源 cascade 和实际 cascade 上复评估 test split。
    """

    scope = TargetScope(target_scope)
    source_scheme = ReferenceConfig.from_key(best_param_source_reference_config)
    actual_scheme = ReferenceConfig.from_key(actual_reference_reference_config)
    adaptive_filter = str(adaptive_filter)
    best_path = Path(best_param_csv_path)
    params = _params_from_best_csv(
        best_path,
        scope,
        source_scheme,
        adaptive_filter,
    )
    motion_dir = best_path.parent
    split_path = motion_dir / "split_files.csv"
    if not split_path.exists():
        raise FileNotFoundError(f"Cannot find split_files.csv next to best params: {split_path}")
    split_df = pd.read_csv(split_path)
    candidates = split_df.copy()
    if "motion_type" in candidates.columns:
        narrowed = candidates[candidates["motion_type"].astype(str) == str(motion_type)]
        if not narrowed.empty:
            candidates = narrowed
    if "split" in candidates.columns:
        narrowed = candidates[candidates["split"].astype(str) == "test"]
        if not narrowed.empty:
            candidates = narrowed
    if group_ids is not None:
        group_set = {str(item) for item in group_ids}
        candidates = candidates[candidates.get("group_id", pd.Series([], dtype=object)).astype(str).isin(group_set)]
    candidates = candidates[
        candidates.get("data_file", pd.Series([], dtype=object)).astype(str).str.len().gt(0)
        & candidates.get("ref_file", pd.Series([], dtype=object)).astype(str).str.len().gt(0)
    ]
    if candidates.empty:
        raise ValueError("No split/test rows with data_file/ref_file are available for batch reference compare")

    rows: list[dict[str, Any]] = []
    for _, row in candidates.iterrows():
        group_id = str(row.get("group_id", ""))
        sensor_csv = Path(str(row.get("data_file", "")))
        ref_csv = Path(str(row.get("ref_file", "")))
        try:
            dataset = load_and_preprocess_protocol(
                sensor_csv,
                ref_csv,
                fs_origin=fs_origin,
                ud_calculation_mode=ud_calculation_mode,
                ud_smoothing_window_s=ud_smoothing_window_s,
            )
            source_run = run_protocol_trial(dataset, source_scheme, scope, params, collect_frame=False)
            actual_run = run_protocol_trial(dataset, actual_scheme, scope, params, collect_frame=False)
            rows.append(
                {
                    "motion_type": str(motion_type),
                    "group_id": group_id,
                    "target_scope": scope.value,
                    "adaptive_filter": adaptive_filter,
                    "source_reference_config": source_scheme.reference_order_key,
                    "actual_reference_reference_config": actual_scheme.reference_order_key,
                    **_batch_compare_run_fields("source", source_run),
                    **_batch_compare_run_fields("actual", actual_run),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "motion_type": str(motion_type),
                    "group_id": group_id,
                    "target_scope": scope.value,
                    "adaptive_filter": adaptive_filter,
                    "source_reference_config": source_scheme.reference_order_key,
                    "actual_reference_reference_config": actual_scheme.reference_order_key,
                    **_empty_batch_compare_run_fields("source", str(exc)),
                    **_empty_batch_compare_run_fields("actual", str(exc)),
                }
            )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"batch_reference_compare_{motion_type}.csv"
    columns = _batch_reference_compare_columns()
    pd.DataFrame(rows, columns=columns).to_csv(csv_path, index=False, encoding="utf-8-sig")
    return {"csv": csv_path}


def _batch_compare_run_fields(prefix: str, run: ProtocolRunResult) -> dict[str, Any]:
    arrays = getattr(run, "metric_arrays", {}) or {}
    n_windows = int(np.asarray(arrays.get("ref_hr_bpm", []), dtype=float).size)
    if n_windows == 0:
        n_windows = int(getattr(run, "posthoc_n_valid_windows", 0) or 0)
    return {
        f"{prefix}_success": bool(run.success),
        f"{prefix}_reason": str(run.reason),
        f"{prefix}_final_aae_bpm": float(run.final_aae_bpm),
        f"{prefix}_final_accuracy_pct": float(run.final_acc_pct),
        f"{prefix}_adaptive_aae_bpm": float(run.adaptive_aae_bpm),
        f"{prefix}_adaptive_accuracy_pct": float(run.adaptive_acc_pct),
        f"{prefix}_num_windows": n_windows,
    }


def _empty_batch_compare_run_fields(prefix: str, reason: str) -> dict[str, Any]:
    return {
        f"{prefix}_success": False,
        f"{prefix}_reason": str(reason),
        f"{prefix}_final_aae_bpm": float("nan"),
        f"{prefix}_final_accuracy_pct": float("nan"),
        f"{prefix}_adaptive_aae_bpm": float("nan"),
        f"{prefix}_adaptive_accuracy_pct": float("nan"),
        f"{prefix}_num_windows": 0,
    }


def _batch_reference_compare_columns() -> list[str]:
    return [
        "motion_type",
        "group_id",
        "target_scope",
        "adaptive_filter",
        "source_reference_config",
        "actual_reference_reference_config",
        "source_success",
        "source_reason",
        "source_final_aae_bpm",
        "source_final_accuracy_pct",
        "source_adaptive_aae_bpm",
        "source_adaptive_accuracy_pct",
        "source_num_windows",
        "actual_success",
        "actual_reason",
        "actual_final_aae_bpm",
        "actual_final_accuracy_pct",
        "actual_adaptive_aae_bpm",
        "actual_adaptive_accuracy_pct",
        "actual_num_windows",
    ]


def _time_bias_after_for_redraw(
    best_row: pd.Series,
    dataset: ProtocolDataset,
    frame: pd.DataFrame,
    combined_hr_bpm: np.ndarray,
    params: ProtocolTrialParams,
    *,
    current_run: ProtocolRunResult,
    enable_time_bias_after: bool | None,
    time_bias_after_range_s: tuple[float, float] | None,
    time_bias_after_step_s: float | None,
    time_bias_after_mode: str | None,
) -> tuple[float, str, str]:
    """Reuse saved scalar bias, or the current sample bias for aggregate records."""
    value = float(best_row["time_bias_after_s"])
    if np.isnan(value) and current_run.time_bias_after is not None:
        sample_bias = float(current_run.time_bias_after.time_bias_after_s)
        if np.isfinite(sample_bias):
            return sample_bias, "current_run", str(current_run.time_bias_after.mode)
    if not np.isfinite(value):
        raise ValueError("Invalid saved time_bias_after_s and no finite current sample bias")
    mode = str(time_bias_after_mode or params.Time_Bias_After_Mode)
    return value, "saved_best_params", mode


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
    scheme: ReferenceConfig,
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
    params = protocol_params_from_record(row)
    return replace(params, adaptive_filter=adaptive_filter)


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
    scheme: ReferenceConfig,
    adaptive_filter: str,
    *,
    fold_id: int | None = None,
    heldout_group_id: str | None = None,
) -> pd.Series:
    """Select the best-params row used by manual redraw and params decoding."""

    df = pd.read_csv(best_param_csv_path)
    _require_result_table_schema(df, str(best_param_csv_path))
    mask = pd.Series(True, index=df.index)
    for column, value in (
        ("target_scope", scope.value),
        ("reference_order_key", scheme.reference_order_key),
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
    else:
        raise ValueError(f"best param CSV is empty: {best_param_csv_path}")
    return row


def _parse_tuple_param(value: Any, default: tuple[Any, ...]) -> tuple[Any, ...]:
    """Restore a tuple parameter from its current CSV JSON representation."""
    parsed = ast.literal_eval(value) if isinstance(value, str) else value
    if not isinstance(parsed, (list, tuple)) or len(parsed) != len(default):
        raise ValueError(f"Invalid tuple parameter: {value!r}")
    return tuple(type(default[idx])(item) for idx, item in enumerate(parsed))


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
    win_len = int(round(float(params.hr_window_s) * fs))
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


def _file_content_signature(path: str | Path) -> str:
    """Return a reproducible SHA-256 signature, including a clear missing marker."""

    source = Path(path)
    if not source.is_file():
        return f"missing:{source.resolve()}"
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_cross_motion_summary_table(
    *,
    table_output_dir: str | Path,
    results_root: str | Path,
    adaptive_filter: str,
    adaptive_data_type: str = "",
    reference_config: str | None = None,
    target_scope: str = "motion_only",
    prewarm_s: float | None = None,
) -> Path:
    """Aggregate Stage-6 result records across all motion types.

    中文说明：该函数只读取已训练完成的记录文件，不重新训练、不回放样本。输出表用于
    横向比较不同运动类型下 baseline/adaptive/final 以及 post-hoc upper-bound 指标。
    """

    output_dir = Path(table_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_type = adaptive_data_type or str(reference_config or "")
    rows: list[dict[str, Any]] = []
    for motion_dir in _scan_stage6_motion_dirs(Path(results_root)):
        metrics_path = motion_dir / "best_metrics.csv"
        params_path = motion_dir / "best_params_and_alignment.csv"
        freq_path = motion_dir / "motion_frequency_and_params.csv"
        if not (metrics_path.exists() and params_path.exists() and freq_path.exists()):
            continue
        metrics_df = pd.read_csv(metrics_path)
        _require_result_table_schema(metrics_df, str(metrics_path))
        params_df = pd.read_csv(params_path)
        _require_result_table_schema(params_df, str(params_path))
        freq_df = pd.read_csv(freq_path)
        _require_result_table_schema(freq_df, str(freq_path))
        selected_metrics = _filter_cross_motion_records(
            metrics_df,
            adaptive_filter=adaptive_filter,
            adaptive_data_type=data_type,
            reference_config=reference_config,
            target_scope=target_scope,
            prewarm_s=prewarm_s,
            filter_column="filter_type",
        )
        fusion_distribution = _read_fusion_distribution(motion_dir / "full_report.json")
        for _, metric_row in selected_metrics.iterrows():
            motion_type = _row_value(metric_row, "motion_type", motion_dir.name)
            params_row = _first_matching_summary_row(params_df, metric_row, adaptive_filter, data_type, reference_config, prewarm_s)
            freq_row = _first_matching_summary_row(freq_df, metric_row, adaptive_filter, data_type, reference_config, prewarm_s)
            rows.append(
                {
                    "motion_type": motion_type,
                    "target_scope": _row_value(metric_row, "target_scope", target_scope),
                    "split": _row_value(metric_row, "split", _row_value(params_row, "split", "")),
                    "mode": _row_value(metric_row, "mode", _row_value(params_row, "mode", "")),
                    "hr_window_s": _row_float(metric_row, "hr_window_s", _row_float(params_row, "hr_window_s")),
                    "prewarm_s": _row_float(metric_row, "prewarm_s", _row_float(params_row, "prewarm_s")),
                    "baseline_aae": _row_float(metric_row, "baseline_aae_bpm"),
                    "adaptive_aae": _row_float(metric_row, "adaptive_aae_bpm"),
                    "final_aae": _row_float(metric_row, "final_aae_bpm"),
                    "posthoc_baseline_aae": _row_float(metric_row, "posthoc_baseline_aae_bpm"),
                    "posthoc_adaptive_aae": _row_float(metric_row, "posthoc_adaptive_aae_bpm"),
                    "posthoc_final_aae": _row_float(metric_row, "posthoc_final_aae_bpm"),
                    "baseline_acc": _row_float(metric_row, "baseline_acc_pct"),
                    "adaptive_acc": _row_float(metric_row, "adaptive_acc_pct"),
                    "final_acc": _row_float(metric_row, "final_acc_pct"),
                    "posthoc_baseline_acc": _row_float(metric_row, "posthoc_baseline_acc_pct"),
                    "posthoc_adaptive_acc": _row_float(metric_row, "posthoc_adaptive_acc_pct"),
                    "posthoc_final_acc": _row_float(metric_row, "posthoc_final_acc_pct"),
                    "best_tdelay_s": _row_float(params_row, "best_tdelay_s"),
                    "time_bias_after_s": _row_float(params_row, "time_bias_after_s"),
                    "motion_frequency_hz": _row_float(freq_row, "motion_frequency_hz"),
                    "penalty_ref_channel": _row_value(freq_row, "penalty_ref_channel", ""),
                    "adaptive_filter": adaptive_filter,
                    "adaptive_data_type": data_type,
                    "reference_order_key": str(reference_config or _row_value(params_row, "reference_order_key", data_type)),
                    "final_source_distribution": fusion_distribution,
                }
            )
    summary = pd.DataFrame(rows)
    if not summary.empty:
        motion_rank = {motion.value: index for index, motion in enumerate(MotionType)}
        summary["_motion_rank"] = summary["motion_type"].map(motion_rank).fillna(len(motion_rank))
        summary = (
            summary.sort_values(["_motion_rank", "target_scope", "split", "mode"])
            .drop(columns="_motion_rank")
            .reset_index(drop=True)
        )
    out_path = output_dir / f"summary_{adaptive_filter}_{data_type}_{_prewarm_run_label(float(prewarm_s or 0.0))}.csv"
    summary.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


def _scan_stage6_motion_dirs(results_root: Path) -> list[Path]:
    """Return motion-type directories that contain compact Stage-6 records."""

    base = results_root / "motion_types" if (results_root / "motion_types").exists() else results_root
    if not base.exists():
        return []
    present = {
        path.name: path
        for path in base.iterdir()
        if path.is_dir() and (path / "best_metrics.csv").exists()
    }
    return [present[motion.value] for motion in MotionType if motion.value in present]


def _filter_cross_motion_records(
    df: pd.DataFrame,
    *,
    adaptive_filter: str,
    adaptive_data_type: str,
    reference_config: str | None,
    target_scope: str,
    prewarm_s: float | None,
    filter_column: str,
) -> pd.DataFrame:
    """Filter one Stage-6 table by the summary selectors that are present."""

    selected = df.copy()
    selectors = [
        ("target_scope", TargetScope(target_scope).value),
        (filter_column, adaptive_filter),
        ("adaptive_filter", adaptive_filter),
        ("adaptive_data_type", adaptive_data_type),
        ("reference_order_key", str(reference_config or "")),
    ]
    for column, expected in selectors:
        if not expected or column not in selected.columns:
            continue
        narrowed = selected[selected[column].astype(str) == str(expected)]
        if not narrowed.empty:
            selected = narrowed
    if prewarm_s is not None and "prewarm_s" in selected.columns:
        values = pd.to_numeric(selected["prewarm_s"], errors="coerce")
        narrowed = selected[np.isclose(values.astype(float), float(prewarm_s), equal_nan=False)]
        if not narrowed.empty:
            selected = narrowed
    return selected


def _first_matching_summary_row(
    df: pd.DataFrame,
    metric_row: pd.Series,
    adaptive_filter: str,
    adaptive_data_type: str,
    reference_config: str | None,
    prewarm_s: float | None,
) -> pd.Series:
    """Pick metadata row matching one metrics row; return an empty row if absent."""

    if df.empty:
        return pd.Series(dtype=object)
    selected = df.copy()
    for column in ("motion_type", "split", "mode", "target_scope"):
        expected = _row_value(metric_row, column, "")
        if expected and column in selected.columns:
            narrowed = selected[selected[column].astype(str) == str(expected)]
            if not narrowed.empty:
                selected = narrowed
    selected = _filter_cross_motion_records(
        selected,
        adaptive_filter=adaptive_filter,
        adaptive_data_type=adaptive_data_type,
        reference_config=reference_config,
        target_scope=_row_value(metric_row, "target_scope", "motion_only"),
        prewarm_s=prewarm_s,
        filter_column="filter_type",
    )
    return selected.iloc[0] if not selected.empty else pd.Series(dtype=object)


def _read_fusion_distribution(report_path: Path) -> str:
    """Read final_source distribution from full_report.json as compact JSON text."""

    if not report_path.exists():
        return "{}"
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "{}"
    _require_result_schema(payload, str(report_path))
    distribution = payload.get("fusion_source_distribution", {})
    if not isinstance(distribution, dict):
        distribution = {}
    return json.dumps(_jsonify(distribution), ensure_ascii=False, sort_keys=True)


def _row_value(row: pd.Series, column: str, default: Any = "") -> Any:
    if row is None or column not in row.index:
        return default
    value = row.get(column)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    return value


def _row_float(row: pd.Series, column: str, default: float = float("nan")) -> float:
    value = _row_value(row, column, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def plot_window_diagnostics_from_records(
    *,
    signal_csv: str | Path,
    ref_csv: str | Path,
    output_dir: str | Path,
    adaptive_filter: str,
    adaptive_data_type: str = "",
    reference_config: str | None = None,
    override_reference_config: str | None = None,
    results_root: str | Path,
    window_start_s: float,
    prewarm_s: float | None = None,
    motion_type: str,
    split: str = "",
    mode: str = "",
    target_scope: str = "motion_only",
    fs_origin: int = 100,
    ud_calculation_mode: str = "smoothed",
    ud_smoothing_window_s: float = 0.1,
    postprocess_method_override: str | None = None,
) -> dict[str, Any]:
    """Draw waveform and spectrum diagnostics for one sensor-domain FFT sub-window.

    中文说明：``window_start_s`` 始终表示传感器时间域的 FFT 子窗起点；诊断图会同时
    标出 ``prewarm_s`` 前置收敛上下文和最终 FFT/HR 提取区间。函数只读取 Stage-6 记录
    恢复参数，不重新训练，也不读取 ref HR 来决定 final 融合。
    """

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    motion_dir = _stage6_motion_record_dir(Path(results_root), motion_type)
    records = _read_stage6_record_tables(motion_dir)
    best_row = _select_stage6_best_record(
        records["best_params"],
        motion_type=motion_type,
        split=split,
        mode=mode,
        target_scope=TargetScope(target_scope).value,
        adaptive_filter=adaptive_filter,
        adaptive_data_type=adaptive_data_type,
        reference_config=reference_config,
        prewarm_s=prewarm_s,
    )
    params = _params_from_stage6_record(best_row, adaptive_filter=adaptive_filter, prewarm_s=prewarm_s)
    if postprocess_method_override:
        params = replace(params, postprocess_method=str(postprocess_method_override))
    scheme_text = str(reference_config or _stage6_row_text(best_row, "reference_order_key") or adaptive_data_type)
    scheme_to_run = override_reference_config if override_reference_config else scheme_text
    scheme = ReferenceConfig.from_key(scheme_to_run)
    scope = TargetScope(target_scope)
    dataset = load_and_preprocess_protocol(
        signal_csv,
        ref_csv,
        fs_origin=fs_origin,
        ud_calculation_mode=ud_calculation_mode,
        ud_smoothing_window_s=ud_smoothing_window_s,
    )
    dataset = resample_protocol_dataset(dataset, fs_target=int(params.Fs_Target))
    label_data_type = adaptive_data_type or scheme_text
    if override_reference_config:
        title_prefix = f"Params: {scheme_text} -> Run: {override_reference_config}"
        label = (
            f"cross_{label_data_type}_to_{override_reference_config}_{motion_type}_"
            f"{params.adaptive_filter}_{_prewarm_run_label(params.prewarm_s)}_{scope.value}_start{_compact_float_label(window_start_s)}s"
        )
    else:
        title_prefix = scheme.reference_order_key
        label = (
            f"{motion_type}_{params.adaptive_filter}_{label_data_type}_"
            f"{_prewarm_run_label(params.prewarm_s)}_{scope.value}_start{_compact_float_label(window_start_s)}s"
        )

    diagnostic = _run_window_diagnostic_variant(
        dataset=dataset, scheme=scheme, scope=scope, params=params,
        output_dir=out_dir, label=label, variant_name="core_enhanced_v1",
        motion_type=motion_type,
        title_prefix=title_prefix, window_start_s=window_start_s,
    )
    return {**diagnostic, "best_params": params.to_dict(), "best_record": best_row.to_dict()}


def _run_window_diagnostic_variant(
    *,
    dataset: ProtocolDataset,
    scheme: ReferenceConfig,
    scope: TargetScope,
    params: ProtocolTrialParams,
    output_dir: Path,
    label: str,
    variant_name: str,
    motion_type: str,
    title_prefix: str,
    window_start_s: float,
) -> dict[str, Any]:
    """Run one diagnostic replay variant and write waveform/spectrum/weights plots."""

    run = run_protocol_trial(dataset, scheme, scope, params, collect_frame=True, collect_stages=True)
    if not run.success:
        raise RuntimeError(f"Window diagnostic replay failed ({variant_name}): {run.reason}")
    frame = run.frame.copy()
    if frame.empty:
        raise RuntimeError(f"Window diagnostic replay produced no frame ({variant_name})")

    dataset = apply_ppg_input_transform(dataset, params)
    row = _select_window_diagnostic_row(frame, window_start_s)
    stages = _parse_window_stages(row)
    raw_time, algorithm_input_ppg = _diagnostic_ppg_context(dataset, row, params)
    filtered = _diagnostic_filtered_signal(stages, algorithm_input_ppg)
    fft_start = float(row.get("fft_start_s", window_start_s))
    fft_end = float(row.get("fft_end_s", fft_start + float(params.hr_window_s)))
    adaptive_start = float(row.get("adaptive_start_s", fft_start - float(params.prewarm_s)))
    penalty_ref_channel = str(row.get("penalty_ref_channel", "")) or _stage_penalty_ref_channel(stages)
    penalty_ref_context = _diagnostic_channel_context(dataset, row, params, penalty_ref_channel, len(algorithm_input_ppg))
    raw_fft = _diagnostic_fft_window(
        algorithm_input_ppg,
        row,
        fs=int(dataset.fs),
        fft_start_s=fft_start,
        fft_end_s=fft_end,
        adaptive_start_s=adaptive_start,
    )
    filtered_fft = _diagnostic_fft_window(
        filtered,
        row,
        fs=int(dataset.fs),
        fft_start_s=fft_start,
        fft_end_s=fft_end,
        adaptive_start_s=adaptive_start,
    )
    penalty_ref_fft = _diagnostic_fft_window(
        penalty_ref_context,
        row,
        fs=int(dataset.fs),
        fft_start_s=fft_start,
        fft_end_s=fft_end,
        adaptive_start_s=adaptive_start,
    )
    waveform_path = output_dir / f"window_waveform_{label}_{variant_name}.png"
    spectrum_path = output_dir / f"window_spectrum_{label}_{variant_name}.png"
    weights_path = output_dir / f"window_weights_{label}_{variant_name}.png"
    title_suffix = variant_name.replace("_", " ")
    _plot_window_waveform_diagnostic(
        waveform_path,
        time_s=raw_time,
        algorithm_input_ppg=algorithm_input_ppg,
        filtered=filtered,
        stages=stages,
        adaptive_start_s=adaptive_start,
        fft_start_s=fft_start,
        fft_end_s=fft_end,
        title=(
            f"{motion_type} | {params.adaptive_filter} | {title_prefix} | "
            f"{_prewarm_run_label(params.prewarm_s)} | {title_suffix}"
        ),
    )
    spectrum_info = _plot_window_spectrum_diagnostic(
        spectrum_path,
        algorithm_input_ppg=raw_fft,
        filtered=filtered_fft,
        penalty_ref=penalty_ref_fft,
        fs=int(dataset.fs),
        penalty_width_hz=float(getattr(params, "Spec_Penalty_Width", 0.2)),
        penalty_weight=float(getattr(params, "Spec_Penalty_Weight", 0.2)),
        penalty_ref_channel=penalty_ref_channel,
        title=f"{motion_type} spectrum | {title_suffix} | start {window_start_s:g}s",
    )
    _plot_window_weight_diagnostic(
        weights_path,
        stages=stages,
        title=f"{motion_type} weights | {params.adaptive_filter} | {title_suffix} | start {window_start_s:g}s",
    )
    rejected_stage_count = sum(1 for stage in stages if stage.get("accepted") is False)
    return {
        "waveform": waveform_path,
        "spectrum": spectrum_path,
        "weights": weights_path,
        "stages": stages,
        "window": row.to_dict(),
        "params": params.to_dict(),
        "spectrum_info": spectrum_info,
        "guard_policy": params.cascade_guard_policy,
        "stage_count": len(stages),
        "rejected_stage_count": int(rejected_stage_count),
    }


def _select_window_diagnostic_row(frame: pd.DataFrame, window_start_s: float) -> pd.Series:
    """Return the row whose FFT sub-window starts closest to the requested time."""

    if "fft_start_s" not in frame.columns:
        raise KeyError("Diagnostic frame is missing fft_start_s")
    starts = pd.to_numeric(frame["fft_start_s"], errors="coerce").to_numpy(dtype=float)
    if starts.size == 0 or not np.isfinite(starts).any():
        raise ValueError("Diagnostic frame has no finite fft_start_s values")
    idx = int(np.nanargmin(np.abs(starts - float(window_start_s))))
    return frame.iloc[idx]


def _parse_window_stages(row: pd.Series) -> list[dict[str, Any]]:
    """Parse cascade stage metadata for one diagnostic window."""

    column = "adaptive_stages_json"
    text = row.get(column, "")
    if text is None or (isinstance(text, float) and math.isnan(text)) or not str(text).strip():
        return []
    try:
        stages = json.loads(str(text))
    except json.JSONDecodeError:
        return []
    return stages if isinstance(stages, list) else []


def _diagnostic_motion_frequency(
    run: ProtocolRunResult,
    motion_frequency_df: pd.DataFrame,
    motion_type: str,
) -> float:
    """Choose motion-frequency metadata for spectrum annotations."""

    if run.motion_frequency is not None and np.isfinite(float(run.motion_frequency)):
        return float(run.motion_frequency)
    if not motion_frequency_df.empty and "motion_frequency_hz" in motion_frequency_df.columns:
        candidates = motion_frequency_df
        if "motion_type" in candidates.columns:
            narrowed = candidates[candidates["motion_type"].astype(str) == str(motion_type)]
            if not narrowed.empty:
                candidates = narrowed
        value = pd.to_numeric(candidates["motion_frequency_hz"], errors="coerce").dropna()
        if not value.empty:
            return float(value.iloc[0])
    return float("nan")


def _diagnostic_ppg_context(
    dataset: ProtocolDataset,
    row: pd.Series,
    params: ProtocolTrialParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the raw PPG context used for the selected diagnostic window."""

    start_idx, end_idx, adaptive_start = _diagnostic_context_indices(dataset, row, params)
    if end_idx <= start_idx:
        fs = int(dataset.fs)
        end_idx = min(len(dataset.ppg_green), start_idx + max(1, int(round(float(params.hr_window_s) * fs))))
    time_s = np.asarray(dataset.time_s[start_idx:end_idx], dtype=float)
    signal = np.asarray(dataset.ppg_green[start_idx:end_idx], dtype=float)
    if time_s.size == 0:
        time_s = np.asarray([adaptive_start], dtype=float)
        signal = np.asarray([np.nan], dtype=float)
    return time_s, signal


def _diagnostic_context_indices(
    dataset: ProtocolDataset,
    row: pd.Series,
    params: ProtocolTrialParams,
) -> tuple[int, int, float]:
    """Return source indices for the displayed prewarm_s + FFT diagnostic context."""

    fs = int(dataset.fs)
    fft_start = float(row.get("fft_start_s", 0.0))
    fft_end = float(row.get("fft_end_s", fft_start + float(params.hr_window_s)))
    adaptive_start = float(row.get("adaptive_start_s", fft_start - float(params.prewarm_s)))
    start_idx = max(0, int(round(adaptive_start * fs)))
    end_idx = min(len(dataset.time_s), int(round(fft_end * fs)))
    return start_idx, end_idx, adaptive_start


def _diagnostic_channel_context(
    dataset: ProtocolDataset,
    row: pd.Series,
    params: ProtocolTrialParams,
    channel: str,
    fallback_len: int,
) -> np.ndarray:
    """Extract one reference-channel context for FFT-window motion-peak diagnostics."""

    if not channel:
        return np.full(int(fallback_len), np.nan, dtype=float)
    channels = dataset.channels()
    if channel not in channels:
        return np.full(int(fallback_len), np.nan, dtype=float)
    start_idx, end_idx, _ = _diagnostic_context_indices(dataset, row, params)
    values = np.asarray(channels[channel][start_idx:end_idx], dtype=float)
    if values.size == int(fallback_len):
        return values
    return _align_signal_to_context(values, int(fallback_len))


def _stage_penalty_ref_channel(stages: list[dict[str, Any]]) -> str:
    """Return the first recorded penalty reference channel from cascade metadata."""

    for stage in stages:
        value = stage.get("penalty_ref_channel", "")
        if value:
            return str(value)
    return ""


def _diagnostic_filtered_signal(
    stages: list[dict[str, Any]],
    algorithm_input_ppg: np.ndarray,
) -> np.ndarray:
    """Return the retained stage output, or input when no adaptive stage ran."""
    signal_keys = ("output_signal",)
    for stage in reversed(stages):
        for signal_key in signal_keys:
            if signal_key not in stage:
                continue
            values = np.asarray(stage.get(signal_key), dtype=float)
            if values.size:
                return _align_signal_to_context(values, len(algorithm_input_ppg))
    return np.asarray(algorithm_input_ppg, dtype=float)


def _align_signal_to_context(values: np.ndarray, target_len: int) -> np.ndarray:
    """Right-align a stage vector to the diagnostic context without drawing fake prefix lines."""

    arr = np.asarray(values, dtype=float)
    if target_len <= 0:
        return arr
    if arr.size == target_len:
        return arr
    if arr.size > target_len:
        return arr[-target_len:]
    return np.concatenate([np.full(target_len - arr.size, np.nan, dtype=float), arr])


def _diagnostic_fft_window(
    values: np.ndarray,
    row: pd.Series,
    *,
    fs: int,
    fft_start_s: float,
    fft_end_s: float,
    adaptive_start_s: float,
) -> np.ndarray:
    """Crop a diagnostic context down to the exact FFT/HR extraction sub-window."""

    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    offset = _safe_int(row.get("fft_offset_samples", None), default=int(round((fft_start_s - adaptive_start_s) * fs)))
    length = _safe_int(row.get("fft_input_samples", None), default=int(round((fft_end_s - fft_start_s) * fs)))
    start = max(0, min(int(offset), arr.size))
    stop = max(start, min(start + max(1, int(length)), arr.size))
    return arr[start:stop]


def _safe_int(value: Any, *, default: int) -> int:
    """Convert saved CSV values to int while tolerating missing/NaN cells."""

    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return int(default)
        return int(round(float(value)))
    except (TypeError, ValueError):
        return int(default)


def _plot_window_waveform_diagnostic(
    out_path: Path,
    *,
    time_s: np.ndarray,
    algorithm_input_ppg: np.ndarray,
    filtered: np.ndarray,
    stages: list[dict[str, Any]],
    adaptive_start_s: float,
    fft_start_s: float,
    fft_end_s: float,
    title: str,
) -> None:
    """Plot raw/stage waveforms and final adaptive output for one window."""

    plt = _prepare_matplotlib(out_path)
    fig, ax_left = plt.subplots(figsize=(10, 4.8))
    ax_right = ax_left.twinx()
    ax_left.axvspan(adaptive_start_s, fft_start_s, color="#b7b7b7", alpha=0.18, lw=0, label="prewarm_s context")
    ax_left.axvspan(fft_start_s, fft_end_s, color="#83c5be", alpha=0.20, lw=0, label="FFT window")
    ax_left.plot(time_s, algorithm_input_ppg, color="#1f77b4", lw=1.2, label="PPG before adaptive")
    stage_colors = ["#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    for idx, stage in enumerate(stages, start=1):
        if "output_signal" not in stage:
            continue
        values = _align_signal_to_context(np.asarray(stage["output_signal"], dtype=float), len(time_s))
        color = stage_colors[(idx - 1) % len(stage_colors)]
        ax_left.plot(
            time_s,
            values,
            color=color,
            lw=0.95,
            alpha=0.62,
            label=f"stage {idx}: {stage.get('channel', '')}",
        )
        if "cascade_full_output_signal" in stage:
            full_values = _align_signal_to_context(
                np.asarray(stage["cascade_full_output_signal"], dtype=float),
                len(time_s),
            )
            ax_left.plot(time_s, full_values, color=color, lw=0.75, alpha=0.35, linestyle="--", label=f"完整级联 {idx}")
        if "cascade_guarded_output_signal" in stage:
            guarded_values = _align_signal_to_context(
                np.asarray(stage["cascade_guarded_output_signal"], dtype=float),
                len(time_s),
            )
            ax_left.plot(time_s, guarded_values, color=color, lw=1.1, alpha=0.75, linestyle="-.", label=f"guard 后实际使用 {idx}")
    ax_right.plot(time_s, filtered, color="#2ca02c", lw=1.6, label="PPG after final adaptive")
    ax_left.set_title(title)
    ax_left.set_xlabel("Sensor time (s)")
    ax_left.set_ylabel("Algorithm input/stage amplitude")
    ax_right.set_ylabel("Final adaptive amplitude")
    ax_left.grid(True, alpha=0.25)
    ax_left.legend(loc="upper left", fontsize=8)
    ax_right.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_window_weight_diagnostic(
    out_path: Path,
    *,
    stages: list[dict[str, Any]],
    title: str,
) -> None:
    """Plot compact per-stage weight/dictionary diagnostics when present."""

    plt = _prepare_matplotlib(out_path)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    plotted = False
    keys = (
        "weight_norm_t",
        "weight_norm_linear_t",
        "weight_norm_quadratic_t",
        "theta_norm_t",
        "dictionary_size_t",
    )
    for stage_idx, stage in enumerate(stages, start=1):
        channel = str(stage.get("channel", ""))
        for key in keys:
            if key not in stage:
                continue
            values = np.asarray(stage.get(key), dtype=float).ravel()
            if values.size == 0:
                continue
            ax.plot(np.arange(values.size, dtype=float), values, lw=1.0, label=f"stage {stage_idx} {channel} {key}")
            plotted = True
    if not plotted:
        ax.plot(np.asarray([0.0]), np.asarray([0.0]), lw=1.0, label="no weight diagnostics")
    ax.set_title(title)
    ax.set_xlabel("diagnostic sample")
    ax.set_ylabel("norm / size")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_window_spectrum_diagnostic(
    out_path: Path,
    *,
    algorithm_input_ppg: np.ndarray,
    filtered: np.ndarray,
    penalty_ref: np.ndarray,
    fs: int,
    penalty_width_hz: float,
    penalty_weight: float,
    penalty_ref_channel: str,
    title: str,
) -> dict[str, float]:
    """Plot FFT-window spectra and annotate motion and penalized-spectrum HR peaks."""

    plt = _prepare_matplotlib(out_path)
    raw_freq, raw_amp = compute_power_spectrum(algorithm_input_ppg, fs, apply_hamming=True, demean=True)
    filt_freq, filt_amp = compute_power_spectrum(filtered, fs, apply_hamming=True, demean=True)
    ref_freq, ref_amp = compute_power_spectrum(penalty_ref, fs, apply_hamming=True, demean=True)
    policy_params = resolve_core_config()
    motion_peaks, motion_scores = spectrum_peaks((ref_freq, ref_amp), policy_params.candidate_peak_threshold_ratio, (0.2, 5.0))
    evidence, _ = spectrum_peaks((filt_freq, filt_amp), policy_params.full_candidate_peak_threshold_ratio, (0.5, 4.0))
    policy = PenaltyPolicy.build(motion_peaks, motion_scores, evidence,
                                width_hz=penalty_width_hz, base_retention=penalty_weight)
    motion_frequency_hz = policy.centers_hz[0] if policy.centers_hz else float("nan")
    penalized_amp = filt_amp * policy.build_weights(filt_freq)
    penalized_hr_hz = _dominant_frequency_from_spectrum(filt_freq, penalized_amp, 0.5, 4.0)
    raw_amp = _normalise_fft_amplitude(raw_amp)
    filt_amp = _normalise_fft_amplitude(filt_amp)
    penalized_plot_amp = _normalise_fft_amplitude(penalized_amp)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(raw_freq, raw_amp, color="#1f77b4", lw=1.2, label="PPG algorithm input FFT window")
    ax.plot(filt_freq, filt_amp, color="#2ca02c", lw=1.3, label="PPG after adaptive FFT window")
    ax.plot(
        filt_freq,
        penalized_plot_amp,
        color="#9467bd",
        lw=1.2,
        ls="-.",
        label="PPG after adaptive + spectral penalty",
    )
    if np.isfinite(motion_frequency_hz):
        width = max(0.0, float(penalty_width_hz))
        ax.axvline(
            motion_frequency_hz,
            color="#d62728",
            lw=1.0,
            ls="--",
            label=f"motion artifact peak {motion_frequency_hz:.2f} Hz",
        )
        ax.axvspan(
            max(0.0, motion_frequency_hz - width),
            motion_frequency_hz + width,
            color="#d62728",
            alpha=0.12,
            lw=0,
            label=f"penalty band {penalty_ref_channel}".strip(),
        )
    if np.isfinite(penalized_hr_hz):
        penalized_hr_bpm = float(penalized_hr_hz * 60.0)
        ax.axvline(
            penalized_hr_hz,
            color="#111111",
            lw=1.0,
            ls=":",
            label=f"penalized spectrum HR {penalized_hr_bpm:.1f} bpm",
        )
    else:
        penalized_hr_bpm = float("nan")
    ax.set_xlim(left=0.0, right=min(max(float(fs) / 2.0, 0.5), 5.0))
    ax.set_title(title)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized FFT amplitude")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return {
        "motion_artifact_peak_hz": float(motion_frequency_hz),
        "penalized_spectrum_hr_bpm": float(penalized_hr_bpm),
    }


def _normalise_fft_amplitude(amp: np.ndarray) -> np.ndarray:
    """Scale one FFT magnitude curve to a maximum finite value of one."""

    arr = np.asarray(amp, dtype=float).copy()
    finite = np.isfinite(arr)
    if not finite.any():
        return arr
    max_amp = float(np.nanmax(arr[finite]))
    if max_amp <= 0.0:
        arr[finite] = 0.0
        return arr
    arr[finite] = arr[finite] / max_amp
    return arr


def _dominant_frequency_from_spectrum(
    freq: np.ndarray,
    amp: np.ndarray,
    low_hz: float,
    high_hz: float,
) -> float:
    """Return the strongest finite frequency in one band from an already computed spectrum."""

    freq_arr = np.asarray(freq, dtype=float)
    amp_arr = np.asarray(amp, dtype=float)
    n = min(freq_arr.size, amp_arr.size)
    if n == 0:
        return float("nan")
    freq_arr = freq_arr[:n]
    amp_arr = amp_arr[:n]
    mask = (
        np.isfinite(freq_arr)
        & np.isfinite(amp_arr)
        & (freq_arr >= float(low_hz))
        & (freq_arr <= float(high_hz))
    )
    if not mask.any():
        return float("nan")
    valid_idx = np.flatnonzero(mask)
    max_amp = float(np.nanmax(amp_arr[valid_idx]))
    if max_amp <= 0.0:
        return float("nan")
    idx = valid_idx[int(np.argmax(amp_arr[valid_idx]))]
    return float(freq_arr[idx])


def _compact_float_label(value: float) -> str:
    text = f"{float(value):g}".replace(".", "p").replace("-", "m")
    return text


def replay_best_record_hr_curves(
    *,
    signal_csv: str | Path,
    ref_csv: str | Path,
    output_dir: str | Path,
    motion_type: str,
    split: str = "",
    mode: str = "",
    target_scope: str = "motion_only",
    adaptive_filter: str = "lms",
    adaptive_data_type: str = "",
    reference_config: str | None = None,
    override_reference_config: str | None = None,
    prewarm_s: float | None = None,
    results_root: str | Path,
    fs_origin: int = 100,
    ud_calculation_mode: str = "smoothed",
    ud_smoothing_window_s: float = 0.1,
    postprocess_method_override: str | None = None,
) -> dict[str, Path]:
    """Replay one sample from compact Stage-6 result records without retraining.

    中文说明：该入口只读取 ``best_params_and_alignment.csv`` 等记录文件来恢复最优
    参数，然后对单个样本重新执行 ``run_protocol_trial``；不会启动 Optuna 或改写训练
    结果。输出 CSV 便于 Notebook 后续重画，PNG 用于快速人工核对 final 融合曲线。
    """

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    motion_dir = _stage6_motion_record_dir(Path(results_root), motion_type)
    records = _read_stage6_record_tables(motion_dir)
    best_row = _select_stage6_best_record(
        records["best_params"],
        motion_type=motion_type,
        split=split,
        mode=mode,
        target_scope=TargetScope(target_scope).value,
        adaptive_filter=adaptive_filter,
        adaptive_data_type=adaptive_data_type,
        reference_config=reference_config,
        prewarm_s=prewarm_s,
    )
    params = _params_from_stage6_record(best_row, adaptive_filter=adaptive_filter, prewarm_s=prewarm_s)
    if postprocess_method_override:
        params = replace(params, postprocess_method=str(postprocess_method_override))
    scheme_text = _stage6_row_text(best_row, "reference_order_key")
    if reference_config:
        scheme_text = str(reference_config)
    elif not scheme_text:
        scheme_text = _stage6_row_text(best_row, "adaptive_data_type") or adaptive_data_type
    scheme_to_run = override_reference_config if override_reference_config else scheme_text
    scheme = ReferenceConfig.from_key(scheme_to_run)
    scope = TargetScope(target_scope)

    dataset = load_and_preprocess_protocol(
        signal_csv,
        ref_csv,
        fs_origin=fs_origin,
        ud_calculation_mode=ud_calculation_mode,
        ud_smoothing_window_s=ud_smoothing_window_s,
    )
    label_data_type = adaptive_data_type or scheme_text

    if override_reference_config:
        title_prefix = f"Params: {scheme_text} -> Run: {override_reference_config}"
        label = f"cross_{label_data_type}_to_{override_reference_config}_{motion_type}_{params.adaptive_filter}_{_prewarm_run_label(params.prewarm_s)}_{scope.value}"
    else:
        title_prefix = scheme.reference_order_key
        label = f"{motion_type}_{params.adaptive_filter}_{label_data_type}_{_prewarm_run_label(params.prewarm_s)}_{scope.value}"

    return _run_replay_variant(
        dataset=dataset, scheme=scheme, scope=scope, params=params,
        output_dir=out_dir, label=label, variant_name="core_enhanced_v1",
        motion_type=motion_type, title_prefix=title_prefix,
    )


def _run_replay_variant(
    *,
    dataset: ProtocolDataset,
    scheme: ReferenceConfig,
    scope: TargetScope,
    params: ProtocolTrialParams,
    output_dir: Path,
    label: str,
    variant_name: str,
    motion_type: str,
    title_prefix: str,
) -> dict[str, Path]:
    """Run one replay variant and write its HR curve CSV/PNG."""

    run = run_protocol_trial(dataset, scheme, scope, params, collect_frame=True, collect_stages=False)
    if not run.success:
        raise RuntimeError(f"Replay failed ({variant_name}): {run.reason}")
    frame = run.frame.copy()
    if frame.empty:
        raise RuntimeError(f"Replay produced no window frame ({variant_name})")

    replay_frame = _build_replay_frame(frame)
    csv_path = output_dir / f"replay_{label}_{variant_name}.csv"
    plot_path = output_dir / f"replay_{label}_{variant_name}.png"
    replay_frame.to_csv(csv_path, index=False, encoding="utf-8-sig")

    _plot_replay_hr_curves(
        plot_path,
        dataset=dataset,
        frame=replay_frame,
        motion_type=motion_type,
        scope=scope,
        scheme=scheme,
        params=params,
        title_prefix=title_prefix,
    )
    return {"plot": plot_path, "csv": csv_path}


def _stage6_motion_record_dir(results_root: Path, motion_type: str) -> Path:
    """Locate the per-motion result directory used by Stage-6 records."""

    root = results_root
    candidates = [
        root / "motion_types" / motion_type,
        root / motion_type,
        root,
    ]
    for item in candidates:
        if (item / "best_params_and_alignment.csv").exists():
            return item
    raise FileNotFoundError(f"Cannot find best_params_and_alignment.csv under {results_root}")


def _read_stage6_record_tables(motion_dir: Path) -> dict[str, pd.DataFrame]:
    """Read compact records and verify the expected Stage-6 files are present."""

    files = {
        "best_params": motion_dir / "best_params_and_alignment.csv",
        "best_metrics": motion_dir / "best_metrics.csv",
        "motion_frequency": motion_dir / "motion_frequency_and_params.csv",
        "full_report": motion_dir / "full_report.json",
    }
    missing = [path.name for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Stage-6 replay records: {', '.join(missing)}")
    # 中文说明：replay 主要依赖 best_params；其余文件在这里一起校验/读取，确保记录目录
    # 是完整的一次训练输出，后续诊断或表格汇总也能复用相同入口。
    report = json.loads(files["full_report"].read_text(encoding="utf-8"))
    _require_result_schema(report, str(files["full_report"]))
    tables = {name: pd.read_csv(path) for name, path in files.items() if name != "full_report"}
    for name, table in tables.items():
        _require_result_table_schema(table, str(files[name]))
    return tables


def _select_stage6_best_record(
    best_df: pd.DataFrame,
    *,
    motion_type: str,
    split: str,
    mode: str,
    target_scope: str,
    adaptive_filter: str,
    adaptive_data_type: str,
    reference_config: str | None,
    prewarm_s: float | None,
) -> pd.Series:
    """Select a current record matching every replay selector."""

    if best_df.empty:
        raise ValueError("best_params_and_alignment.csv is empty")
    candidates = best_df.copy()
    filters: list[tuple[str, str]] = [
        ("motion_type", motion_type),
        ("split", split),
        ("mode", mode),
        ("target_scope", target_scope),
        ("adaptive_filter", adaptive_filter),
        ("adaptive_data_type", adaptive_data_type),
    ]
    if reference_config:
        filters.append(("reference_order_key", str(reference_config)))
    for column, expected in filters:
        if not expected:
            continue
        candidates = candidates[candidates[column].astype(str) == str(expected)]
    if prewarm_s is not None:
        values = pd.to_numeric(candidates["prewarm_s"], errors="coerce")
        narrowed = candidates[np.isclose(values.astype(float), float(prewarm_s), equal_nan=False)]
        candidates = narrowed
    if candidates.empty:
        raise ValueError("No replay record matches all selectors")
    return candidates.iloc[0]


def _params_from_stage6_record(
    row: pd.Series,
    *,
    adaptive_filter: str,
    prewarm_s: float | None,
) -> ProtocolTrialParams:
    """Restore ProtocolTrialParams from a Stage-6 record using param columns first."""

    params = protocol_params_from_record(row)
    row_filter = _stage6_row_text(row, "adaptive_filter")
    fixed_filter = adaptive_filter or row_filter or params.adaptive_filter
    fixed_prewarm = float(prewarm_s) if prewarm_s is not None else float(getattr(params, "prewarm_s", 0.0))
    return replace(params, adaptive_filter=fixed_filter, prewarm_s=fixed_prewarm)


def _parse_stage6_params_json(raw: Any) -> dict[str, Any]:
    """Parse JSON/dict-like params saved in CSV without trusting column order."""

    if isinstance(raw, dict):
        return dict(raw)
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return {}
    text = str(raw).strip()
    if not text:
        return {}
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Parameters JSON must contain an object")
    return parsed


def _stage6_row_text(row: pd.Series, column: str) -> str:
    if column not in row.index:
        return ""
    value = row.get(column)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value)


def _build_replay_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize solver window columns into a stable replay CSV schema."""

    required = ["time_s", "ref_hr_bpm", "baseline_hr_bpm", "adaptive_hr_bpm", "final_hr_bpm", "final_source", "segment_label"]
    out = frame.loc[:, required].rename(columns={"ref_hr_bpm": "reference_hr_bpm"}).copy()
    out.insert(0, "result_schema_version", RESULT_SCHEMA_VERSION)
    if "fusion_reason" in frame:
        out["fusion_reason"] = frame["fusion_reason"]
    return out


def _plot_replay_hr_curves(
    out_path: Path,
    *,
    dataset: ProtocolDataset,
    frame: pd.DataFrame,
    motion_type: str,
    scope: TargetScope,
    scheme: ReferenceConfig,
    params: ProtocolTrialParams,
    title_prefix: str = "",
) -> None:
    """Draw reference/baseline/adaptive/final HR curves for one replay run."""

    plt = _prepare_matplotlib(out_path)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    if dataset.ref_time_s.size and dataset.ref_hr_bpm.size:
        ax.plot(dataset.ref_time_s, dataset.ref_hr_bpm, color="#222222", lw=1.3, label="reference HR")
    time_s = np.asarray(frame["time_s"], dtype=float)
    ax.plot(time_s, frame["baseline_hr_bpm"], color="#1f77b4", lw=1.1, marker="o", ms=3, label="baseline FFT")
    ax.plot(time_s, frame["adaptive_hr_bpm"], color="#ff7f0e", lw=1.1, marker="o", ms=3, label="adaptive")
    ax.plot(time_s, frame["final_hr_bpm"], color="#2ca02c", lw=1.6, marker="o", ms=3, label="final")
    _shade_replay_segments(ax, frame)
    prefix = title_prefix if title_prefix else scheme.reference_order_key
    metrics_title = _replay_final_metrics_title(frame)
    ax.set_title(
        f"{motion_type} | {params.adaptive_filter} | {prefix} | {scope.value} | "
        f"{_prewarm_run_label(params.prewarm_s)} | {metrics_title}"
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("HR (bpm)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _replay_final_metrics_title(frame: pd.DataFrame) -> str:
    """Return compact final-HR AAE/accuracy text for replay plot titles."""

    if "reference_hr_bpm" not in frame.columns or "final_hr_bpm" not in frame.columns:
        return "final AAE=n/a | final accuracy=n/a"
    reference = pd.to_numeric(frame["reference_hr_bpm"], errors="coerce").to_numpy(dtype=float)
    final = pd.to_numeric(frame["final_hr_bpm"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(reference) & np.isfinite(final)
    if not mask.any():
        return "final AAE=n/a | final accuracy=n/a"
    abs_err = np.abs(final[mask] - reference[mask])
    aae_bpm = float(np.mean(abs_err))
    acc_pct = _accuracy_from_abs_err(abs_err)
    return f"final AAE={aae_bpm:.2f} bpm | final accuracy={acc_pct:.1f}%"


def _shade_replay_segments(ax: Any, frame: pd.DataFrame) -> None:
    """Add light motion/recovery bands without relying on reference HR."""

    if "segment_label" not in frame or "time_s" not in frame or len(frame) == 0:
        return
    labels = frame["segment_label"].astype(str).to_numpy()
    times = np.asarray(frame["time_s"], dtype=float)
    if times.size == 1:
        half_step = 0.5
    else:
        diffs = np.diff(np.sort(times))
        finite = diffs[np.isfinite(diffs) & (diffs > 0)]
        half_step = float(np.median(finite) / 2.0) if finite.size else 0.5
    color_map = {"motion": "#f6c85f", "recovery": "#9fd3c7"}
    start = 0
    while start < labels.size:
        label = labels[start]
        end = start + 1
        while end < labels.size and labels[end] == label:
            end += 1
        if label in color_map:
            ax.axvspan(times[start] - half_step, times[end - 1] + half_step, color=color_map[label], alpha=0.18, lw=0)
        start = end


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
