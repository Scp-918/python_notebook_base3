"""Top-level batch runner for the adaptive protocol experiment.

中文说明：
本模块是 Notebook 和脚本调用的“总调度器”。它按顺序完成：
文件配对 -> QC -> 预处理 -> 14 组协议优化 -> CSV/JSON/PNG 输出。
稳定旧 API 的同时，这里也负责把进度信息包装成 Notebook 方便显示的 dict。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..params import CascadeScheme, ProtocolParams, TargetScope
from .batch_pairing import SamplePair, discover_sample_pairs_with_unpaired
from .preprocess_protocol import load_and_preprocess_protocol
from .protocol_optimizer import ProtocolModeResult, optimise_all_protocol_modes
from .protocol_outputs import (
    SampleOutputPaths,
    write_batch_summary,
    write_metric_matrix_tables,
    write_qc_tables,
    write_sample_outputs,
)
from .protocol_search_space import ProtocolSearchSpace, default_protocol_search_space
from .qc import QcResult, quality_filter_sample

__all__ = ["BatchProtocolResult", "run_batch_adaptive_protocol"]


@dataclass
class BatchProtocolResult:
    """Return payload from :func:`run_batch_adaptive_protocol`."""

    input_dir: Path
    csv_out_dir: Path
    report_out_dir: Path
    fig_out_dir: Path
    sample_outputs: dict[str, SampleOutputPaths] = field(default_factory=dict)
    mode_results: dict[str, list[ProtocolModeResult]] = field(default_factory=dict)
    good_samples: list[QcResult] = field(default_factory=list)
    bad_samples: list[QcResult] = field(default_factory=list)
    pairs: list[SamplePair] = field(default_factory=list)
    qc_tables: dict[str, Path] = field(default_factory=dict)
    batch_summary_csv: Path | None = None
    metric_matrix_tables: dict[str, Path] = field(default_factory=dict)


def run_batch_adaptive_protocol(
    *,
    input_dir: str | Path,
    csv_out_dir: str | Path,
    report_out_dir: str | Path,
    fig_out_dir: str | Path | None = None,
    max_iterations: int = 350,
    num_repeats: int = 3,
    random_state: int = 42,
    num_seed_points: int = 10,
    fs_origin: int = 100,
    penalty_value: float = 999.0,
    parallel_repeats: int = 1,
    n_jobs: int | None = None,
    debug_mode: bool = False,
    search_space: ProtocolSearchSpace | None = None,
    target_scopes: list[TargetScope] | None = None,
    cascade_schemes: list[CascadeScheme] | None = None,
    verbose: bool = True,
    on_log: Callable[[str], None] | None = None,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> BatchProtocolResult:
    """Run pair discovery, QC, 14-mode optimisation, and output generation."""

    input_path = Path(input_dir).resolve()
    csv_dir = Path(csv_out_dir).resolve()
    report_dir = Path(report_out_dir).resolve()
    fig_dir = Path(fig_out_dir).resolve() if fig_out_dir is not None else report_dir
    csv_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

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
    _log(f"Discovered {len(discovery.pairs)} paired samples and {len(discovery.unpaired)} unpaired CSVs")

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
            ref_csv=pair.ref_csv,
        )
        qc_by_stem[pair.stem] = qc
        if qc.is_good:
            good_qc.append(qc)
            good_pairs.append(pair)
        else:
            bad_qc.append(qc)
            _log(f"Skipping bad sample {pair.stem}: {qc.reason}")

    qc_tables = write_qc_tables(csv_dir, good=good_qc, bad=bad_qc, unpaired=discovery.unpaired)

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

    outputs: dict[str, SampleOutputPaths] = {}
    mode_results: dict[str, list[ProtocolModeResult]] = {}
    summary_rows: list[dict[str, Any]] = []

    for pair in discovery.pairs:
        qc = qc_by_stem[pair.stem]
        if not qc.is_good:
            summary_rows.append(
                {
                    "sample": pair.stem,
                    "status": "skipped_qc",
                    "reason": qc.reason,
                }
            )

    for unpaired in discovery.unpaired:
        summary_rows.append(
            {
                "sample": unpaired.file_path.stem.removesuffix("_ref"),
                "status": "skipped_unpaired",
                "reason": unpaired.reason,
            }
        )

    for sample_idx, pair in enumerate(good_pairs, start=1):
        expected_output = csv_dir / f"adaptive_results_{pair.motion_id}.csv"
        _progress(
            {
                "stage": "sample",
                "sample_idx": sample_idx,
                "sample_total": len(good_pairs),
                "current": sample_idx,
                "total": len(good_pairs),
                "sample": pair.stem,
                "output_path": str(expected_output),
            }
        )
        try:
            _log(f"[{sample_idx}/{len(good_pairs)}] Loading {pair.stem}")
            dataset = load_and_preprocess_protocol(pair.sensor_csv, pair.ref_csv, fs_origin=fs_origin)
            _log(f"[{sample_idx}/{len(good_pairs)}] Optimising {pair.stem}")

            def _sample_progress(info: dict[str, Any]) -> None:
                payload = {
                    **info,
                    "sample_idx": sample_idx,
                    "sample_total": len(good_pairs),
                    "sample": pair.stem,
                    "output_path": str(expected_output),
                }
                _progress(payload)

            results = optimise_all_protocol_modes(
                dataset,
                config=cfg,
                space=space,
                target_scopes=target_scopes,
                cascade_schemes=cascade_schemes,
                on_progress=_sample_progress,
            )
            _log(f"[{sample_idx}/{len(good_pairs)}] Writing outputs for {pair.stem}")
            paths = write_sample_outputs(
                pair.stem,
                results,
                csv_out_dir=csv_dir,
                report_out_dir=report_dir,
                fig_out_dir=fig_dir,
                dataset=dataset,
            )
            outputs[pair.stem] = paths
            mode_results[pair.stem] = results
            failed_modes = [r for r in results if not r.best_run.success]
            status = "ok" if not failed_modes else "partial_failed"
            reason = "; ".join(f"{r.mode_key}: {r.best_run.reason}" for r in failed_modes[:3])
            summary_rows.append(
                {
                    "sample": pair.stem,
                    "status": status,
                    "reason": reason,
                    "result_csv": str(paths.result_csv),
                    "report_json": str(paths.report_json),
                    "signal_png": str(paths.signal_png or ""),
                    "bayes_motion_png": str(paths.bayes_motion_png or ""),
                    "bayes_motion_recovery_png": str(paths.bayes_motion_recovery_png or ""),
                    "motion_png": str(paths.motion_png),
                    "motion_recovery_png": str(paths.motion_recovery_png),
                }
            )
            _progress(
                {
                    "stage": "output",
                    "sample_idx": sample_idx,
                    "sample_total": len(good_pairs),
                    "sample": pair.stem,
                    "result_csv": str(paths.result_csv),
                    "report_json": str(paths.report_json),
                    "signal_png": str(paths.signal_png or ""),
                    "bayes_motion_png": str(paths.bayes_motion_png or ""),
                    "bayes_motion_recovery_png": str(paths.bayes_motion_recovery_png or ""),
                    "motion_png": str(paths.motion_png),
                    "motion_recovery_png": str(paths.motion_recovery_png),
                    "output_path": str(paths.result_csv),
                }
            )
        except Exception as exc:
            _log(f"Skipping {pair.stem}: {exc}")
            summary_rows.append(
                {
                    "sample": pair.stem,
                    "status": "failed",
                    "reason": str(exc),
                }
            )

    metric_tables = write_metric_matrix_tables(csv_dir.parent / "metric_matrix_tables", mode_results)
    summary_csv = write_batch_summary(csv_dir / "batch_summary.csv", summary_rows)
    return BatchProtocolResult(
        input_dir=input_path,
        csv_out_dir=csv_dir,
        report_out_dir=report_dir,
        fig_out_dir=fig_dir,
        sample_outputs=outputs,
        mode_results=mode_results,
        good_samples=good_qc,
        bad_samples=bad_qc,
        pairs=discovery.pairs,
        qc_tables=qc_tables,
        batch_summary_csv=summary_csv,
        metric_matrix_tables=metric_tables,
    )
