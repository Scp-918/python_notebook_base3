from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ppg_hr.experimental import run_batch_protocol as rbp
from ppg_hr.experimental.protocol_search_space import ProtocolTrialParams
from ppg_hr.params import CascadeScheme, TargetScope


def _history_row() -> dict[str, object]:
    return {
        "motion_type": "tiaosheng",
        "target_scope": "motion_only",
        "cascade_scheme": "acc3",
        "adaptive_filter": "lms",
        "repeat_idx": 0,
        "trial_idx": 0,
        "objective_value": 3.0,
        "aae_bpm": 3.0,
        "accuracy_pct": 90.0,
        "final_aae_bpm": 3.0,
        "final_acc_pct": 90.0,
        "best_so_far": 3.0,
        "success": True,
        "reason": "",
        "params": "{}",
        "param_TW": 8,
    }


def _single_split_result() -> rbp._ModeOptimisation:
    params = ProtocolTrialParams(TW=8, TW_F=1.5, Fs_Target=50, adaptive_filter="lms")
    return rbp._ModeOptimisation(
        motion_type="tiaosheng",
        target_scope=TargetScope.MOTION_ONLY,
        cascade_scheme=CascadeScheme.ACC3,
        adaptive_filter="lms",
        objective_mode="accuracy",
        data_split_mode="split",
        best_params=params,
        best_repeat_idx=0,
        best_trial_idx=0,
        n_trials=1,
        n_repeats=1,
        train_metrics={"success": True, "final_aae_bpm": 3.5, "final_acc_pct": 88.0},
        val_metrics={"success": True, "final_aae_bpm": 3.0, "final_acc_pct": 90.0},
        test_metrics={
            "success": True,
            "reason": "",
            "baseline_aae_bpm": 9.0,
            "adaptive_aae_bpm": 5.0,
            "final_aae_bpm": 3.0,
            "baseline_acc_pct": 40.0,
            "adaptive_acc_pct": 70.0,
            "final_acc_pct": 90.0,
            "posthoc_baseline_aae_bpm": 8.0,
            "posthoc_adaptive_aae_bpm": 4.0,
            "posthoc_final_aae_bpm": 2.0,
            "posthoc_baseline_acc_pct": 50.0,
            "posthoc_adaptive_acc_pct": 80.0,
            "posthoc_final_acc_pct": 95.0,
            "time_bias_after_s": 1.0,
            "num_windows": 12,
        },
        per_group_rows=[{"motion_type": "tiaosheng", "group_id": "g1", "split": "test"}],
        history=[_history_row()],
        success=True,
        reason="",
        result_level="single_split",
        params_semantics="best_params_for_this_split",
    )


def test_mode_done_progress_payload_reports_current_and_acc3_metrics() -> None:
    result = _single_split_result()
    result.cascade_scheme = CascadeScheme.HF2
    result.acc3_compare_metrics = {
        "acc3_compare_aae_bpm": 6.5,
        "acc3_compare_accuracy_pct": 66.0,
        "acc3_compare_num_windows": 9,
        "acc3_compare_status": "ok",
        "acc3_compare_reason": "",
    }

    payload = rbp._mode_done_progress_payload(result, mode_idx=2, mode_total=7)

    assert payload["stage"] == "optimization_mode_done"
    assert payload["motion_type"] == "tiaosheng"
    assert payload["mode_idx"] == 2
    assert payload["mode_total"] == 7
    assert payload["cascade_scheme"] == "HF2"
    assert payload["current_final_aae_bpm"] == 3.0
    assert payload["current_final_acc_pct"] == 90.0
    assert payload["acc3_compare_aae_bpm"] == 6.5
    assert payload["acc3_compare_accuracy_pct"] == 66.0
    assert payload["acc3_compare_status"] == "ok"
    assert payload["resumed"] is False


def _logo_fold_result(fold_id: int, heldout: str, aae: float) -> rbp._ModeOptimisation:
    params = ProtocolTrialParams(TW=6 + fold_id, TW_F=1.5, Fs_Target=50, adaptive_filter="lms")
    arrays = {
        "ref_hr_bpm": np.asarray([70.0, 72.0]),
        "baseline_hr_bpm": np.asarray([71.0, 73.0]),
        "adaptive_hr_bpm": np.asarray([70.0 + aae, 72.0 + aae]),
        "adaptive_abs_err_bpm": np.asarray([aae, aae]),
        "baseline_abs_err_bpm": np.asarray([1.0, 1.0]),
        "filtered_mask": np.asarray([True, True]),
    }
    return rbp._ModeOptimisation(
        motion_type="tiaosheng",
        target_scope=TargetScope.MOTION_ONLY,
        cascade_scheme=CascadeScheme.ACC3,
        adaptive_filter="lms",
        objective_mode="aae",
        data_split_mode="leave_one_group_out",
        best_params=params,
        best_repeat_idx=0,
        best_trial_idx=fold_id,
        n_trials=1,
        n_repeats=1,
        train_metrics={"success": True, "adaptive_aae_bpm": aae, "adaptive_acc_pct": 100.0},
        val_metrics={"success": False},
        test_metrics={
            "success": True,
            "adaptive_aae_bpm": aae,
            "adaptive_acc_pct": 100.0,
            "final_aae_bpm": aae,
            "final_acc_pct": 100.0,
        },
        per_group_rows=[{"motion_type": "tiaosheng", "group_id": heldout, "split": "test"}],
        history=[{**_history_row(), "fold_id": fold_id, "heldout_group_id": heldout}],
        success=True,
        reason="",
        fold_id=fold_id,
        heldout_group_id=heldout,
        train_group_ids=["g0", "g1"],
        test_group_id=heldout,
        metric_arrays_by_split={"test": arrays, "train": arrays},
        result_level="fold",
        params_semantics="fold_best_params",
    )


def test_load_completed_mode_results_restores_lightweight_results(tmp_path: Path) -> None:
    motion_dir = tmp_path / "tiaosheng"
    motion_dir.mkdir()
    result = _single_split_result()

    rbp._persist_mode_artifacts(motion_dir, result, checkpoint_status="done")

    restored = rbp._load_completed_mode_results(motion_dir)

    assert [item.mode_key for item in restored] == [result.mode_key]
    assert restored[0].history == []
    assert restored[0].history_path.endswith("history.csv")
    rbp._write_motion_type_outputs(motion_dir, "tiaosheng", restored)
    assert (motion_dir / "best_metrics.csv").exists()
    per_group = pd.read_csv(motion_dir / "per_group_aae.csv")
    assert per_group["group_id"].tolist() == ["g1"]


def test_load_completed_mode_results_ignores_modes_without_done_checkpoint(tmp_path: Path) -> None:
    motion_dir = tmp_path / "tiaosheng"
    motion_dir.mkdir()
    result = _single_split_result()

    manifest_path = rbp._write_mode_manifest(motion_dir, result)
    checkpoint_path = motion_dir / "_checkpoint.json"
    checkpoint_path.write_text(
        json.dumps(
            {
                "motion_type": "tiaosheng",
                "modes": {
                    result.mode_key: {
                        "status": "running",
                        "mode_manifest_path": str(manifest_path),
                    }
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    restored = rbp._load_completed_mode_results(motion_dir)

    assert restored == []


def test_finalize_completed_split_mode_updates_outputs_and_clears_heavy_fields(tmp_path: Path) -> None:
    motion_dir = tmp_path / "tiaosheng"
    motion_dir.mkdir()
    result = _single_split_result()
    result.metric_arrays_by_split = {
        "test": {
            "ref_hr_bpm": np.asarray([70.0, 72.0]),
            "baseline_hr_bpm": np.asarray([71.0, 73.0]),
            "adaptive_hr_bpm": np.asarray([70.5, 72.5]),
            "adaptive_abs_err_bpm": np.asarray([0.5, 0.5]),
            "baseline_abs_err_bpm": np.asarray([1.0, 1.0]),
            "filtered_mask": np.asarray([True, True]),
        }
    }

    rbp._finalize_completed_mode(
        motion_dir,
        "tiaosheng",
        [result],
        result,
        write_best_params_all=False,
    )

    assert (motion_dir / "best_metrics.csv").exists()
    assert (motion_dir / "bayes_curve_data.csv").exists()
    assert not (motion_dir / "best_params_all.json").exists()
    checkpoint = json.loads((motion_dir / "_checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["modes"][result.mode_key]["status"] == "done"
    assert result.history == []
    assert result.metric_arrays_by_split == {}
    assert result.history_path.endswith("history.csv")


def test_finalize_completed_logo_mode_writes_fold_files_and_clears_fold_heavy_fields(tmp_path: Path) -> None:
    motion_dir = tmp_path / "tiaosheng"
    motion_dir.mkdir()
    fold0 = _logo_fold_result(0, "g0", 2.0)
    fold1 = _logo_fold_result(1, "g1", 1.0)
    result = rbp._aggregate_logo_fold_results(
        motion_type="tiaosheng",
        scope=TargetScope.MOTION_ONLY,
        scheme=CascadeScheme.ACC3,
        adaptive_filter="lms",
        objective_mode="aae",
        data_split_mode="leave_one_group_out",
        fold_results=[fold0, fold1],
        n_trials=1,
        n_repeats=1,
    )

    rbp._finalize_completed_mode(
        motion_dir,
        "tiaosheng",
        [result],
        result,
        write_best_params_all=False,
    )

    mode_dir = motion_dir / "_modes" / result.mode_key
    assert (mode_dir / "fold_0_history.csv").exists()
    assert (mode_dir / "fold_1_history.csv").exists()
    assert (mode_dir / "fold_0_per_group.csv").exists()
    assert (mode_dir / "fold_1_per_group.csv").exists()
    checkpoint = json.loads((motion_dir / "_checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["modes"][result.mode_key]["status"] == "done"
    assert fold0.history == []
    assert fold0.per_group_rows == []
    assert fold0.metric_arrays_by_split == {}
    assert fold1.history == []
    assert fold1.per_group_rows == []
    assert fold1.metric_arrays_by_split == {}
    assert result.history == []
    assert result.per_group_rows == []


def test_aggregate_logo_fold_results_keeps_non_acc3_compare_metrics() -> None:
    fold0 = _logo_fold_result(0, "g0", 2.0)
    fold1 = _logo_fold_result(1, "g1", 1.0)
    for fold in (fold0, fold1):
        fold.cascade_scheme = CascadeScheme.HF2
        fold.acc3_compare_metrics = {
            "acc3_compare_aae_bpm": 1.0,
            "acc3_compare_accuracy_pct": 100.0,
            "acc3_compare_num_windows": 2,
            "acc3_compare_status": "ok",
            "acc3_compare_reason": "",
        }
        fold.acc3_compare_arrays = fold.metric_arrays_by_split["test"]

    result = rbp._aggregate_logo_fold_results(
        motion_type="tiaosheng",
        scope=TargetScope.MOTION_ONLY,
        scheme=CascadeScheme.HF2,
        adaptive_filter="lms",
        objective_mode="aae",
        data_split_mode="leave_one_group_out",
        fold_results=[fold0, fold1],
        n_trials=1,
        n_repeats=1,
    )

    assert result.acc3_compare_metrics["acc3_compare_status"] == "ok"
    assert result.acc3_compare_metrics["acc3_compare_num_windows"] == 4


def test_plot_bayes_curves_uses_saved_history_after_history_is_cleared(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    motion_dir = tmp_path / "tiaosheng"
    motion_dir.mkdir()
    result = _single_split_result()
    rbp._persist_mode_artifacts(motion_dir, result, checkpoint_status="done")
    result.history = []

    class _FakeAxis:
        def __init__(self) -> None:
            self.plot_calls: list[tuple[list[float], list[float]]] = []
            self.text_calls: list[str] = []

        def grid(self, *args: object, **kwargs: object) -> None:
            return None

        def set_title(self, *args: object, **kwargs: object) -> None:
            return None

        def set_xlabel(self, *args: object, **kwargs: object) -> None:
            return None

        def set_ylabel(self, *args: object, **kwargs: object) -> None:
            return None

        def plot(self, x: np.ndarray, y: np.ndarray, *args: object, **kwargs: object) -> None:
            self.plot_calls.append((x.tolist(), y.tolist()))

        def legend(self, *args: object, **kwargs: object) -> None:
            return None

        def text(self, *args: object, **kwargs: object) -> None:
            if len(args) >= 3 and isinstance(args[2], str):
                self.text_calls.append(args[2])

        @property
        def transAxes(self) -> object:
            return object()

        def axis(self, *args: object, **kwargs: object) -> None:
            return None

    class _FakeFigure:
        def suptitle(self, *args: object, **kwargs: object) -> None:
            return None

        def tight_layout(self, *args: object, **kwargs: object) -> None:
            return None

        def savefig(self, path: Path, dpi: int = 150) -> None:
            Path(path).write_bytes(b"fake-png")

    class _FakePlt:
        def __init__(self) -> None:
            self.axes = np.asarray([[_FakeAxis()]])

        def subplots(self, *args: object, **kwargs: object) -> tuple[_FakeFigure, np.ndarray]:
            return _FakeFigure(), self.axes

        def close(self, fig: object) -> None:
            return None

    fake_plt = _FakePlt()
    monkeypatch.setattr(rbp, "_prepare_matplotlib", lambda _motion_dir: fake_plt)

    rbp._plot_bayes_curves(motion_dir, "tiaosheng", [result], "accuracy")

    axis = fake_plt.axes.ravel()[0]
    assert len(axis.plot_calls) == 2
    assert axis.plot_calls[0][0] == [0.0]
    assert all("no trial history" not in text for text in axis.text_calls)


def test_load_history_rows_from_empty_file_returns_empty_list(tmp_path: Path) -> None:
    empty_csv = tmp_path / "empty_history.csv"
    empty_csv.write_text("", encoding="utf-8")

    rows = rbp._load_history_rows_from_path(str(empty_csv))

    assert rows == []


def test_failed_mode_optimisation_returns_checkpointable_result() -> None:
    result = rbp._failed_mode_optimisation(
        motion_type="tiaosheng",
        scope=TargetScope.MOTION_ONLY,
        scheme=CascadeScheme.HF2,
        adaptive_filter="lms",
        objective_mode="aae",
        data_split_mode="leave_one_group_out",
        delay_estimation_mode="envelope",
        space=rbp.default_protocol_search_space(),
        n_trials=1,
        n_repeats=1,
        reason="not enough groups",
    )

    assert isinstance(result, rbp._ModeOptimisation)
    assert result.success is False
    assert result.acc3_compare_metrics["acc3_compare_status"] == "failed"
    assert result.acc3_compare_metrics["acc3_compare_reason"] == "not enough groups"


def test_run_batch_adaptive_protocol_skips_done_modes_on_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_root = tmp_path / "outputs" / "resume_case"
    pair = SimpleNamespace(
        sensor_csv=input_dir / "multi_tiaosheng1.csv",
        ref_csv=input_dir / "multi_tiaosheng1_ref.csv",
        motion_type="tiaosheng",
        motion_index=1,
        motion_id="tiaosheng1",
        stem="multi_tiaosheng1",
    )
    dataset = SimpleNamespace(_trial_base_cache={}, _global_tdelay_cache={})
    calls: list[str] = []
    first_progress: list[dict[str, object]] = []
    second_progress: list[dict[str, object]] = []

    monkeypatch.setattr(
        rbp,
        "discover_sample_pairs_with_unpaired",
        lambda _input_dir: SimpleNamespace(pairs=[pair], unpaired=[]),
    )
    monkeypatch.setattr(rbp, "quality_filter_sample", lambda *args, **kwargs: SimpleNamespace(is_good=True))
    monkeypatch.setattr(rbp, "write_qc_tables", lambda *args, **kwargs: {})
    monkeypatch.setattr(rbp, "load_and_preprocess_protocol", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(rbp, "detect_activity_segments", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(rbp, "plot_signal_figures", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        rbp,
        "_build_split_plan",
        lambda *args, **kwargs: {
            "tiaosheng": [
                {
                    "status": "ok",
                    "fold_id": 0,
                    "train": ["tiaosheng1"],
                    "val": [],
                    "test": ["tiaosheng1"],
                    "heldout_group_id": "",
                }
            ]
        },
    )

    def _fake_optimise_group_mode(**kwargs: object) -> rbp._ModeOptimisation:
        calls.append("called")
        return _single_split_result()

    monkeypatch.setattr(rbp, "_optimise_group_mode", _fake_optimise_group_mode)
    monkeypatch.setattr(rbp, "_write_final_summary", lambda *args, **kwargs: {})
    monkeypatch.setattr(rbp, "_write_batch_summary", lambda path, rows: Path(path))
    monkeypatch.setattr(rbp, "clear_all_caches", lambda _dataset: None)
    monkeypatch.setattr(rbp, "clear_trial_heavy_caches", lambda _dataset: None)

    first = rbp.run_batch_adaptive_protocol(
        input_dir=input_dir,
        output_root=output_root,
        target_scopes=[TargetScope.MOTION_ONLY],
        cascade_schemes=[CascadeScheme.ACC3],
        adaptive_filters=["lms"],
        data_split_mode="split",
        verbose=False,
        progress_callback=lambda info: first_progress.append(info),
    )
    second = rbp.run_batch_adaptive_protocol(
        input_dir=input_dir,
        output_root=output_root,
        target_scopes=[TargetScope.MOTION_ONLY],
        cascade_schemes=[CascadeScheme.ACC3],
        adaptive_filters=["lms"],
        data_split_mode="split",
        verbose=False,
        progress_callback=lambda info: second_progress.append(info),
    )

    assert len(calls) == 1
    assert "tiaosheng" in first.mode_results
    assert "tiaosheng" in second.mode_results
    assert second.mode_results["tiaosheng"][0].history == []
    first_done = [item for item in first_progress if item.get("stage") == "optimization_mode_done"]
    second_done = [item for item in second_progress if item.get("stage") == "optimization_mode_done"]
    assert len(first_done) == 1
    assert len(second_done) == 1
    assert first_done[0]["resumed"] is False
    assert second_done[0]["resumed"] is True
    assert second_done[0]["current_final_aae_bpm"] == 3.0
    assert second_done[0]["acc3_compare_status"] == "same_as_original"


def test_run_batch_adaptive_protocol_emits_mode_done_after_checkpoint_done(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_root = tmp_path / "outputs" / "checkpoint_order"
    pair = SimpleNamespace(
        sensor_csv=input_dir / "multi_tiaosheng1.csv",
        ref_csv=input_dir / "multi_tiaosheng1_ref.csv",
        motion_type="tiaosheng",
        motion_index=1,
        motion_id="tiaosheng1",
        stem="multi_tiaosheng1",
    )
    dataset = SimpleNamespace(_trial_base_cache={}, _global_tdelay_cache={})
    checkpoint_statuses: list[str] = []

    monkeypatch.setattr(
        rbp,
        "discover_sample_pairs_with_unpaired",
        lambda _input_dir: SimpleNamespace(pairs=[pair], unpaired=[]),
    )
    monkeypatch.setattr(rbp, "quality_filter_sample", lambda *args, **kwargs: SimpleNamespace(is_good=True))
    monkeypatch.setattr(rbp, "write_qc_tables", lambda *args, **kwargs: {})
    monkeypatch.setattr(rbp, "load_and_preprocess_protocol", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(rbp, "detect_activity_segments", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(rbp, "plot_signal_figures", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        rbp,
        "_build_split_plan",
        lambda *args, **kwargs: {
            "tiaosheng": [
                {
                    "status": "ok",
                    "fold_id": 0,
                    "train": ["tiaosheng1"],
                    "val": [],
                    "test": ["tiaosheng1"],
                    "heldout_group_id": "",
                }
            ]
        },
    )
    monkeypatch.setattr(rbp, "_optimise_group_mode", lambda **kwargs: _single_split_result())
    monkeypatch.setattr(rbp, "_write_final_summary", lambda *args, **kwargs: {})
    monkeypatch.setattr(rbp, "_write_batch_summary", lambda path, rows: Path(path))
    monkeypatch.setattr(rbp, "clear_all_caches", lambda _dataset: None)
    monkeypatch.setattr(rbp, "clear_trial_heavy_caches", lambda _dataset: None)

    def _progress(info: dict[str, object]) -> None:
        if info.get("stage") != "optimization_mode_done":
            return
        checkpoint = json.loads(
            (output_root / "motion_types" / "tiaosheng" / "_checkpoint.json").read_text(encoding="utf-8")
        )
        mode_key = "motion_only__ACC3__lms"
        checkpoint_statuses.append(checkpoint["modes"][mode_key]["status"])

    rbp.run_batch_adaptive_protocol(
        input_dir=input_dir,
        output_root=output_root,
        target_scopes=[TargetScope.MOTION_ONLY],
        cascade_schemes=[CascadeScheme.ACC3],
        adaptive_filters=["lms"],
        data_split_mode="split",
        verbose=False,
        progress_callback=_progress,
    )

    assert checkpoint_statuses == ["done"]
