from __future__ import annotations

import json
from pathlib import Path

from ppg_hr.params import RESULT_SCHEMA_VERSION

import pandas as pd

from ppg_hr.experimental import run_batch_protocol as rbp
from ppg_hr.experimental.protocol_search_space import ProtocolTrialParams
from ppg_hr.params import ReferenceConfig, TargetScope


def _stage6_result() -> rbp._ModeOptimisation:
    params = ProtocolTrialParams(hr_window_s=8, prewarm_s=0.0, Fs_Target=50, adaptive_filter="lms")
    return rbp._ModeOptimisation(
        motion_type="tiaosheng",
        target_scope=TargetScope.MOTION_ONLY,
        reference_config=ReferenceConfig(('ACC',)),
        adaptive_filter="lms",
        objective_mode="accuracy",
        data_split_mode="all_train",
        best_params=params,
        best_repeat_idx=0,
        best_trial_idx=0,
        n_trials=1,
        n_repeats=1,
        train_metrics={"success": True, "final_aae_bpm": 3.0, "final_acc_pct": 90.0},
        val_metrics={"success": True},
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
        per_group_rows=[],
        history=[],
        success=True,
        reason="",
        result_level="single_split",
        params_semantics="best_params_for_this_split",
    )


def test_motion_type_outputs_include_stage6_record_files(tmp_path: Path) -> None:
    rbp._write_motion_type_outputs(tmp_path, "tiaosheng", [_stage6_result()])

    expected = [
        "best_params_and_alignment.csv",
        "best_metrics.csv",
        "motion_frequency_and_params.csv",
        "full_report.json",
    ]
    for name in expected:
        assert (tmp_path / name).exists()

    params_df = pd.read_csv(tmp_path / "best_params_and_alignment.csv")
    assert {
        "best_params_json",
        "prewarm_s",
        "normalization_mode",
        "best_tdelay_s",
        "acc_compare_aae_bpm",
        "acc_compare_accuracy_pct",
        "acc_compare_posthoc_aae_bpm",
        "acc_compare_posthoc_accuracy_pct",
        "acc_compare_num_windows",
        "acc_compare_status",
        "acc_compare_reason",
        "param_prewarm_s",
        "param_ppg_input_transform",
        "param_global_objective_strategy",
        "param_cascade_guard_policy",
    }.issubset(params_df.columns)
    assert params_df.loc[0, "acc_compare_status"] == "same_as_original"
    assert float(params_df.loc[0, "acc_compare_aae_bpm"]) == 3.0
    assert json.loads(params_df.loc[0, "best_params_json"])["prewarm_s"] == 0.0

    metrics_df = pd.read_csv(tmp_path / "best_metrics.csv")
    assert {"baseline_aae_bpm", "adaptive_aae_bpm", "final_aae_bpm", "n_recovery_fallback"}.issubset(
        metrics_df.columns
    )
    assert float(metrics_df.loc[0, "final_aae_bpm"]) == 3.0

    freq_df = pd.read_csv(tmp_path / "motion_frequency_and_params.csv")
    assert {"motion_frequency_hz", "penalty_ref_channel", "reference_channel_ranking_summary"}.issubset(
        freq_df.columns
    )

    report = json.loads((tmp_path / "full_report.json").read_text(encoding="utf-8"))
    assert report["motion_type"] == "tiaosheng"
    assert "fusion_source_distribution" in report
    assert "git_commit_hash" in report


def test_flatten_params_for_record_roundtrips_all_protocol_fields() -> None:
    params = ProtocolTrialParams(
        prewarm_s=0.0,
        ppg_input_transform="log_absorbance",
        log_absorbance_ratio_clip=(1e-3, 1e3),
        rff_err_clip=None,
        klms_normalized_update=True,
    )

    flat = rbp.flatten_params_for_record(params)
    restored = rbp.protocol_params_from_record(flat)

    for field in params.to_dict():
        assert f"param_{field}" in flat
    assert restored.prewarm_s == 0.0
    assert restored.ppg_input_transform == "log_absorbance"
    assert restored.log_absorbance_ratio_clip == (1e-3, 1e3)
    assert restored.rff_err_clip is None
    assert restored.klms_normalized_update is True


def test_protocol_params_from_record_prefers_param_columns_over_json() -> None:
    json_params = ProtocolTrialParams(ppg_input_transform="raw_bandpass", hr_window_s=8, adaptive_filter="klms").to_dict()
    row = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "best_params_json": json.dumps(json_params),
        "param_ppg_input_transform": "log_absorbance",
        "param_memory_ms": 80,
        "adaptive_filter": "klms",
    }

    restored = rbp.protocol_params_from_record(row)

    assert restored.ppg_input_transform == "log_absorbance"
    assert restored.memory_ms == 80
    assert restored.adaptive_filter == "klms"


def test_protocol_params_rejects_incomplete_old_rff_record() -> None:
    import pytest

    old_params = ProtocolTrialParams(adaptive_filter="rff_lms").to_dict()
    old_params.pop("rff_sigma_scale")
    with pytest.raises(ValueError, match="missing"):
        rbp.protocol_params_from_record({"result_schema_version": RESULT_SCHEMA_VERSION, "best_params_json": json.dumps(old_params)})


def test_append_history_includes_params_json_and_param_columns() -> None:
    history: list[dict[str, object]] = []
    params = ProtocolTrialParams(ppg_input_transform="log_absorbance", cascade_guard_policy="rms_guard")

    rbp._append_history(
        history,
        "tiaosheng",
        TargetScope.GLOBAL,
        ReferenceConfig(('ACC',)),
        "lms",
        params,
        repeat_idx=0,
        trial_idx=1,
        objective_value=3.0,
        metrics={"success": True, "final_aae_bpm": 3.0, "final_acc_pct": 90.0},
        best_so_far=3.0,
    )

    row = history[0]
    assert isinstance(row["params"], str)
    assert json.loads(row["params"])["ppg_input_transform"] == "log_absorbance"
    assert row["param_ppg_input_transform"] == "log_absorbance"
    assert row["param_cascade_guard_policy"] == "rms_guard"


def test_best_params_all_json_can_be_rebuilt_from_saved_history_paths(tmp_path: Path) -> None:
    result = _stage6_result()
    rbp._append_history(
        result.history,
        "tiaosheng",
        TargetScope.MOTION_ONLY,
        ReferenceConfig(('ACC',)),
        "lms",
        result.best_params,
        repeat_idx=0,
        trial_idx=0,
        objective_value=3.0,
        metrics={"success": True, "final_aae_bpm": 3.0, "final_acc_pct": 90.0},
        best_so_far=3.0,
    )
    rbp._persist_mode_artifacts(tmp_path, result, checkpoint_status="done")
    result.history = []

    rbp._write_motion_type_outputs(tmp_path, "tiaosheng", [result], write_best_params_all=True)

    payload = json.loads((tmp_path / "best_params_all.json").read_text(encoding="utf-8"))
    assert payload[result.mode_key]["trial_history"][0]["trial_idx"] == 0


def test_acc_compare_fields_are_written_for_non_acc_result(tmp_path: Path) -> None:
    result = _stage6_result()
    result.reference_config = ReferenceConfig(('HF',))
    result.acc_compare_metrics = {
        "acc_compare_aae_bpm": 6.5,
        "acc_compare_accuracy_pct": 66.0,
        "acc_compare_num_windows": 9,
        "acc_compare_status": "ok",
        "acc_compare_reason": "",
    }

    rbp._write_motion_type_outputs(tmp_path, "tiaosheng", [result])

    summary = pd.read_csv(tmp_path / "mode_summary_aae.csv")
    params = pd.read_csv(tmp_path / "best_params_and_alignment.csv")
    assert summary.loc[0, "acc_compare_status"] == "ok"
    assert float(summary.loc[0, "acc_compare_aae_bpm"]) == 6.5
    assert int(summary.loc[0, "acc_compare_num_windows"]) == 9
    assert params.loc[0, "acc_compare_status"] == "ok"

    payload = json.loads((tmp_path / "best_params_all.json").read_text(encoding="utf-8"))
    assert payload[result.mode_key]["acc_compare_metrics"]["acc_compare_accuracy_pct"] == 66.0


def test_stage6_reader_rejects_unversioned_records(tmp_path: Path) -> None:
    import pytest

    rbp._write_motion_type_outputs(tmp_path, "tiaosheng", [_stage6_result()])
    report_path = tmp_path / "full_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report.pop("result_schema_version", None)
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        rbp._read_stage6_record_tables(tmp_path)


def test_resume_rejects_unversioned_checkpoint(tmp_path: Path) -> None:
    import pytest

    (tmp_path / "_checkpoint.json").write_text(json.dumps({"modes": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        rbp._load_completed_mode_results(tmp_path)


def test_current_save_resume_replay_records_share_schema(tmp_path: Path) -> None:
    result = _stage6_result()
    rbp._persist_mode_artifacts(tmp_path, result, checkpoint_status="done")
    rbp._write_motion_type_outputs(tmp_path, result.motion_type, [result])
    restored = rbp._load_completed_mode_results(tmp_path)
    assert restored[0].best_params.to_dict() == result.best_params.to_dict()
    tables = rbp._read_stage6_record_tables(tmp_path)
    assert all(table["result_schema_version"].eq(RESULT_SCHEMA_VERSION).all() for table in tables.values())
    report = json.loads((tmp_path / "full_report.json").read_text(encoding="utf-8"))
    checkpoint = json.loads((tmp_path / "_checkpoint.json").read_text(encoding="utf-8"))
    assert report["result_schema_version"] == checkpoint["result_schema_version"] == RESULT_SCHEMA_VERSION
    assert not (tmp_path / "best_params_lms.csv").exists()


def test_artifact_readers_reject_mismatched_schema(tmp_path: Path) -> None:
    import pytest

    result = _stage6_result()
    payload = rbp._mode_manifest_payload(result)
    for version in (None, RESULT_SCHEMA_VERSION + 1):
        payload["result_schema_version"] = version
        with pytest.raises(ValueError, match="schema"):
            rbp._mode_result_from_manifest_payload(payload)
    rbp._write_motion_type_outputs(tmp_path, result.motion_type, [result])
    for name in ("best_params_and_alignment.csv", "best_metrics.csv", "motion_frequency_and_params.csv"):
        path = tmp_path / name
        table = pd.read_csv(path)
        table["result_schema_version"] = RESULT_SCHEMA_VERSION + 1
        table.to_csv(path, index=False)
        with pytest.raises(ValueError, match="schema"):
            rbp._read_stage6_record_tables(tmp_path)
        table["result_schema_version"] = RESULT_SCHEMA_VERSION
        table.to_csv(path, index=False)
    (tmp_path / "_checkpoint.json").write_text(json.dumps({"result_schema_version": RESULT_SCHEMA_VERSION + 1, "modes": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        rbp._load_completed_mode_results(tmp_path)
