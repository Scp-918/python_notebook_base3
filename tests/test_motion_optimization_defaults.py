from __future__ import annotations

import inspect
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ppg_hr.experimental.batch_pairing import SamplePair
from ppg_hr.experimental import run_batch_protocol as rbp
from ppg_hr.experimental.run_batch_protocol import (
    _build_mode_fingerprint,
    _build_split_plan,
    _derive_mode_seed,
    _decode_with_seed,
    _mode_key_for,
    _motion_types_in_canonical_order,
    run_batch_adaptive_protocol,
)
from ppg_hr.experimental.protocol_search_space import ProtocolTrialParams
from ppg_hr.experimental.protocol_search_space import default_protocol_search_space
from ppg_hr.params import CascadeScheme, MotionType, ProtocolParams, TargetScope


def _pair(motion: str, index: int) -> SamplePair:
    stem = f"subject_name_{motion}_{index}"
    return SamplePair(
        motion_id=stem,
        motion_type=motion,
        motion_index=index,
        stem=stem,
        sensor_csv=Path(f"{stem}_sensor.csv"),
        ref_csv=Path(f"{stem}_HRdata.csv"),
        subject="subject_name",
    )


def test_default_batch_contract_is_single_subject_all_train_post10_lms_200_trials() -> None:
    signature = inspect.signature(run_batch_adaptive_protocol)
    assert "subject_dir" in signature.parameters
    assert "input_dir" not in signature.parameters
    assert signature.parameters["max_iterations"].default == 200
    assert signature.parameters["num_repeats"].default == 1
    assert signature.parameters["objective_mode"].default == "posthoc_aae"
    assert signature.parameters["data_split_mode"].default == "all_train"
    assert ProtocolParams().max_iterations == 200
    assert ProtocolParams().num_repeats == 1
    assert ProtocolTrialParams().ppg_input_transform == "log_absorbance"
    assert ProtocolTrialParams().objective_mode == "posthoc_aae"


def test_all_train_uses_every_index_once_per_motion_and_canonical_order() -> None:
    grouped = {
        motion.value: [_pair(motion.value, index) for index in (4, 1, 3, 2)]
        for motion in reversed(list(MotionType))
    }

    plans = _build_split_plan(
        grouped,
        data_split_mode="all_train",
        val_groups_per_type=1,
        test_groups_per_type=1,
        random_state=7,
    )

    assert _motion_types_in_canonical_order(grouped) == [motion.value for motion in MotionType]
    for motion in MotionType:
        fold = plans[motion.value][0]
        expected = [f"subject_name_{motion.value}_{index}" for index in (1, 2, 3, 4)]
        assert fold["train"] == expected
        assert fold["val"] == []
        assert fold["test"] == expected


def test_motion_is_part_of_mode_identity_sampler_seed_and_rff_seed_namespace() -> None:
    identities = {
        _mode_key_for(motion.value, TargetScope.MOTION_POST10, CascadeScheme.ACC, "lms")
        for motion in MotionType
    }
    seeds = {
        _derive_mode_seed(
            random_state=42,
            motion_type=motion.value,
            scope=TargetScope.MOTION_POST10,
            scheme=CascadeScheme.ACC,
            adaptive_filter="lms",
            repeat_idx=0,
        )
        for motion in MotionType
    }
    assert len(identities) == 4
    assert len(seeds) == 4
    assert seeds == {
        _derive_mode_seed(
            random_state=42,
            motion_type=motion.value,
            scope=TargetScope.MOTION_POST10,
            scheme=CascadeScheme.ACC,
            adaptive_filter="lms",
            repeat_idx=0,
        )
        for motion in MotionType
    }
    space = default_protocol_search_space()
    idx_map = {name: 0 for name in space.names_for_filter("rff_lms")}
    write_params = _decode_with_seed(
        space,
        idx_map,
        adaptive_filter="rff_lms",
        objective_mode="posthoc_aae",
        delay_estimation_mode="envelope",
        mode_key=_mode_key_for("write", TargetScope.MOTION_POST10, CascadeScheme.ACC, "rff_lms"),
        repeat_idx=0,
        random_state=42,
    )
    run_params = _decode_with_seed(
        space,
        idx_map,
        adaptive_filter="rff_lms",
        objective_mode="posthoc_aae",
        delay_estimation_mode="envelope",
        mode_key=_mode_key_for("run", TargetScope.MOTION_POST10, CascadeScheme.ACC, "rff_lms"),
        repeat_idx=0,
        random_state=42,
    )
    assert write_params.rff_seed != run_params.rff_seed


def test_mode_checkpoint_fingerprint_covers_inputs_calibration_space_and_budget() -> None:
    base = _build_mode_fingerprint(
        motion_type="write",
        scope=TargetScope.MOTION_POST10,
        scheme=CascadeScheme.ACC,
        adaptive_filter="lms",
        objective_mode="posthoc_aae",
        data_split_mode="all_train",
        input_signatures={"subject_name_write_1": "csv-a"},
        calibration_signature="mat-a",
        search_space_signature="space-a",
        n_trials=200,
        n_repeats=1,
        fixed_config={"random_state": 42},
    )
    assert len(base) == 64
    assert base != _build_mode_fingerprint(
        motion_type="run",
        scope=TargetScope.MOTION_POST10,
        scheme=CascadeScheme.ACC,
        adaptive_filter="lms",
        objective_mode="posthoc_aae",
        data_split_mode="all_train",
        input_signatures={"subject_name_write_1": "csv-a"},
        calibration_signature="mat-a",
        search_space_signature="space-a",
        n_trials=200,
        n_repeats=1,
        fixed_config={"random_state": 42},
    )


def test_ud_preprocessing_marker_only_invalidates_ud_modes() -> None:
    marker = {"ud_preprocessing": "calibrated_formula_qc_interpolated_no_bandpass_v1"}

    assert rbp._ud_preprocessing_fingerprint_fields(CascadeScheme.UD2) == marker
    assert rbp._ud_preprocessing_fingerprint_fields(CascadeScheme.ACC_UD2) == marker
    assert rbp._ud_preprocessing_fingerprint_fields(CascadeScheme.ACC) == {}
    assert rbp._ud_preprocessing_fingerprint_fields(CascadeScheme.HF2) == {}


def test_one_all_train_objective_receives_all_indices_of_one_motion(monkeypatch) -> None:
    dataset_ids = [f"subject_name_write_{index}" for index in (1, 2, 3, 4)]
    datasets = {item: SimpleNamespace() for item in dataset_ids}
    calls: list[tuple[str, ...]] = []

    def fake_evaluate(dataset_map, *args, **kwargs):
        calls.append(tuple(dataset_map))
        return {
            "success": True,
            "reason": "",
            "posthoc_final_aae_bpm": 2.0,
            "final_aae_bpm": 3.0,
        }, [], {}

    monkeypatch.setattr(rbp, "optuna", None)
    monkeypatch.setattr(rbp, "TPESampler", None)
    monkeypatch.setattr(rbp, "_evaluate_dataset_map", fake_evaluate)

    rbp._optimise_group_mode(
        motion_type="write",
        train_sets=datasets,
        val_sets={},
        test_sets=datasets,
        scope=TargetScope.MOTION_POST10,
        scheme=CascadeScheme.ACC,
        adaptive_filter="lms",
        objective_mode="posthoc_aae",
        data_split_mode="all_train",
        delay_estimation_mode="envelope",
        cfg=ProtocolParams(max_iterations=1, num_repeats=1),
        space=default_protocol_search_space(),
        trial_param_overrides=None,
        n_trials=1,
        n_repeats=1,
        trial_cache=OrderedDict(),
        penalty_value=999.0,
        mode_idx=1,
        mode_total=1,
        random_state=42,
        on_progress=lambda _info: None,
    )

    assert calls
    assert all(call == tuple(dataset_ids) for call in calls if call)


def test_batch_creates_one_independent_all_train_mode_per_motion(tmp_path, monkeypatch) -> None:
    subject_dir = tmp_path / "subject_name"
    subject_dir.mkdir()
    pairs = [_pair(motion.value, index) for motion in MotionType for index in (1, 2, 3, 4)]
    datasets = {pair.motion_id: SimpleNamespace(accx=np.zeros(2), accy=np.zeros(2), accz=np.zeros(2), fs=100) for pair in pairs}
    calls: list[tuple[str, tuple[str, ...], int, int, str, str]] = []
    finalized: list[str] = []

    monkeypatch.setattr(rbp, "load_subject_calibration", lambda *a, **k: SimpleNamespace(source_sha256="mat"))
    monkeypatch.setattr(
        rbp,
        "discover_sample_pairs_with_unpaired",
        lambda _root: SimpleNamespace(pairs=pairs, unpaired=[]),
    )
    monkeypatch.setattr(rbp, "quality_filter_sample", lambda *a, **k: SimpleNamespace(is_good=True))
    monkeypatch.setattr(rbp, "write_qc_tables", lambda *a, **k: {})
    monkeypatch.setattr(
        rbp,
        "load_and_preprocess_protocol",
        lambda sensor, *a, **k: datasets[next(pair.motion_id for pair in pairs if pair.sensor_csv == sensor)],
    )
    monkeypatch.setattr(rbp, "detect_activity_segments", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(rbp, "plot_signal_figures", lambda *a, **k: {})

    def fake_optimise(**kwargs):
        calls.append(
            (
                kwargs["motion_type"],
                tuple(kwargs["train_sets"]),
                kwargs["n_trials"],
                kwargs["n_repeats"],
                kwargs["scope"].value,
                kwargs["objective_mode"],
            )
        )
        return rbp._ModeOptimisation(
            motion_type=kwargs["motion_type"],
            target_scope=kwargs["scope"],
            cascade_scheme=kwargs["scheme"],
            adaptive_filter=kwargs["adaptive_filter"],
            objective_mode=kwargs["objective_mode"],
            data_split_mode=kwargs["data_split_mode"],
            best_params=ProtocolTrialParams(),
            best_repeat_idx=0,
            best_trial_idx=0,
            n_trials=kwargs["n_trials"],
            n_repeats=kwargs["n_repeats"],
            train_metrics={"success": True},
            val_metrics={},
            test_metrics={"success": True},
            per_group_rows=[],
            history=[],
            success=True,
            reason="",
        )

    monkeypatch.setattr(rbp, "_optimise_group_mode", fake_optimise)
    monkeypatch.setattr(
        rbp,
        "_finalize_completed_mode",
        lambda _dir, motion, *_args, **_kwargs: finalized.append(motion),
    )
    monkeypatch.setattr(rbp, "_write_motion_type_outputs", lambda *a, **k: None)
    monkeypatch.setattr(rbp, "_plot_bayes_curves", lambda directory, *a, **k: Path(directory) / "bayes.csv")
    monkeypatch.setattr(rbp, "_write_final_summary", lambda *a, **k: {})
    monkeypatch.setattr(rbp, "_write_batch_summary", lambda path, rows: Path(path))
    monkeypatch.setattr(rbp, "clear_all_caches", lambda _dataset: None)

    result = run_batch_adaptive_protocol(
        subject_dir=subject_dir,
        output_root=tmp_path / "outputs" / "four_motion",
        cascade_schemes=[CascadeScheme.ACC],
        verbose=False,
    )

    assert [call[0] for call in calls] == [motion.value for motion in MotionType]
    assert all(len(call[1]) == 4 for call in calls)
    assert all(call[2:] == (200, 1, "motion_post10", "posthoc_aae") for call in calls)
    assert finalized == [motion.value for motion in MotionType]
    assert list(result.mode_results) == [motion.value for motion in MotionType]
