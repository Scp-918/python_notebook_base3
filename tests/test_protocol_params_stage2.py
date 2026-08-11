from __future__ import annotations

import numpy as np

from ppg_hr.experimental.cascade_solver import _normalise_array
from ppg_hr.experimental.protocol_search_space import ProtocolTrialParams, decode_protocol_search_space
from ppg_hr.experimental.run_batch_protocol import _VALID_FILTERS, build_output_run_name
from ppg_hr.params import CascadeScheme, ProtocolSearchParams, TargetScope


def test_fixed_trial_params_roundtrip_and_stay_out_of_search_space() -> None:
    params = ProtocolTrialParams(
        TW_F=2.5,
        normalization_mode="zscore",
        qc_policy="drop",
        rff_update_mode="nlms",
        klms_center_prune_policy="freeze_new_centers",
    )
    clone = ProtocolTrialParams(**params.to_dict())
    space = ProtocolSearchParams()

    assert clone.TW_F == 2.5
    assert clone.normalization_mode == "zscore"
    assert clone.qc_policy == "drop"
    assert clone.rff_update_mode == "nlms"
    assert clone.klms_center_prune_policy == "freeze_new_centers"
    assert "TW_F" not in space.names_for_filter("lms")
    assert "normalization_mode" not in space.names_for_filter("volterra")
    assert "qc_policy" not in space.names_for_filter("rff_lms")
    for fixed_name in (
        "smooth_win_len",
        "Rest_HR_Track_Band_BPM",
        "Rest_HR_Slew_Limit_BPM",
        "Rest_HR_Slew_Step_BPM",
        "rff_update_mode",
        "klms_max_dictionary_size",
    ):
        assert fixed_name not in space.names_for_filter("lms")


def test_stage1_search_space_uses_stabilized_candidate_lists() -> None:
    space = ProtocolSearchParams()

    assert space.options("Fs_Target") == [25, 50]
    assert space.options("LMS_Mu_Base") == [0.004, 0.006, 0.008]
    assert space.options("alpha_u") == [0.005, 0.01, 0.03, 0.05, 0.1]
    assert space.options("M2") == [2, 3]
    assert space.options("RFF_LMS_Mu_Base") == [0.001, 0.002, 0.004, 0.006]
    assert space.options("rff_D") == [50, 100, 200]
    assert space.options("rff_sigma_scale") == [0.5, 1.0, 2.0, 4.0]
    assert "rff_sigma" not in space.names_for_filter("rff_lms")
    assert space.options("klms_step_size") == [0.005, 0.01, 0.02, 0.05]
    assert space.options("klms_sigma") == [0.5, 1.0, 2.0, 5.0]
    assert space.options("klms_epsilon") == [0.005, 0.01, 0.02, 0.05, 0.1]


def test_stage1_fixed_rest_params_remain_overrideable() -> None:
    params = ProtocolTrialParams(
        smooth_win_len=7,
        Rest_HR_Track_Band_BPM=30.0,
        Rest_HR_Slew_Limit_BPM=6.0,
        Rest_HR_Slew_Step_BPM=4.0,
    )

    assert params.smooth_win_len == 7
    assert params.Rest_HR_Track_Band_BPM == 30.0
    assert params.Rest_HR_Slew_Limit_BPM == 6.0
    assert params.Rest_HR_Slew_Step_BPM == 4.0


def test_stage2_strategy_fields_roundtrip_and_affect_cache_key() -> None:
    base = ProtocolTrialParams()
    transformed = ProtocolTrialParams(ppg_input_transform="log_absorbance")
    deployment = ProtocolTrialParams(global_objective_strategy="deployment_global")
    guarded = ProtocolTrialParams(cascade_guard_policy="rms_guard")

    assert transformed.to_dict()["ppg_input_transform"] == "log_absorbance"
    assert deployment.to_dict()["global_objective_strategy"] == "deployment_global"
    assert guarded.to_dict()["cascade_guard_policy"] == "rms_guard"
    assert base.cache_key() != transformed.cache_key()
    assert base.cache_key() != deployment.cache_key()
    assert base.cache_key() != guarded.cache_key()


def test_klms_has_independent_search_parameters_and_is_batch_selectable() -> None:
    space = ProtocolSearchParams()

    names = space.names_for_filter("klms")

    assert "klms" in _VALID_FILTERS
    assert {"klms_step_size", "klms_sigma", "klms_epsilon"}.issubset(names)
    assert "LMS_Mu_Base" not in names
    assert "RFF_LMS_Mu_Base" not in names
    assert "rff_D" not in names
    assert "rff_sigma" not in names
    assert "rff_sigma_scale" not in names
    assert "alpha_u" not in names
    assert "M2" not in names
    assert space.options("klms_step_size") == [0.005, 0.01, 0.02, 0.05]
    assert space.options("klms_sigma") == [0.5, 1.0, 2.0, 5.0]
    assert space.options("klms_epsilon") == [0.005, 0.01, 0.02, 0.05, 0.1]


def test_decode_klms_search_space_writes_filter_specific_values() -> None:
    space = ProtocolSearchParams()
    idx_map = {name: 0 for name in space.names_for_filter("klms")}
    idx_map.update({"klms_step_size": 2, "klms_sigma": 3, "klms_epsilon": 1})

    params = decode_protocol_search_space(space, idx_map, adaptive_filter="klms", objective_mode="accuracy")

    assert params.adaptive_filter == "klms"
    assert params.objective_mode == "accuracy"
    assert params.klms_step_size == 0.02
    assert params.klms_sigma == 5.0
    assert params.klms_epsilon == 0.01


def test_target_scope_global_aliases_use_canonical_global_value() -> None:
    assert TargetScope("global") is TargetScope.GLOBAL
    assert TargetScope("all") is TargetScope.GLOBAL
    assert TargetScope("global_all") is TargetScope.GLOBAL
    assert TargetScope.GLOBAL.value == "global"


def test_output_run_name_contains_tw_f_label() -> None:
    name0 = build_output_run_name(
        [TargetScope.MOTION_ONLY],
        [CascadeScheme.ACC3],
        ["lms"],
        "aae",
        "all_train",
        tw_f_s=0.0,
    )
    name25 = build_output_run_name(
        ["all"],
        ["ACC3"],
        ["lms"],
        "accuracy",
        "split",
        tw_f_s=2.5,
    )

    assert name0.endswith("__TW_F0s__hr_fft")
    assert "global__ACC__lms__accuracy__split__TW_F2p5s__hr_fft" == name25


def test_output_run_name_is_isolated_by_postprocess_method() -> None:
    fft_name = build_output_run_name(
        [TargetScope.MOTION_ONLY],
        [CascadeScheme.ACC3],
        ["lms"],
        "aae",
        "split",
        tw_f_s=0.0,
        postprocess_method="fft",
    )
    ssr_name = build_output_run_name(
        [TargetScope.MOTION_ONLY],
        [CascadeScheme.ACC3],
        ["lms"],
        "aae",
        "split",
        tw_f_s=0.0,
        postprocess_method="ssr",
    )

    assert fft_name != ssr_name
    assert fft_name.endswith("__TW_F0s__hr_fft")
    assert ssr_name.endswith("__TW_F0s__hr_ssr")


def test_normalisation_modes_keep_minmax_default_and_support_zscore_none() -> None:
    values = np.asarray([1.0, 2.0, 3.0])

    np.testing.assert_allclose(_normalise_array(values), [0.0, 0.5, 1.0])
    np.testing.assert_allclose(_normalise_array(values, mode="none"), values)
    z = _normalise_array(values, mode="zscore")
    assert abs(float(np.mean(z))) < 1e-12
    assert np.isclose(float(np.std(z)), 1.0)
