from __future__ import annotations

import numpy as np

from ppg_hr.experimental.cascade_solver import _normalise_array
from ppg_hr.experimental.protocol_search_space import ProtocolTrialParams, decode_protocol_search_space
from ppg_hr.experimental.run_batch_protocol import _VALID_FILTERS, build_output_run_name
from ppg_hr.params import CascadeScheme, ProtocolSearchParams, TargetScope


def test_fixed_trial_params_roundtrip_and_stay_out_of_search_space() -> None:
    params = ProtocolTrialParams(TW_F=2.5, normalization_mode="zscore", qc_policy="drop")
    clone = ProtocolTrialParams(**params.to_dict())
    space = ProtocolSearchParams()

    assert clone.TW_F == 2.5
    assert clone.normalization_mode == "zscore"
    assert clone.qc_policy == "drop"
    assert "TW_F" not in space.names_for_filter("lms")
    assert "normalization_mode" not in space.names_for_filter("volterra")
    assert "qc_policy" not in space.names_for_filter("rff_lms")


def test_klms_has_independent_search_parameters_and_is_batch_selectable() -> None:
    space = ProtocolSearchParams()

    names = space.names_for_filter("klms")

    assert "klms" in _VALID_FILTERS
    assert {"klms_step_size", "klms_sigma", "klms_epsilon"}.issubset(names)
    assert "LMS_Mu_Base" not in names
    assert "RFF_LMS_Mu_Base" not in names
    assert "rff_D" not in names
    assert "rff_sigma" not in names
    assert "alpha_u" not in names
    assert "M2" not in names
    assert space.options("klms_step_size") == [0.01, 0.05, 0.1, 0.2, 0.5]
    assert space.options("klms_sigma") == [0.1, 0.5, 1.0, 2.0, 5.0]
    assert space.options("klms_epsilon") == [0.01, 0.05, 0.1, 0.2]


def test_decode_klms_search_space_writes_filter_specific_values() -> None:
    space = ProtocolSearchParams()
    idx_map = {name: 0 for name in space.names_for_filter("klms")}
    idx_map.update({"klms_step_size": 2, "klms_sigma": 3, "klms_epsilon": 1})

    params = decode_protocol_search_space(space, idx_map, adaptive_filter="klms", objective_mode="accuracy")

    assert params.adaptive_filter == "klms"
    assert params.objective_mode == "accuracy"
    assert params.klms_step_size == 0.1
    assert params.klms_sigma == 2.0
    assert params.klms_epsilon == 0.05


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

    assert name0.endswith("__TW_F0s")
    assert "global__ACC3__lms__accuracy__split__TW_F2p5s" == name25


def test_normalisation_modes_keep_minmax_default_and_support_zscore_none() -> None:
    values = np.asarray([1.0, 2.0, 3.0])

    np.testing.assert_allclose(_normalise_array(values), [0.0, 0.5, 1.0])
    np.testing.assert_allclose(_normalise_array(values, mode="none"), values)
    z = _normalise_array(values, mode="zscore")
    assert abs(float(np.mean(z))) < 1e-12
    assert np.isclose(float(np.std(z)), 1.0)
