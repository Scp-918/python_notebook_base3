from __future__ import annotations

import numpy as np

from ppg_hr.experimental.cascade_solver import _normalise_array
from ppg_hr.experimental.run_batch_protocol import build_output_run_name
from ppg_hr.experimental.protocol_search_space import ProtocolTrialParams
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
