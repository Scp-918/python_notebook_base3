import json

import pytest

from ppg_hr.params import ProtocolTrialParams, core_preset_metadata, resolve_core_config


def test_frozen_core_roundtrip_and_approved_experiment_axes() -> None:
    params = resolve_core_config({"Fs_Target": 25, "memory_ms": 80, "postprocess_method": "ssr"})
    metadata = json.loads(json.dumps(core_preset_metadata(params)))
    restored = resolve_core_config(metadata["core_parameters"])
    assert restored == params
    assert metadata["preset_name"] == "core_enhanced_v1"
    assert metadata["feature_flags"]["simple_reset_fft_handoff"]
    assert not metadata["feature_flags"]["gap_rescue"]
    assert params.Spec_Penalty_Weight == 0.2
    for override in ({"TW": 8}, {"tracking_range_up_bpm": 40}, {"Spec_Penalty_Weight": 0.4}, {"handoff_deadline_s": 30},
                     {"enable_high_lock_recovery": True}):
        with pytest.raises(ValueError):
            resolve_core_config(override)


def test_preset_resolution_does_not_use_future_dataclass_defaults(monkeypatch) -> None:
    monkeypatch.setattr(ProtocolTrialParams, "Spec_Penalty_Weight", 0.9)
    assert resolve_core_config().Spec_Penalty_Weight == 0.2
