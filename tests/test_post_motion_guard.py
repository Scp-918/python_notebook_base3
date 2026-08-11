from __future__ import annotations

import json

import numpy as np

from ppg_hr.experimental.fusion import fuse_final_hr
from ppg_hr.experimental.post_motion_guard import post_motion_switch_policy
from ppg_hr.experimental.protocol_search_space import ProtocolTrialParams
from ppg_hr.experimental.run_batch_protocol import flatten_params_for_record, protocol_params_from_record


def test_post_motion_guard_switches_on_stable_crossover() -> None:
    time = np.asarray([98, 99, 100, 101, 102, 103, 104, 105], dtype=float)
    adaptive = np.asarray([120, 115, 110, 104, 101, 98, 95, 92], dtype=float)
    reset_fft = np.asarray([118, 112, 108, 103, 100, 97, 94, 91], dtype=float)

    mask, reasons, events = post_motion_switch_policy(
        time, adaptive, reset_fft, motion_start_s=80.0, motion_end_s=100.0, params=ProtocolTrialParams(
            post_motion_guard_min_elapsed_s=1.0,
            post_motion_guard_stable_windows=3,
            post_motion_guard_crossover_gap_bpm=3.0,
        )
    )

    assert mask.tolist() == [True, True, True, True, True, True, False, False]
    assert reasons[6] == "stable_crossover"
    assert events[0]["switch_reason"] == "stable_crossover"


def test_post_motion_guard_gap_rescue_requires_stable_reset_fft() -> None:
    time = np.asarray([98, 99, 100, 101, 102, 103, 104, 105, 106], dtype=float)
    adaptive = np.asarray([130, 128, 126, 124, 121, 118, 115, 112, 109], dtype=float)
    reset_fft = np.asarray([118, 112, 108, 108, 88, 86, 84, 82, 80], dtype=float)
    params = ProtocolTrialParams(
        post_motion_guard_min_elapsed_s=1.0,
        post_motion_guard_crossover_gap_bpm=2.0,
        post_motion_guard_rescue_gap_bpm=20.0,
        post_motion_guard_gap_rescue_windows=4,
        post_motion_guard_gap_rescue_min_hits=4,
        post_motion_guard_fft_stable_windows=3,
        post_motion_guard_fft_stable_bpm=5.0,
    )

    mask, reasons, events = post_motion_switch_policy(
        time, adaptive, reset_fft, motion_start_s=80.0, motion_end_s=100.0, params=params
    )

    assert mask.tolist() == [True, True, True, True, True, True, True, False, False]
    assert reasons[7] == "gap_rescue"
    assert events[0]["hard_switch"] is True
    assert events[0]["gap_rescue_count"] == 4


def test_fusion_uses_reset_fft_after_guard_without_reference_hr() -> None:
    result = fuse_final_hr(
        time_s=np.asarray([99.0, 101.0, 102.0, 103.0, 104.0]),
        baseline_hr_bpm=np.asarray([100.0, 98.0, 96.0, 94.0, 92.0]),
        reset_fft_hr_bpm=np.asarray([100.0, 95.0, 94.0, 93.0, 92.0]),
        adaptive_hr_bpm=np.asarray([110.0, 103.0, 98.0, 94.0, 93.0]),
        segment_label=np.asarray(["motion", "recovery", "recovery", "recovery", "recovery"]),
        qc_status=np.asarray(["ok"] * 5),
        adaptive_filter="lms",
        motion_start_s=90.0,
        motion_end_s=100.0,
        params=ProtocolTrialParams(
            post_motion_guard_min_elapsed_s=0.0,
            post_motion_guard_stable_windows=2,
            post_motion_guard_crossover_gap_bpm=5.0,
        ),
        target_scope="motion_post10",
    )

    assert "stable_crossover" in result.switch_reason
    switch_idx = int(np.flatnonzero(result.switch_reason == "stable_crossover")[0])
    assert result.final_source[switch_idx] == "reset_fft"
    assert result.final_hr_bpm[switch_idx] == result.reset_fft_hr_bpm[switch_idx]


def test_six_switches_roundtrip_through_cache_csv_json_and_replay_parser() -> None:
    params = ProtocolTrialParams(
        tracker_mode="legacy",
        enable_directional_tracking=False,
        enable_dynamic_penalty=True,
        enable_continuity_protection=False,
        enable_low_lock_recovery=True,
        enable_high_lock_recovery=False,
        enable_post_motion_protection=True,
    )
    flat = flatten_params_for_record(params)
    restored = protocol_params_from_record(flat)
    json_restored = ProtocolTrialParams(**json.loads(json.dumps(params.to_dict())))

    names = [
        "tracker_mode",
        "enable_directional_tracking",
        "enable_dynamic_penalty",
        "enable_continuity_protection",
        "enable_low_lock_recovery",
        "enable_high_lock_recovery",
        "enable_post_motion_protection",
    ]
    assert {name: getattr(restored, name) for name in names} == {
        name: getattr(params, name) for name in names
    }
    assert {name: getattr(json_restored, name) for name in names} == {
        name: getattr(params, name) for name in names
    }
    assert params.cache_key() != ProtocolTrialParams().cache_key()
