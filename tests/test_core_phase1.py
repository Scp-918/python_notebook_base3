from dataclasses import replace
import numpy as np
import pytest
from ppg_hr.experimental.segmentation import (
    apply_motion_protocol,
    detect_source_imu_segments,
    clean_motion_flags,
    select_motion_runs,
)
from ppg_hr.experimental.preprocess_protocol import (
    SourceImu,
    ProtocolDataset,
    resample_protocol_dataset,
)
from ppg_hr.experimental.cascade_solver import run_protocol_trial, _window_in_scope
from ppg_hr.experimental.protocol_search_space import ProtocolTrialParams
from ppg_hr.params import TargetScope


def source(duration=140, bouts=((40, 60), (90, 110)), gyro=True):
    t = np.arange(duration * 100) / 100
    active = np.zeros(t.size, bool)
    for start, end in bouts:
        active |= (t >= start) & (t < end)
    signal = active * np.sin(2 * np.pi * 2 * t)
    z = np.zeros(t.size)
    return SourceImu(100, t, z, z, z, signal if gyro else z, z, z)


def dataset(imu):
    t = imu.time_s
    ppg = np.sin(2 * np.pi * 1.2 * t)
    z = np.zeros(t.size)
    return ProtocolDataset(
        sample_stem="synthetic",
        fs=100,
        time_s=t,
        ppg_green=ppg,
        ppg_red=ppg,
        ppg_ir=ppg,
        hf1=z,
        hf2=z,
        cf1=z,
        cf2=z,
        accx=imu.accx,
        accy=imu.accy,
        accz=imu.accz,
        gyrox=imu.gyrox,
        gyroy=imu.gyroy,
        gyroz=imu.gyroz,
        ref_time_s=t[::100],
        ref_hr_bpm=np.full(t[::100].size, 72.0),
        source_imu=imu,

        ppg_clean_green=ppg,
        ppg_clean_red=ppg,
        ppg_clean_ir=ppg,
    )


def test_gyro_detection_preserves_all_runs_and_selects_policy():
    result = detect_source_imu_segments(source())
    assert result.is_valid
    assert len(result.detected_runs) == len(result.motion_runs) == 2
    assert result.gyro_threshold > 0
    assert not np.any(result.acc_scores > result.acc_threshold)
    one = apply_motion_protocol(detect_source_imu_segments(source()))
    assert len(one.detected_runs) == 2 and len(one.motion_runs) == 1
    assert one.motion_runs[0] == one.detected_runs[0]
    assert np.all(result.labels[result.motion_flags] == "motion")


def test_morphology_bridges_before_dropping_and_only_interior_gaps():
    flags = np.array(
        [False] * 2
        + [True] * 3
        + [False] * 3
        + [True] * 3
        + [False] * 4
        + [True] * 4
        + [False] * 2
    )
    cleaned = clean_motion_flags(flags)
    assert cleaned.tolist() == [False] * 2 + [True] * 9 + [False] * 10
    assert select_motion_runs(((4.0, 9.0), (15.0, 20.0)), "single_bout") == (
        (4.0, 9.0),
    )


def test_center_half_open_and_episode_elapsed():
    result = detect_source_imu_segments(source())
    for start, end in result.motion_runs:
        labels, episode, elapsed = result.states_at(
            np.array([start - 1, start, end - 1, end, end + 1])
        )
        assert labels[1:4].tolist() == ["motion", "motion", "recovery"]
        assert elapsed[3:5].tolist() == [0.0, 1.0]
    assert result.window_centers_s[0] == 4
    assert result.window_ends_s[0] == 8


@pytest.mark.parametrize("bouts", [(), ((0, 65),), ((40, 140),)])
def test_no_motion_and_truncated_motion_are_valid(bouts):
    result = detect_source_imu_segments(source(bouts=bouts))
    assert result.is_valid
    if not bouts:
        assert result.motion_runs == () and set(result.labels) == {"rest"}


def test_source_survives_resampling_and_missing_source_is_rejected():
    ds = dataset(source())
    assert resample_protocol_dataset(ds, 25).source_imu is ds.source_imu
    with pytest.raises(ValueError, match="source_imu"):
        detect_source_imu_segments(replace(ds, source_imu=None).source_imu)


def test_prediction_reference_invariance_and_full_sensor_grid():
    ds = dataset(source(duration=80, bouts=((40, 60),)))
    p = ProtocolTrialParams(Fs_Target=25, hr_window_s=8, memory_ms=4, prewarm_s=0)
    a = run_protocol_trial(ds, ('ACC',), "global", p)
    changed = replace(
        ds, ref_time_s=np.array([20.0, 21.0]), ref_hr_bpm=np.array([np.nan, 160.0])
    )
    b = run_protocol_trial(
        changed, ('ACC',), "global", replace(p, Time_Bias_After_Range_S=(-1.0, 1.0))
    )
    assert a.frame.shape[0] == b.frame.shape[0] == 73, (a.reason, b.reason)
    for field in [
        "time_s",
        "segment_label",
        "baseline_hr_bpm",
        "adaptive_hr_bpm",
        "final_hr_bpm",
        "episode_id",
        "handoff_state",
        "selected_hr_bpm",
    ]:
        np.testing.assert_equal(a.metric_arrays[field], b.metric_arrays[field])
    assert a.frame.time_s.iloc[0] == 4 and a.frame.time_s.iloc[-1] == 76
    assert a.alignment_info.best_tdelay_s == a.alignment_info.ref_shift_s == 0


def test_motion_post10_uses_each_episode_end():
    seg = detect_source_imu_segments(source())
    for _, end in seg.motion_runs:
        assert _window_in_scope("recovery", TargetScope.MOTION_POST10, end, seg)
        assert _window_in_scope("recovery", TargetScope.MOTION_POST10, end + 10, seg)
        assert not _window_in_scope(
            "recovery", TargetScope.MOTION_POST10, end + 11, seg
        )


def test_online_sensor_view_does_not_expose_reference():
    view = dataset(source()).sensor_view()
    assert not hasattr(view, "ref_time_s") and not hasattr(view, "ref_hr_bpm")
    assert view.source_imu.fs == 100


@pytest.mark.parametrize("scope", ["motion_only", "motion_recovery", "motion_post10"])
def test_no_motion_scope_still_returns_full_prediction(scope):
    result = run_protocol_trial(
        dataset(source(duration=45, bouts=())),
        ('ACC',),
        scope,
        ProtocolTrialParams(Fs_Target=25, memory_ms=4),
    )
    assert result.success, result.reason
    assert len(result.frame) == 38
    assert result.metadata["scoring_status"] == "no_valid_reference_windows"
    assert np.isnan(result.final_aae_bpm)


def test_physical_memory_and_fixed_observation():
    from ppg_hr.experimental.noncausal_lms import map_delay_to_lms_params

    ds = dataset(source(duration=80, bouts=((40, 60),)))
    predictions = []
    for fs, memory in ((25, 160), (50, 160), (100, 160), (100, 80)):
        params = ProtocolTrialParams(Fs_Target=fs, memory_ms=memory)
        design = map_delay_to_lms_params(1000, "ACC", params, fs)
        assert design.M == params.history_tap_limit
        assert abs(design.M * 1000 / fs - memory) <= 500 / fs
        result = run_protocol_trial(ds, ('ACC',), "global", params, collect_stages=False)
        assert result.success, result.reason
        assert np.all(result.frame.window_end_s - result.frame.window_start_s == 8)
        assert np.all(np.diff(result.frame.window_center_s) == 1)
        predictions.append(result.frame.window_center_s.to_numpy())
    for times in predictions[1:]:
        np.testing.assert_equal(times, predictions[0])
    with pytest.raises(ValueError, match="frozen"):
        ProtocolTrialParams(hr_window_s=6)
