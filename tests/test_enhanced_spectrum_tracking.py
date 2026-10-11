from __future__ import annotations

import numpy as np
import pytest

from ppg_hr.experimental.cascade_solver import _track_candidate_spectrum
from ppg_hr.experimental.enhanced_tracking import SpectrumTrackingState, track_spectrum_candidates
from ppg_hr.experimental.protocol_search_space import ProtocolTrialParams


def _spectrum(peaks: dict[float, float]) -> tuple[np.ndarray, np.ndarray]:
    freq = np.arange(0.5, 4.001, 1.0 / 60.0)
    amp = np.full(freq.size, 0.001)
    for bpm, value in peaks.items():
        idx = int(np.argmin(np.abs(freq * 60.0 - bpm)))
        amp[max(0, idx - 1) : idx + 2] = [0.01, value, 0.01]
    return freq, amp


def test_soft_penalty_keeps_a_strong_penalized_candidate_selectable() -> None:
    freq, amp = _spectrum({110: 10.0, 120: 1.6})
    hr, trace = track_spectrum_candidates(
        freq, amp, previous_hr_bpm=105.0,
        params=ProtocolTrialParams(enable_continuity_protection=False),
        state=SpectrumTrackingState(), window_kind="motion", path="adaptive",
        enable_penalty=True, penalty_peaks_hz=np.array([110 / 60]),
        penalty_peak_amplitudes=np.array([1.0]),
    )
    assert trace.penalty_applied
    assert trace.tracked_hr_bpm == pytest.approx(110.0)
    assert hr == pytest.approx(110.0)


def test_directional_tracking_uses_asymmetric_range_and_slew() -> None:
    freq, amp = _spectrum({84: 0.8, 120: 1.0})
    params = ProtocolTrialParams(
        motion_tracking_range_up_bpm=12.0,
        motion_tracking_range_down_bpm=30.0,
        motion_tracking_slew_limit_up_bpm=6.0,
        motion_tracking_slew_step_up_bpm=3.0,
        motion_tracking_slew_limit_down_bpm=30.0,
        motion_tracking_slew_step_down_bpm=20.0,
    )

    hr, trace = track_spectrum_candidates(
        freq,
        amp,
        previous_hr_bpm=90.0,
        params=params,
        state=SpectrumTrackingState(),
        window_kind="motion",
        path="adaptive",
    )

    assert hr == pytest.approx(84.0)
    assert trace.search_min_bpm == pytest.approx(60.0)
    assert trace.search_max_bpm == pytest.approx(102.0)
    freq2, amp2 = _spectrum({120: 1.0})
    slew_params = ProtocolTrialParams(
        motion_tracking_range_up_bpm=40.0,
        motion_tracking_range_down_bpm=40.0,
        motion_tracking_slew_limit_up_bpm=6.0,
        motion_tracking_slew_step_up_bpm=3.0,
    )
    hr2, _ = track_spectrum_candidates(
        freq2,
        amp2,
        previous_hr_bpm=90.0,
        params=slew_params,
        state=SpectrumTrackingState(),
        window_kind="motion",
        path="adaptive",
    )
    assert hr2 == pytest.approx(93.0)


def test_reference_directional_defaults_are_selected_by_path_and_stage() -> None:
    freq, amp = _spectrum({88: 0.8, 132: 1.0})
    params = ProtocolTrialParams()

    baseline_hr, baseline_trace = track_spectrum_candidates(
        freq,
        amp,
        previous_hr_bpm=100.0,
        params=params,
        state=SpectrumTrackingState(),
        window_kind="motion",
        path="baseline",
    )
    motion_hr, motion_trace = track_spectrum_candidates(
        freq,
        amp,
        previous_hr_bpm=100.0,
        params=params,
        state=SpectrumTrackingState(),
        window_kind="motion",
        path="adaptive",
    )
    recovery_hr, recovery_trace = track_spectrum_candidates(
        freq,
        amp,
        previous_hr_bpm=100.0,
        params=params,
        state=SpectrumTrackingState(),
        window_kind="recovery",
        path="adaptive",
    )

    assert (baseline_trace.search_min_bpm, baseline_trace.search_max_bpm) == pytest.approx(
        (70.0, 130.0)
    )
    assert baseline_hr == pytest.approx(96.0)
    assert (motion_trace.search_min_bpm, motion_trace.search_max_bpm) == pytest.approx(
        (85.0, 135.0)
    )
    assert motion_hr == pytest.approx(103.5)
    assert (recovery_trace.search_min_bpm, recovery_trace.search_max_bpm) == pytest.approx(
        (75.0, 120.0)
    )
    assert recovery_hr == pytest.approx(97.0)


def test_post_motion_reset_uses_reference_specific_recovery_slew() -> None:
    freq = np.arange(0.5, 4.001, 0.1 / 60.0)
    amp = np.full(freq.size, 0.001)
    peak_idx = int(np.argmin(np.abs(freq * 60.0 - 96.7)))
    amp[peak_idx - 1 : peak_idx + 2] = [0.01, 1.0, 0.01]
    params = ProtocolTrialParams()

    adaptive_hr, _ = track_spectrum_candidates(
        freq,
        amp,
        previous_hr_bpm=100.0,
        params=params,
        state=SpectrumTrackingState(),
        window_kind="recovery",
        path="adaptive",
    )
    reset_hr, _ = track_spectrum_candidates(
        freq,
        amp,
        previous_hr_bpm=100.0,
        params=params,
        state=SpectrumTrackingState(),
        window_kind="recovery",
        path="fft_post_motion_reset",
    )

    assert adaptive_hr == pytest.approx(96.7)
    assert reset_hr == pytest.approx(97.0)


def test_dynamic_penalty_uses_confidence_and_only_existing_harmonic() -> None:
    freq, amp = _spectrum({60: 1.0, 120: 0.8, 150: 0.7})
    _, trace = track_spectrum_candidates(
        freq,
        amp,
        previous_hr_bpm=None,
        params=ProtocolTrialParams(Spec_Penalty_Weight=0.2, Spec_Penalty_Width=0.1),
        state=SpectrumTrackingState(),
        window_kind="motion",
        path="adaptive",
        enable_penalty=True,
        penalty_peaks_hz=np.asarray([1.0, 1.4]),
        penalty_peak_amplitudes=np.asarray([1.0, 0.8]),
    )

    assert trace.penalty_confidence == pytest.approx(0.2)
    assert trace.penalty_centers_bpm == pytest.approx((60.0, 120.0))
    assert trace.harmonic_penalty_applied is True
    assert trace.penalty_weight_min > 0.2

    freq_no_harmonic, amp_no_harmonic = _spectrum({60: 1.0, 150: 0.7})
    _, no_harmonic = track_spectrum_candidates(
        freq_no_harmonic,
        amp_no_harmonic,
        previous_hr_bpm=None,
        params=ProtocolTrialParams(),
        state=SpectrumTrackingState(),
        window_kind="motion",
        path="adaptive",
        enable_penalty=True,
        penalty_peaks_hz=np.asarray([1.0]),
        penalty_peak_amplitudes=np.asarray([1.0]),
    )
    assert no_harmonic.penalty_centers_bpm == pytest.approx((60.0,))


def test_reference_candidate_thresholds_keep_full_ppg_but_filter_motion_peaks() -> None:
    fs = 100
    seconds = 8
    t = np.arange(fs * seconds, dtype=float) / fs
    penalty_ref = np.sin(2.0 * np.pi * 1.0 * t) + 0.2 * np.sin(2.0 * np.pi * 1.5 * t)
    freq, amp = _spectrum({60: 1.0, 90: 0.2})
    params = ProtocolTrialParams(
        candidate_peak_threshold_ratio=0.30,
        full_candidate_peak_threshold_ratio=0.15,
    )

    _, trace = _track_candidate_spectrum(
        np.ones(t.size),
        fs,
        None,
        params,
        enable_penalty=True,
        penalty_ref=penalty_ref,
        precomputed_spectrum=(freq, amp),
        tracking_state=SpectrumTrackingState(),
        window_kind="motion",
        path="adaptive",
        return_trace=True,
    )

    assert any(value == pytest.approx(90.0) for value in trace["candidate_peaks_bpm"])
    assert trace["penalty_confidence"] == pytest.approx(1.0)
    assert trace["candidate_peak_threshold_ratio"] == pytest.approx(0.30)
    assert trace["full_candidate_peak_threshold_ratio"] == pytest.approx(0.15)


@pytest.mark.parametrize(
    ("candidate_ratio", "full_ratio"),
    [(0.0, 0.15), (1.1, 0.15), (0.3, 0.0), (0.2, 0.3)],
)
def test_candidate_thresholds_reject_invalid_hierarchy(candidate_ratio: float, full_ratio: float) -> None:
    freq, amp = _spectrum({60: 1.0})
    with pytest.raises(ValueError, match="candidate peak threshold"):
        track_spectrum_candidates(
            freq,
            amp,
            previous_hr_bpm=None,
            params=ProtocolTrialParams(
                candidate_peak_threshold_ratio=candidate_ratio,
                full_candidate_peak_threshold_ratio=full_ratio,
            ),
            state=SpectrumTrackingState(),
            window_kind="motion",
            path="adaptive",
        )


def test_continuity_protection_can_yield_to_strong_non_penalty_challenger() -> None:
    freq, amp = _spectrum({117: 0.55, 138: 1.0})
    hr, trace = track_spectrum_candidates(
        freq,
        amp,
        previous_hr_bpm=137.0,
        params=ProtocolTrialParams(
            motion_tracking_range_up_bpm=25.0,
            motion_tracking_range_down_bpm=25.0,
            motion_tracking_slew_limit_down_bpm=30.0,
        ),
        state=SpectrumTrackingState(),
        window_kind="motion",
        path="adaptive",
        enable_penalty=True,
        penalty_peaks_hz=np.asarray([138.0 / 60.0]),
        penalty_peak_amplitudes=np.asarray([1.0]),
    )

    assert hr == pytest.approx(117.0)
    assert trace.protection_suppressed is True
    assert trace.protection_suppression_reason == "motion_core_challenger"
    assert trace.protection_challenger_bpm == pytest.approx(117.0)


def test_low_lock_recovery_confirms_then_moves_toward_stable_challenger() -> None:
    freq, amp = _spectrum({65: 0.8, 110: 1.0})
    state = SpectrumTrackingState(low_lock_count=4)
    params = ProtocolTrialParams(enable_low_lock_recovery=True,low_lock_confirm_windows=3, low_lock_step_bpm=30.0)
    outputs = []
    traces = []
    previous = 65.0
    for _ in range(3):
        previous, trace = track_spectrum_candidates(
            freq,
            amp,
            previous_hr_bpm=previous,
            params=params,
            state=state,
            window_kind="motion",
            path="adaptive",
            adaptive_filter="lms",
        )
        outputs.append(previous)
        traces.append(trace)

    assert traces[-1].low_lock_triggered is True
    assert outputs[-1] > 65.0
    assert traces[-1].low_lock_mode == "reacquiring"

    _, non_lms_trace = track_spectrum_candidates(
        freq,
        amp,
        previous_hr_bpm=65.0,
        params=params,
        state=SpectrumTrackingState(low_lock_count=10),
        window_kind="motion",
        path="adaptive",
        adaptive_filter="klms",
    )
    assert non_lms_trace.low_lock_requested is True
    assert non_lms_trace.low_lock_effective is False


def test_high_lock_recovery_records_risk_confirmation_and_cooldown() -> None:
    freq, amp = _spectrum({100: 0.8, 145: 1.0})
    state = SpectrumTrackingState()
    params = ProtocolTrialParams(enable_high_lock_recovery=True,high_lock_confirm_windows=2, high_lock_cooldown_windows=3)
    previous = 145.0
    traces = []
    for _ in range(2):
        previous, trace = track_spectrum_candidates(
            freq,
            amp,
            previous_hr_bpm=previous,
            params=params,
            state=state,
            window_kind="motion",
            path="adaptive",
            enable_penalty=True,
            penalty_peaks_hz=np.asarray([145.0 / 60.0]),
            penalty_peak_amplitudes=np.asarray([1.0]),
        )
        traces.append(trace)

    assert traces[-1].high_lock_triggered is True
    assert "near_motion_peak" in traces[-1].high_lock_labels
    assert previous < 145.0
    for _ in range(4):
        previous, trace = track_spectrum_candidates(
            freq,
            amp,
            previous_hr_bpm=previous,
            params=params,
            state=state,
            window_kind="motion",
            path="adaptive",
        )
    assert trace.high_lock_cooldown >= 0


def test_core_rejects_legacy_tracker_and_rescue_overrides() -> None:
    from ppg_hr.params import resolve_core_config
    for override in ({"tracker_mode": "legacy"}, {"enable_low_lock_recovery": True},
                     {"enable_high_lock_recovery": True}, {"post_motion_guard_gap_rescue_enable": True}):
        with pytest.raises(ValueError, match="Frozen"):
            resolve_core_config(override)
