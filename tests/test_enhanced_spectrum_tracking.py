from __future__ import annotations

import numpy as np
import pytest

from ppg_hr.experimental.cascade_solver import _extract_fft_hr
from ppg_hr.experimental.enhanced_tracking import SpectrumTrackingState, track_spectrum_candidates
from ppg_hr.experimental.protocol_search_space import ProtocolTrialParams


def _spectrum(peaks: dict[float, float]) -> tuple[np.ndarray, np.ndarray]:
    freq = np.arange(0.5, 4.001, 1.0 / 60.0)
    amp = np.full(freq.size, 0.001)
    for bpm, value in peaks.items():
        idx = int(np.argmin(np.abs(freq * 60.0 - bpm)))
        amp[max(0, idx - 1) : idx + 2] = [0.01, value, 0.01]
    return freq, amp


def test_directional_tracking_uses_asymmetric_range_and_slew() -> None:
    freq, amp = _spectrum({84: 0.8, 120: 1.0})
    params = ProtocolTrialParams(
        tracking_range_up_bpm=12.0,
        tracking_range_down_bpm=30.0,
        tracking_slew_limit_up_bpm=6.0,
        tracking_slew_step_up_bpm=3.0,
        tracking_slew_limit_down_bpm=30.0,
        tracking_slew_step_down_bpm=20.0,
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
        tracking_range_up_bpm=40.0,
        tracking_range_down_bpm=40.0,
        tracking_slew_limit_up_bpm=6.0,
        tracking_slew_step_up_bpm=3.0,
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


def test_continuity_protection_can_yield_to_strong_non_penalty_challenger() -> None:
    freq, amp = _spectrum({117: 0.55, 138: 1.0})
    hr, trace = track_spectrum_candidates(
        freq,
        amp,
        previous_hr_bpm=137.0,
        params=ProtocolTrialParams(
            tracking_range_up_bpm=25.0,
            tracking_range_down_bpm=25.0,
            tracking_slew_limit_down_bpm=30.0,
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
    params = ProtocolTrialParams(low_lock_confirm_windows=3, low_lock_step_bpm=30.0)
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
    params = ProtocolTrialParams(high_lock_confirm_windows=2, high_lock_cooldown_windows=3)
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


def test_legacy_equals_enhanced_with_all_six_features_disabled() -> None:
    freq, amp = _spectrum({72: 0.8, 126: 1.0})
    legacy = ProtocolTrialParams(tracker_mode="legacy")
    disabled = ProtocolTrialParams(
        tracker_mode="enhanced",
        enable_directional_tracking=False,
        enable_dynamic_penalty=False,
        enable_continuity_protection=False,
        enable_low_lock_recovery=False,
        enable_high_lock_recovery=False,
        enable_post_motion_protection=False,
    )

    legacy_hr = _extract_fft_hr(
        np.ones(10), 100, 120.0, legacy, enable_penalty=False, penalty_ref=None, precomputed_spectrum=(freq, amp)
    )
    disabled_hr = _extract_fft_hr(
        np.ones(10), 100, 120.0, disabled, enable_penalty=False, penalty_ref=None, precomputed_spectrum=(freq, amp)
    )
    assert disabled_hr == pytest.approx(legacy_hr)
