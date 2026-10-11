from __future__ import annotations

import numpy as np
import pytest

from ppg_hr.experimental.cascade_solver import _extract_hr, _extract_hr_result
from ppg_hr.experimental.protocol_search_space import ProtocolTrialParams
from ppg_hr.experimental.spectral_utils import sparse_spectrum_hr_candidates


def test_fft_ssr_use_common_fft_harmonic_evidence(monkeypatch) -> None:
    from ppg_hr.experimental import cascade_solver as solver

    fs = 50
    time = np.arange(fs * 8) / fs
    signal = np.sin(2 * np.pi * 1.5 * time) + 0.6 * np.sin(2 * np.pi * 3.0 * time)
    reference = np.sin(2 * np.pi * 1.5 * time)
    frequencies, amplitude = solver._spectrum(signal, fs)
    sparse = np.zeros(amplitude.shape)
    sparse[np.argmin(np.abs(frequencies - 1.5))] = 1.0
    monkeypatch.setattr(solver, "sparse_spectrum_hr_candidates", lambda *args, **kwargs:
                        {"status": "ok", "frequencies_hz": frequencies, "spectrum_score": sparse})
    traces = []
    for method in ("fft", "ssr"):
        result = solver._extract_hr_result(signal, fs, None,
            ProtocolTrialParams(postprocess_method=method, enable_continuity_protection=False),
            enable_penalty=True, penalty_ref=reference, window_kind="motion")
        traces.append(result["spectrum_tracking"])
    assert len(traces[0]["penalty_centers_bpm"]) == 2
    assert traces[0]["penalty_centers_bpm"] == traces[1]["penalty_centers_bpm"]
    assert traces[0]["penalty_weight_min"] == pytest.approx(traces[1]["penalty_weight_min"])


def test_soft_weight_half_width_and_overlap_leave_outside_unchanged() -> None:
    from ppg_hr.experimental.penalty_policy import penalty_weights

    frequencies = np.array([0.5, 1.0, 1.1, 1.2, 1.5])
    weights = penalty_weights(frequencies, (1.0, 1.2), 0.2, 0.2)
    assert weights == pytest.approx([1.0, 0.2, 0.6, 0.2, 1.0])


def _tone(fs: int, seconds: float, bpm: float, amplitude: float = 1.0) -> np.ndarray:
    t = np.arange(int(round(fs * seconds)), dtype=float) / float(fs)
    return amplitude * np.sin(2.0 * np.pi * (float(bpm) / 60.0) * t)


def test_fft_postprocess_keeps_existing_extract_hr_behavior() -> None:
    fs = 50
    signal = _tone(fs, 8.0, 72.0)
    params = ProtocolTrialParams(postprocess_method="fft")

    old_value = _extract_hr(
        signal,
        fs,
        previous_hr=None,
        params=params,
        enable_penalty=False,
        penalty_ref=None,
    )
    result = _extract_hr_result(
        signal,
        fs,
        previous_hr=None,
        params=params,
        enable_penalty=False,
        penalty_ref=None,
    )

    assert result["postprocess_method"] == "fft"
    assert result["postprocess_fallback"] == ""
    assert np.isclose(float(result["hr_bpm"]), old_value)


def test_ssr_candidates_use_the_common_tracker_once(monkeypatch):
    from ppg_hr.experimental import cascade_solver as solver

    fs = 50
    signal = _tone(fs, 8, 90, 0.8) + _tone(fs, 8, 180, 1.4)
    sparse = sparse_spectrum_hr_candidates(
        signal, fs, hr_band_bpm=(40, 200), num_atoms=6, lambda_threshold=0.05
    )
    assert "hr_bpm" not in sparse
    assert any(abs(x - 90) < 2 for x in sparse["candidate_bpm"])
    original = solver.track_spectrum_candidates
    calls = []

    def track(*args, **kwargs):
        calls.append(kwargs["path"])
        return original(*args, **kwargs)

    monkeypatch.setattr(solver, "track_spectrum_candidates", track)
    result = solver._extract_hr_result(
        signal,
        fs,
        92,
        ProtocolTrialParams(postprocess_method="ssr"),
        enable_penalty=False,
        penalty_ref=None,
    )
    assert result["actual_estimator"] == "ssr"
    assert abs(result["hr_bpm"] - 90) <= 2
    assert calls == ["adaptive"]


def test_ssr_extract_hr_falls_back_to_fft_when_candidates_are_empty() -> None:
    fs = 50
    signal = _tone(fs, 8.0, 75.0)
    params = ProtocolTrialParams(
        postprocess_method="ssr",
        SSR_Lambda=2.0,
        SSR_Fallback_To_FFT=True,
        SSR_Grid_Resolution_BPM=1.0,
    )

    result = _extract_hr_result(
        signal,
        fs,
        previous_hr=None,
        params=params,
        enable_penalty=False,
        penalty_ref=None,
    )

    assert result["postprocess_method"] == "ssr"
    assert result["postprocess_fallback"] == "fft"
    assert np.isfinite(float(result["hr_bpm"]))
    assert abs(float(result["hr_bpm"]) - 75.0) < 5.0
