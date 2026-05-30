from __future__ import annotations

import numpy as np

from ppg_hr.experimental.cascade_solver import _extract_hr, _extract_hr_result
from ppg_hr.experimental.protocol_search_space import ProtocolTrialParams
from ppg_hr.experimental.spectral_utils import sparse_spectrum_hr_candidates


def _tone(fs: int, seconds: float, bpm: float, amplitude: float = 1.0) -> np.ndarray:
    t = np.arange(int(round(fs * seconds)), dtype=float) / float(fs)
    return amplitude * np.sin(2.0 * np.pi * (float(bpm) / 60.0) * t)


def test_fft_postprocess_keeps_existing_extract_hr_behavior() -> None:
    fs = 50
    signal = _tone(fs, 8.0, 72.0)
    params = ProtocolTrialParams(postprocess_method="fft")

    old_value = _extract_hr(signal, fs, previous_hr=None, params=params, enable_penalty=False, penalty_ref=None)
    result = _extract_hr_result(signal, fs, previous_hr=None, params=params, enable_penalty=False, penalty_ref=None)

    assert result["postprocess_method"] == "fft"
    assert result["postprocess_fallback"] == ""
    assert np.isclose(float(result["hr_bpm"]), old_value)


def test_ssr_harmonic_tracking_prefers_previous_hr_half_frequency_candidate() -> None:
    fs = 50
    signal = _tone(fs, 8.0, 90.0, amplitude=0.8) + _tone(fs, 8.0, 180.0, amplitude=1.4)
    result = sparse_spectrum_hr_candidates(
        signal,
        fs,
        previous_hr=92.0,
        hr_band_bpm=(40.0, 200.0),
        num_atoms=6,
        lambda_threshold=0.05,
        harmonic_tol_bpm=4.0,
        grid_resolution_bpm=1.0,
        slew_limit_bpm=20.0,
        slew_step_bpm=7.0,
    )

    assert result["status"] == "ok"
    assert abs(float(result["hr_bpm"]) - 90.0) <= 2.0
    assert result["harmonic_adjusted"] is True


def test_ssr_extract_hr_falls_back_to_fft_when_candidates_are_empty() -> None:
    fs = 50
    signal = _tone(fs, 8.0, 75.0)
    params = ProtocolTrialParams(
        postprocess_method="ssr",
        SSR_Lambda=2.0,
        SSR_Fallback_To_FFT=True,
        SSR_Grid_Resolution_BPM=1.0,
    )

    result = _extract_hr_result(signal, fs, previous_hr=None, params=params, enable_penalty=False, penalty_ref=None)

    assert result["postprocess_method"] == "ssr"
    assert result["postprocess_fallback"] == "fft"
    assert np.isfinite(float(result["hr_bpm"]))
    assert abs(float(result["hr_bpm"]) - 75.0) < 5.0
