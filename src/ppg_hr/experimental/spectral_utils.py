"""Shared FFT helpers for experimental protocol modules.

The helpers keep the existing magnitude-spectrum semantics used by the
alignment and cascade code while centralising Hamming/frequency-axis caches.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.signal import find_peaks
from scipy.signal.windows import hamming

__all__ = [
    "cached_hamming",
    "cached_rfftfreq",
    "compute_power_spectrum",
    "dominant_frequency_in_band",
    "fft_peak_candidates",
    "prepare_fft_window",
    "sparse_spectrum_hr_candidates",
]


@lru_cache(maxsize=64)
def cached_hamming(signal_len: int, *, sym: bool = False) -> np.ndarray:
    """Return a read-only Hamming window keyed by length and symmetry."""

    win = hamming(int(signal_len), sym=bool(sym))
    win.setflags(write=False)
    return win


@lru_cache(maxsize=128)
def cached_rfftfreq(fs: float, nfft: int) -> np.ndarray:
    """Return a read-only real-FFT frequency axis."""

    freq = np.fft.rfftfreq(int(nfft), d=1.0 / float(fs))
    freq.setflags(write=False)
    return freq


def prepare_fft_window(
    signal: np.ndarray,
    *,
    apply_hamming: bool = True,
    demean: bool = True,
    hamming_sym: bool = False,
) -> np.ndarray:
    """Return a finite FFT input window without mutating the caller's array."""

    sig = np.asarray(signal, dtype=float).ravel()
    if sig.size == 0:
        return sig.copy()
    sig = sig.copy()
    sig[~np.isfinite(sig)] = 0.0
    if demean:
        sig = sig - float(np.mean(sig))
    if apply_hamming:
        sig = sig * cached_hamming(sig.size, sym=bool(hamming_sym))
    return sig


def compute_power_spectrum(
    signal: np.ndarray,
    fs: float,
    *,
    nfft: int | None = None,
    apply_hamming: bool = True,
    demean: bool = True,
    hamming_sym: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return FFT frequency bins and magnitude spectrum.

    The name says "power" for API readability, but the returned second value
    intentionally remains ``abs(rfft)`` to preserve the existing HR logic.
    """

    sig = prepare_fft_window(
        signal,
        apply_hamming=apply_hamming,
        demean=demean,
        hamming_sym=hamming_sym,
    )
    if sig.size == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    nfft_i = int(nfft) if nfft is not None else max(8192, 1 << int(np.ceil(np.log2(max(sig.size, 1)))))
    freq = cached_rfftfreq(float(fs), nfft_i)
    amp = np.abs(np.fft.rfft(sig, n=nfft_i))
    return freq, amp


def dominant_frequency_in_band(
    signal: np.ndarray,
    fs: float,
    fmin: float,
    fmax: float,
    *,
    nfft: int | None = None,
    fallback: float = float("nan"),
) -> float:
    """Return the dominant magnitude peak in ``[fmin, fmax]``."""

    freq, amp = compute_power_spectrum(signal, fs, nfft=nfft)
    mask = (freq >= float(fmin)) & (freq <= float(fmax))
    if not mask.any():
        return float(fallback)
    peaks, _ = find_peaks(amp[mask])
    valid_idx = np.flatnonzero(mask)
    if peaks.size:
        idx = valid_idx[peaks[int(np.argmax(amp[valid_idx][peaks]))]]
    else:
        idx = valid_idx[int(np.argmax(amp[valid_idx]))]
    return float(freq[idx])


def fft_peak_candidates(
    signal: np.ndarray,
    fs: float,
    hr_band_bpm: tuple[float, float],
    percent: float,
    *,
    nfft: int = 1 << 13,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference-style local FFT peak candidates in an HR band."""

    sig = np.asarray(signal, dtype=float).ravel()
    if sig.size == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    spectrum = np.fft.fft(sig, int(nfft))
    amp_full = np.abs(spectrum) / max(sig.size, 1)
    half = int(nfft) // 2
    amp = amp_full[:half].copy()
    amp[1:] *= 2.0
    freq = float(fs) * np.arange(half, dtype=float) / float(nfft)
    peaks_idx, _ = find_peaks(amp)
    if peaks_idx.size == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    low_hz = float(hr_band_bpm[0]) / 60.0
    high_hz = float(hr_band_bpm[1]) / 60.0
    valid = (freq[peaks_idx] >= low_hz) & (freq[peaks_idx] <= high_hz)
    valid_idx = peaks_idx[valid]
    if valid_idx.size == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    threshold = float(np.max(amp[valid_idx])) * float(percent)
    keep_idx = valid_idx[amp[valid_idx] > threshold]
    if keep_idx.size == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    return freq[keep_idx], amp[keep_idx]


def sparse_spectrum_hr_candidates(
    signal: np.ndarray,
    fs: float,
    *,
    previous_hr: float | None,
    hr_band_bpm: tuple[float, float],
    penalty_ref: np.ndarray | None = None,
    penalty_width_hz: float = 0.2,
    penalty_weight: float = 0.4,
    num_atoms: int = 5,
    lambda_threshold: float = 0.15,
    harmonic_tol_bpm: float = 5.0,
    grid_resolution_bpm: float = 1.0,
    slew_limit_bpm: float = 10.0,
    slew_step_bpm: float = 7.0,
) -> dict[str, object]:
    """Troika-style sparse spectrum peak selection for HR postprocessing.

    This is a lightweight peak-extraction branch, not a full TROIKA pipeline.
    It uses only the PPG window, an optional motion penalty reference, and the
    previous HR estimate.
    """

    sig = prepare_fft_window(signal, apply_hamming=True, demean=True)
    if sig.size < 4 or not np.isfinite(sig).any():
        return _empty_sparse_result("signal too short")
    resolution = float(grid_resolution_bpm)
    if not np.isfinite(resolution) or resolution <= 0.0:
        resolution = 1.0
    low_bpm, high_bpm = float(hr_band_bpm[0]), float(hr_band_bpm[1])
    if low_bpm >= high_bpm:
        return _empty_sparse_result("invalid HR band")
    bpm_grid = np.arange(low_bpm, high_bpm + resolution * 0.5, resolution, dtype=float)
    bpm_grid = bpm_grid[bpm_grid <= high_bpm + 1e-9]
    if bpm_grid.size == 0:
        return _empty_sparse_result("empty HR grid")

    t = np.arange(sig.size, dtype=float) / float(fs)
    freqs_hz = bpm_grid / 60.0
    basis = np.exp(-2j * np.pi * freqs_hz[:, None] * t[None, :])
    scores = np.abs(basis @ sig) / max(sig.size, 1)
    scores[~np.isfinite(scores)] = 0.0

    penalty_freq_hz = float("nan")
    if penalty_ref is not None and np.asarray(penalty_ref).size:
        penalty_freq_hz = dominant_frequency_in_band(penalty_ref, fs, 0.2, 5.0)
        if np.isfinite(penalty_freq_hz) and penalty_freq_hz > 0.0:
            mask = (np.abs(freqs_hz - penalty_freq_hz) < float(penalty_width_hz)) | (
                np.abs(freqs_hz - 2.0 * penalty_freq_hz) < float(penalty_width_hz)
            )
            scores[mask] *= float(penalty_weight)

    max_score = float(np.max(scores)) if scores.size else 0.0
    threshold = max_score * float(lambda_threshold)
    if not np.isfinite(threshold) or threshold <= 0.0:
        threshold = 0.0
    peak_idx, _ = find_peaks(scores)
    if peak_idx.size == 0 and scores.size:
        peak_idx = np.asarray([int(np.argmax(scores))], dtype=int)
    keep = peak_idx[scores[peak_idx] >= threshold]
    if keep.size == 0:
        return {
            **_empty_sparse_result("no SSR candidates above threshold"),
            "candidate_bpm": [],
            "candidate_score": [],
            "penalty_freq_hz": penalty_freq_hz,
        }
    order = keep[np.argsort(scores[keep])[::-1]]
    order = order[: max(1, int(num_atoms))]
    candidates = bpm_grid[order].astype(float)
    candidate_scores = scores[order].astype(float)
    chosen, harmonic_adjusted = _choose_sparse_hr_candidate(
        candidates,
        candidate_scores,
        previous_hr=previous_hr,
        harmonic_tol_bpm=float(harmonic_tol_bpm),
        slew_limit_bpm=float(slew_limit_bpm),
        slew_step_bpm=float(slew_step_bpm),
    )
    return {
        "status": "ok",
        "reason": "",
        "hr_bpm": float(chosen),
        "candidate_bpm": candidates.tolist(),
        "candidate_score": candidate_scores.tolist(),
        "harmonic_adjusted": bool(harmonic_adjusted),
        "penalty_freq_hz": penalty_freq_hz,
    }


def _choose_sparse_hr_candidate(
    candidates: np.ndarray,
    scores: np.ndarray,
    *,
    previous_hr: float | None,
    harmonic_tol_bpm: float,
    slew_limit_bpm: float,
    slew_step_bpm: float,
) -> tuple[float, bool]:
    """Choose one sparse-spectrum candidate with simple harmonic tracking."""

    cand = np.asarray(candidates, dtype=float)
    score = np.asarray(scores, dtype=float)
    finite = np.isfinite(cand) & np.isfinite(score)
    cand = cand[finite]
    score = score[finite]
    if cand.size == 0:
        return float("nan"), False
    prev_ok = previous_hr is not None and np.isfinite(previous_hr) and float(previous_hr) > 0.0
    if not prev_ok:
        return float(cand[int(np.argmax(score))]), False
    prev = float(previous_hr)
    harmonic_adjusted = False
    suppressed: set[int] = set()
    for idx, value in enumerate(cand):
        half = value / 2.0
        half_matches = np.flatnonzero(np.abs(cand - half) <= harmonic_tol_bpm)
        if half_matches.size and abs(half - prev) < abs(value - prev):
            suppressed.add(idx)
            harmonic_adjusted = True
        double = value * 2.0
        double_matches = np.flatnonzero(np.abs(cand - double) <= harmonic_tol_bpm)
        if double_matches.size and abs(double - prev) < abs(value - prev):
            suppressed.add(idx)
            harmonic_adjusted = True

    allowed = np.asarray([idx for idx in range(cand.size) if idx not in suppressed], dtype=int)
    if allowed.size == 0:
        allowed = np.arange(cand.size, dtype=int)
    within = allowed[np.abs(cand[allowed] - prev) <= float(slew_limit_bpm)]
    pool = within if within.size else allowed
    idx = pool[np.lexsort((-score[pool], np.abs(cand[pool] - prev)))][0]
    raw = float(cand[int(idx)])
    diff = raw - prev
    if diff > float(slew_limit_bpm):
        return prev + float(slew_step_bpm), harmonic_adjusted
    if diff < -float(slew_limit_bpm):
        return prev - float(slew_step_bpm), harmonic_adjusted
    return raw, harmonic_adjusted


def _empty_sparse_result(reason: str) -> dict[str, object]:
    return {
        "status": "failed",
        "reason": str(reason),
        "hr_bpm": float("nan"),
        "candidate_bpm": [],
        "candidate_score": [],
        "harmonic_adjusted": False,
        "penalty_freq_hz": float("nan"),
    }
