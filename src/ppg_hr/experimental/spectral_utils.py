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
