"""Shared soft spectral policy; continuity protection belongs to the tracker."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks


def spectrum_peaks(spectrum: tuple[np.ndarray, np.ndarray], threshold: float,
                   band_hz: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Extract common unpenalized peak evidence for both estimators."""
    frequencies, scores = spectrum
    band = np.isfinite(frequencies) & np.isfinite(scores) & (frequencies >= band_hz[0]) & (frequencies <= band_hz[1])
    frequencies, scores = frequencies[band], scores[band]
    indices, _ = find_peaks(scores)
    if not indices.size and scores.size and np.max(scores) > 0:
        indices = np.array([int(np.argmax(scores))])
    if indices.size:
        indices = indices[scores[indices] > np.max(scores) * threshold]
    return frequencies[indices], scores[indices]


def estimate_confidence(amplitudes: np.ndarray) -> float:
    """Measure dominant motion peak separation from the runner-up."""
    values = np.asarray(amplitudes, dtype=float)
    values = np.sort(values[np.isfinite(values) & (values > 0)])[::-1]
    if not values.size:
        return 0.0
    return 1.0 if values.size == 1 else float((values[0] - values[1]) / values[0])


def penalty_weights(frequencies_hz: np.ndarray, centers_hz: tuple[float, ...],
                    width_hz: float, retention: float) -> np.ndarray:
    """Triangular attenuation: retained center score, unity at/outside half-width.

    Overlap takes the smaller weight rather than repeatedly multiplying weights.
    """
    frequencies = np.asarray(frequencies_hz, dtype=float)
    weights = np.ones(frequencies.shape, dtype=float)
    if width_hz > 0:
        for center in centers_hz:
            distance = np.abs(frequencies - center)
            inside = distance < width_hz
            weights[inside] = np.minimum(weights[inside],
                retention + (1.0 - retention) * distance[inside] / width_hz)
    return weights


@dataclass(frozen=True)
class PenaltyPolicy:
    """Estimator-independent motion centers, confidence and Hz half-width."""

    centers_hz: tuple[float, ...]
    confidence: float
    width_hz: float
    retention: float

    @classmethod
    def build(cls, motion_peaks_hz: np.ndarray, motion_amplitudes: np.ndarray,
              ppg_fft_peaks_hz: np.ndarray, *, width_hz: float,
              base_retention: float) -> PenaltyPolicy:
        """Build harmonic evidence from common unpenalized PPG FFT peaks."""
        peaks = np.asarray(motion_peaks_hz, dtype=float)
        amplitudes = np.asarray(motion_amplitudes, dtype=float)
        valid = np.isfinite(peaks) & np.isfinite(amplitudes) & (amplitudes > 0)
        peaks, amplitudes = peaks[valid], amplitudes[valid]
        confidence = estimate_confidence(amplitudes)
        centers: list[float] = []
        if peaks.size:
            main = float(peaks[int(np.argmax(amplitudes))])
            centers.append(main)
            if np.any(np.abs(np.asarray(ppg_fft_peaks_hz) - 2 * main) <= max(width_hz, 1 / 60)):
                centers.append(2 * main)
        retention = 1.0 - max(confidence, 0.9) * (1.0 - base_retention)
        return cls(tuple(centers), confidence, float(width_hz), float(retention))

    def build_weights(self, frequencies_hz: np.ndarray) -> np.ndarray:
        """Return base weights without any previous-HR state."""
        return penalty_weights(frequencies_hz, self.centers_hz, self.width_hz, self.retention)
