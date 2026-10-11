"""Sensor-only evidence types shared by optional reference mechanisms."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class DirectionalTrackingParams:
    range_up_bpm: float
    range_down_bpm: float
    limit_up_bpm: float
    step_up_bpm: float
    limit_down_bpm: float
    step_down_bpm: float


@dataclass(frozen=True)
class RawFftCandidateFrame:
    frequencies_hz: np.ndarray
    amplitudes: np.ndarray
    peak_indices: np.ndarray
    ordered_peak_indices: np.ndarray

    def top(self, count: int = 5) -> tuple[tuple[float, float], ...]:
        return tuple((float(self.frequencies_hz[i] * 60), float(self.amplitudes[i]))
                     for i in self.ordered_peak_indices[:count])
