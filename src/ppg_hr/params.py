"""Default parameter set for :func:`ppg_hr.core.heart_rate_solver.solve`."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from enum import Enum
from pathlib import Path
from typing import Any

__all__ = [
    "CascadeScheme",
    "ProtocolParams",
    "ProtocolSearchParams",
    "SolverParams",
    "TargetScope",
]


class CascadeScheme(str, Enum):
    """Adaptive-filter cascade variants used by the experimental protocol."""

    ACC3 = "ACC3"
    HF2 = "HF2"
    CF2 = "CF2"
    HF2_CF2 = "HF2_CF2"
    CF2_HF2 = "CF2_HF2"
    ACC3_HF2 = "ACC3_HF2"
    HF2_ACC3 = "HF2_ACC3"


class TargetScope(str, Enum):
    """Window groups where adaptive filtering is applied."""

    MOTION_ONLY = "motion_only"
    MOTION_AND_RECOVERY = "motion_recovery"


@dataclass
class ProtocolParams:
    """Batch adaptive protocol runtime defaults.

    These settings are intentionally separate from :class:`SolverParams`; the
    existing solver/CLI keeps its MATLAB-aligned defaults unchanged.
    """

    fs_origin: int = 100
    max_iterations: int = 250
    num_repeats: int = 3
    random_state: int = 42
    num_seed_points: int = 10
    penalty_value: float = 999.0
    parallel_repeats: int = 1
    debug_mode: bool = False


@dataclass
class ProtocolSearchParams:
    """Discrete search grid for the batch adaptive protocol."""

    Fs_Target: list[int] = field(default_factory=lambda: [25, 50])
    TW: list[int] = field(default_factory=lambda: [6, 8, 10])
    Kstop: list[float] = field(default_factory=lambda: [0.2, 0.3, 0.5])
    max_order: list[int] = field(default_factory=lambda: [8, 12, 16, 20])
    M_base: list[int] = field(default_factory=lambda: [1, 2])
    C_scale: list[float] = field(default_factory=lambda: [1, 1.2, 1.5])
    K_max: list[int] = field(default_factory=lambda: [8, 12, 16, 20])
    Spec_Penalty_Width: list[float] = field(default_factory=lambda: [0.1, 0.2, 0.3])
    Spec_Penalty_Weight: list[float] = field(default_factory=lambda: [0.1, 0.2, 0.4])
    smooth_win_len: list[int] = field(default_factory=lambda: [3, 5, 7])
    hr_range_hz: list[float] = field(
        default_factory=lambda: [x / 60.0 for x in (15, 20, 25, 30, 35, 40)]
    )
    slew_limit_bpm: list[int] = field(default_factory=lambda: list(range(8, 16)))
    slew_step_bpm: list[int] = field(default_factory=lambda: [5, 7, 9])

    def names(self) -> list[str]:
        """Return active parameter names in stable dataclass order."""
        return [f.name for f in fields(self)]

    def options(self, name: str) -> list[Any]:
        """Return the candidate list for ``name``."""
        if not hasattr(self, name):
            raise KeyError(name)
        values = getattr(self, name)
        return list(values)


@dataclass
class SolverParams:
    """All knobs accepted by the heart-rate solver.

    Field defaults match the MATLAB reference (``HeartRateSolver_cas_chengfa.m``
    + ``AutoOptimize_Bayes_Search_cas_chengfa.m``).
    """

    file_name: str | Path = ""
    ref_file: str | Path | None = None  # required when file_name is a CSV
    fs_target: int = 100
    max_order: int = 16

    time_start: float = 1.0
    time_buffer: float = 10.0
    calib_time: float = 30.0

    motion_th_scale: float = 2.5
    spec_penalty_enable: bool = True
    spec_penalty_weight: float = 0.2
    spec_penalty_width: float = 0.2

    hr_range_hz: float = 25.0 / 60.0
    slew_limit_bpm: float = 10.0
    slew_step_bpm: float = 7.0

    hr_range_rest: float = 30.0 / 60.0
    slew_limit_rest: float = 6.0
    slew_step_rest: float = 4.0

    smooth_win_len: int = 7
    time_bias: float = 5.0

    # LMS cascade fixed parameters
    num_cascade_hf: int = 2
    num_cascade_acc: int = 3
    lms_mu_base: float = 0.01

    # Bandpass filter
    bp_low_hz: float = 0.5
    bp_high_hz: float = 5.0
    bp_order: int = 4

    # Adaptive filter selection (new in 2026-04)
    adaptive_filter: str = "lms"  # one of: "lms", "klms", "volterra"
    ppg_mode: str = "green"  # one of: "green", "red", "ir"

    # Delay-search prefit controls. ``adaptive`` narrows the PPG-vs-motion
    # lag search per dataset; ``fixed`` preserves the original +/-0.2 s scan.
    delay_search_mode: str = "adaptive"  # one of: "adaptive", "fixed"
    delay_prefit_max_seconds: float = 0.2
    delay_prefit_windows: int = 8
    delay_prefit_min_corr: float = 0.15
    delay_prefit_margin_samples: int = 2
    delay_prefit_min_span_samples: int = 2

    # KLMS-specific parameters (only used when adaptive_filter == "klms")
    klms_step_size: float = 0.1
    klms_sigma: float = 1.0
    klms_epsilon: float = 0.1

    # Volterra-specific parameters (only used when adaptive_filter == "volterra")
    volterra_max_order_vol: int = 3

    extras: dict[str, Any] = field(default_factory=dict)

    def replace(self, **changes) -> SolverParams:
        """Return a copy with the given fields overridden."""
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}
