"""Default parameter set for :func:`ppg_hr.core.heart_rate_solver.solve`.

中文说明：本模块只保存协议级枚举和默认参数，不直接读写数据。实验入口会
根据这些 dataclass 生成 Optuna 搜索空间，并保持旧 solver 参数的向后兼容。
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from enum import Enum
from pathlib import Path
from typing import Any

__all__ = [
    "CascadeScheme",
    "MotionType",
    "ProtocolParams",
    "ProtocolSearchParams",
    "SolverParams",
    "TargetScope",
]


class MotionType(str, Enum):
    """Canonical motion types accepted by the subject-directory protocol."""

    WRITE = "write"
    GRIPPER = "gripper"
    RUN = "run"
    ROPE = "rope"


class CascadeScheme(str, Enum):
    """Adaptive-filter cascade variants used by the experimental protocol."""

    ACC = "ACC"
    HF2 = "HF2"
    UD2 = "UD2"
    ACC_HF2 = "ACC_HF2"
    HF2_CF2 = "HF2_CF2"
    ACC_UD2 = "ACC_UD2"

    # Source compatibility for code that referenced the former member names.
    ACC3 = "ACC"
    ACC3_HF2 = "ACC_HF2"

    @property
    def display_name(self) -> str:
        """Return the ordered user-facing cascade label."""

        return self.value.replace("_", "+")

    @classmethod
    def _missing_(cls, value: object) -> "CascadeScheme | None":
        """Read the two renamed legacy identifiers without exposing them for training."""

        return {
            "ACC3": cls.ACC,
            "ACC3_HF2": cls.ACC_HF2,
        }.get(str(value))


class TargetScope(str, Enum):
    """Window groups where adaptive filtering is applied."""

    MOTION_ONLY = "motion_only"
    MOTION_AND_RECOVERY = "motion_recovery"
    MOTION_POST10 = "motion_post10"
    GLOBAL = "global"

    @classmethod
    def _missing_(cls, value: object) -> "TargetScope | None":
        """Accept user-facing aliases for whole-record/global objectives."""

        text = str(value).strip().lower()
        if text in {"all", "global_all"}:
            return cls.GLOBAL
        return None


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
    """Discrete search grid for the batch adaptive protocol.

    中文说明：公共项对 LMS/Volterra/RFF-LMS/KLMS 四类滤波器都生效；滤波器专属项
    由 ``names_for_filter`` 动态选择，避免 LMS 误采样 RFF 或 Volterra 参数。
    """

    Fs_Target: list[int] = field(default_factory=lambda: [25, 50])
    TW: list[int] = field(default_factory=lambda: [6, 8, 10])
    Kstop: list[float] = field(default_factory=lambda: [0.2, 0.3, 0.5])
    max_order: list[int] = field(default_factory=lambda: [8, 12, 16, 20])
    M_base: list[int] = field(default_factory=lambda: [1, 2])
    C_scale: list[float] = field(default_factory=lambda: [0.6, 0.9, 1.2, 1.5])
    K_max: list[int] = field(default_factory=lambda: [8, 12, 16, 20, 30])
    Spec_Penalty_Width: list[float] = field(default_factory=lambda: [0.1, 0.2, 0.3])
    smooth_win_len: list[int] = field(default_factory=lambda: [5, 7, 9])
    hr_range_hz: list[float] = field(
        default_factory=lambda: [x / 60.0 for x in (20, 25, 30, 35, 40)]
    )
    slew_limit_bpm: list[int] = field(default_factory=lambda: [8, 10, 12, 14])
    slew_step_bpm: list[int] = field(default_factory=lambda: [5, 7, 9])
    Rest_HR_Track_Band_BPM: list[float] = field(
        default_factory=lambda: [20.0, 30.0, 50.0, 60.0, 80.0]
    )
    Rest_HR_Slew_Limit_BPM: list[float] = field(
        default_factory=lambda: [1.0, 3.0, 5.0, 6.0, 8.0, 25.0]
    )
    Rest_HR_Slew_Step_BPM: list[float] = field(
        default_factory=lambda: [0.5, 2.0, 4.0, 5.0, 8.0, 12.0]
    )
    LMS_Mu_Base: list[float] = field(default_factory=lambda: [0.004, 0.006, 0.008])
    RFF_LMS_Mu_Base: list[float] = field(default_factory=lambda: [0.001, 0.002, 0.004, 0.006])
    alpha_u: list[float] = field(default_factory=lambda: [0.005, 0.01, 0.03, 0.05, 0.1])
    M2: list[int] = field(default_factory=lambda: [2, 3])
    rff_D: list[int] = field(default_factory=lambda: [50, 100, 200])
    rff_sigma_scale: list[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 4.0])
    # Deprecated: kept for old CSV/JSON replay. New training searches rff_sigma_scale.
    rff_sigma: list[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 5.0])
    klms_step_size: list[float] = field(default_factory=lambda: [0.005, 0.01, 0.02, 0.05])
    klms_sigma: list[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 5.0])
    klms_epsilon: list[float] = field(default_factory=lambda: [0.005, 0.01, 0.02, 0.05, 0.1])

    def names(self) -> list[str]:
        """Return active parameter names in stable dataclass order."""
        return [f.name for f in fields(self)]

    def names_for_filter(self, adaptive_filter: str) -> list[str]:
        """Return parameter names that should be sampled for one filter.

        中文说明：RFF-LMS 的 ``LMS_Mu_Base`` 使用独立候选列表，但解码后仍写回同名
        字段，便于级联求解器统一读取步长基准。
        """

        common = [
            "Fs_Target",
            "TW",
            "Kstop",
            "max_order",
            "M_base",
            "C_scale",
            "K_max",
            "Spec_Penalty_Width",
            "hr_range_hz",
            "slew_limit_bpm",
            "slew_step_bpm",
        ]
        if adaptive_filter == "lms":
            return [*common, "LMS_Mu_Base"]
        if adaptive_filter == "volterra":
            return [*common, "LMS_Mu_Base", "alpha_u", "M2"]
        if adaptive_filter == "rff_lms":
            return [*common, "RFF_LMS_Mu_Base", "rff_D", "rff_sigma_scale"]
        if adaptive_filter == "klms":
            return [*common, "klms_step_size", "klms_sigma", "klms_epsilon"]
        raise ValueError(f"Unsupported adaptive_filter: {adaptive_filter}")

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
    adaptive_filter: str = "lms"  # one of: "lms", "volterra", "rff_lms", "klms"
    ppg_mode: str = "green"  # one of: "green", "red", "ir"

    # Delay-search prefit controls. ``adaptive`` narrows the PPG-vs-motion
    # lag search per dataset; ``fixed`` preserves the original +/-0.2 s scan.
    delay_search_mode: str = "adaptive"  # one of: "adaptive", "fixed"
    delay_prefit_max_seconds: float = 0.2
    delay_prefit_windows: int = 8
    delay_prefit_min_corr: float = 0.15
    delay_prefit_margin_samples: int = 2
    delay_prefit_min_span_samples: int = 2

    # Adaptive-filter shared lower bound. LMS/Volterra/RFF-LMS all clamp mu with it.
    lms_mu_min: float = 1e-6

    # Volterra-specific parameters (only used when adaptive_filter == "volterra")
    volterra_alpha_u: float = 0.1
    volterra_M2: int = 3

    # RFF-LMS-specific parameters (only used when adaptive_filter == "rff_lms")
    rff_D: int = 100
    rff_sigma: float = 1.0
    rff_seed: int = 42

    # KLMS-specific parameters (only used when adaptive_filter == "klms")
    klms_step_size: float = 0.05
    klms_sigma: float = 1.0
    klms_epsilon: float = 0.1

    extras: dict[str, Any] = field(default_factory=dict)

    def replace(self, **changes) -> SolverParams:
        """Return a copy with the given fields overridden."""
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}
