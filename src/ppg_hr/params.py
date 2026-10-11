"""Central protocol, detector, and concrete trial parameters.

中文说明：本模块只保存协议级枚举和默认参数，不直接读写数据。实验入口会
根据这些 dataclass 生成 Optuna 搜索空间，参数只在本模块定义。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from types import MappingProxyType
from typing import Any

RESULT_SCHEMA_VERSION: int = 1

__all__ = [
    "RESULT_SCHEMA_VERSION",
    "ReferenceConfig",
    "DEFAULT_REFERENCE_CONFIGS",
    "MotionType",
    "ProtocolParams",
    "ProtocolSearchParams",
    "ProtocolTrialParams",
    "TargetScope",
]


@dataclass(frozen=True)
class MotionDetectorParams:
    """Fixed source-domain IMU activity detector settings."""

    window_s: float = 8.0
    step_s: float = 1.0
    baseline_s: float = 30.0
    filter_order: int = 4
    acc_band_hz: tuple[float, float] = (0.5, 5.0)
    gyro_band_hz: tuple[float, float] = (0.5, 10.0)
    baseline_multiplier: float = 2.5
    peak_fraction: float = 0.05
    bridge_gap_windows: int = 3
    minimum_run_windows: int = 5


DEFAULT_MOTION_DETECTOR = MotionDetectorParams()
DEFAULT_MOTION_POLICY = "single_bout"


@dataclass(frozen=True)
class Physical4DSettings:
    """Fixed grid and strict alignment/replay settings, in physical units."""

    sampling_rates: tuple[int, ...] = (25, 50, 100)
    memories_ms: tuple[int, ...] = (40, 80, 120, 160, 200)
    penalty_widths_bpm: tuple[int, ...] = (3, 6, 12, 18)
    lms_steps: tuple[float, ...] = (0.006, 0.008, 0.010, 0.012, 0.016)
    volterra_steps: tuple[float, ...] = (0.002, 0.004, 0.008, 0.012, 0.016)
    rff_lms_steps: tuple[float, ...] = (0.001, 0.002, 0.004, 0.006, 0.008)
    klms_steps: tuple[float, ...] = (0.005, 0.010, 0.020, 0.050, 0.100)
    tdelay_range_s: tuple[float, float] = (-5.0, 5.0)
    tdelay_step_s: float = 0.1
    alignment_min_windows: int = 2
    accuracy_threshold_bpm: float = 5.0
    replay_metric_atol: float = 1e-6
    alignment_tie_rule: str = "aae_abs_delay_rmse_candidate_order"
    alignment_fallback: str = "zero_delay_continue"


@dataclass(frozen=True)
class PreprocessingParams:
    """Source-domain numerical controls; shared by all coordinates."""

    fs_origin: int = 100
    filter_order: int = 4
    ppg_band_hz: tuple[float, float] = (0.5, 5.0)
    thermal_band_hz: tuple[float, float] = (0.1, 5.0)
    imu_band_hz: tuple[float, float] = (0.5, 10.0)
    cutoff_nyquist_fraction: float = 0.45
    outlier_window_s: float = 1.0
    missing_repair: str = "linear_interpolation_edge_hold"
    ud_mode: str = "hf_minus_cf"


PHYSICAL4D_SETTINGS = Physical4DSettings()
DEFAULT_PREPROCESSING = PreprocessingParams()


class MotionType(str, Enum):
    """Canonical motion types accepted by the subject-directory protocol."""

    WRITE = "write"
    GRIPPER = "gripper"
    RUN = "run"
    ROPE = "rope"


REFERENCE_REGISTRY = MappingProxyType({
    "ACC": ("accx", "accy", "accz"), "HF": ("hf1", "hf2"),
    "CF": ("cf1", "cf2"), "UD": ("ud1", "ud2"),
})


@dataclass(frozen=True)
class ReferenceConfig:
    """One source of truth: ordered groups expand to fixed channel order."""

    reference_groups_order: tuple[str, ...]

    def __post_init__(self) -> None:
        groups = self.reference_groups_order
        if isinstance(groups, ReferenceConfig):
            groups = groups.reference_groups_order
        if not isinstance(groups, tuple) or not groups:
            raise ValueError("reference_groups_order must be a nonempty tuple")
        if len(set(groups)) != len(groups) or any(group not in REFERENCE_REGISTRY for group in groups):
            raise ValueError(f"Invalid reference groups: {groups}")
        object.__setattr__(self, "reference_groups_order", groups)

    @property
    def reference_sequence(self) -> tuple[str, ...]:
        return tuple(channel for group in self.reference_groups_order for channel in REFERENCE_REGISTRY[group])

    @property
    def reference_order_key(self) -> str:
        return "+".join(self.reference_groups_order)

    @classmethod
    def from_key(cls, key: str) -> ReferenceConfig:
        """Parse the current canonical artifact identity; no legacy aliases."""
        config = cls(tuple(key.split("+")))
        if config.reference_order_key != key:
            raise ValueError(f"Noncanonical reference order key: {key}")
        return config


DEFAULT_REFERENCE_CONFIGS = tuple(ReferenceConfig(groups) for groups in
    (("ACC",), ("HF",), ("UD",), ("ACC", "HF"), ("HF", "CF"), ("ACC", "UD")))


class TargetScope(str, Enum):
    """Window groups where adaptive filtering is applied."""

    MOTION_ONLY = "motion_only"
    MOTION_AND_RECOVERY = "motion_recovery"
    MOTION_POST10 = "motion_post10"
    GLOBAL = "global"


@dataclass
class ProtocolParams:
    """Batch adaptive protocol runtime defaults.

    Runtime settings for the Notebook batch experiment.
    """

    motion_policy: str = DEFAULT_MOTION_POLICY
    fs_origin: int = 100
    max_iterations: int = 200
    num_repeats: int = 1
    random_state: int = 42
    num_seed_points: int = 10
    penalty_value: float = 999.0
    parallel_repeats: int = 1
    debug_mode: bool = False


@dataclass
class ProtocolSearchParams:
    """Four-dimensional core search; filter-specific step lists stay distinct."""

    Fs_Target: list[int] = field(default_factory=lambda: [25, 50])
    memory_ms: list[int] = field(default_factory=lambda: [40, 80, 120, 160, 200])
    LMS_Mu_Base: list[float] = field(default_factory=lambda: [0.004, 0.006, 0.008])
    Spec_Penalty_Width: list[float] = field(default_factory=lambda: [0.1, 0.2, 0.3])
    rff_step_candidates: list[float] = field(default_factory=lambda: [0.001, 0.002, 0.004, 0.006])
    klms_step_size: list[float] = field(default_factory=lambda: [0.005, 0.01, 0.02, 0.05])

    def names(self) -> list[str]:
        """Return the four LMS search coordinates."""
        return self.names_for_filter("lms")

    def names_for_filter(self, adaptive_filter: str) -> list[str]:
        """Choose only the approved step-size coordinate for this filter."""
        if adaptive_filter not in {"lms", "volterra", "rff_lms", "klms"}:
            raise ValueError(f"Unsupported adaptive_filter: {adaptive_filter}")
        step = "klms_step_size" if adaptive_filter == "klms" else "LMS_Mu_Base"
        return ["Fs_Target", "memory_ms", step, "Spec_Penalty_Width"]

    def options(self, name: str, *, adaptive_filter: str = "lms") -> list[Any]:
        """Get candidates, using RFF's established list under LMS_Mu_Base."""
        if name not in self.names_for_filter(adaptive_filter):
            raise KeyError(name)
        values = self.rff_step_candidates if name == "LMS_Mu_Base" and adaptive_filter == "rff_lms" else getattr(self, name)
        return list(values)


@dataclass(frozen=True)
class ProtocolTrialParams:
    """Concrete parameter values for one protocol trial.

    中文说明：这里同时保存公共参数、滤波器类型和滤波器专属参数。未被当前
    ``adaptive_filter`` 使用的字段会保留默认值，便于旧代码继续构造对象。
    """

    motion_policy: str = DEFAULT_MOTION_POLICY
    Fs_Target: int = 100
    hr_window_s: float = 8.0
    hr_step_s: float = 1.0
    # 中文说明：prewarm_s 是自适应滤波前置收敛上下文长度，固定实验参数，不进入
    # Optuna/Bayes 搜索空间；0 表示沿用旧的单 hr_window_s 窗口行为。
    prewarm_s: float = 0.0
    normalization_mode: str = "minmax"
    Kstop: float = 0.3
    memory_ms: float = 160.0
    M_base: int = 2
    C_scale: float = 1.2
    K_max: int = 12
    Spec_Penalty_Width: float = 0.2
    Spec_Penalty_Weight: float = 0.2
    smooth_win_len: int = 7
    hr_range_hz: float = 25.0 / 60.0
    slew_limit_bpm: int = 10
    slew_step_bpm: int = 7
    postprocess_method: str = "fft"
    SSR_Num_Atoms: int = 5
    SSR_Lambda: float = 0.15
    SSR_Harmonic_Tol_BPM: float = 5.0
    SSR_Fallback_To_FFT: bool = True
    SSR_Grid_Resolution_BPM: float = 1.0
    LMS_Mu_Base: float = 0.01
    LMS_Mu_Min: float = 1e-6
    adaptive_filter: str = "lms"
    objective_mode: str = "posthoc_aae"
    # 中文说明：窗口级 QC 的默认策略是部署保守的 baseline 回退；该参数是固定实验
    # 配置，不进入 Optuna/Bayes 搜索空间。
    qc_policy: str = "fallback_baseline"
    # 中文说明：默认保留旧的 Hilbert 包络时延估计；Notebook/脚本可显式改成 direct。
    delay_estimation_mode: str = "envelope"
    # 中文说明：evaluation_window_s 专用于静息段 PPG-HR 提取和全局 Tdelay 搜索；
    # hr_window_s 仍专用于后续自适应滤波训练/验证/测试切窗，二者不能混用。
    evaluation_window_s: float = 8.0
    evaluation_step_s: float = 1.0
    Rest_HR_Band_BPM: tuple[float, float] = (40.0, 180.0)
    Rest_HR_Track_Band_BPM: float = 30.0
    Rest_HR_Slew_Limit_BPM: float = 6.0
    Rest_HR_Slew_Step_BPM: float = 4.0
    Rest_HR_Smooth_Method: str = "median"
    Rest_HR_Smooth_Win: int = 3
    # 中文说明：以下参数只控制静息段 PPG-HR 的参考式频谱后处理，不进入 Optuna 搜索空间。
    # 运动惩罚默认开启以贴近参考仓库；只有调用方同时传入运动参考信号时才会真正生效。
    Rest_HR_Peak_Percent: float = 0.3
    Rest_HR_Spec_Penalty_Enable: bool = True
    Rest_HR_Spec_Penalty_Weight: float = 0.2
    Rest_HR_Spec_Penalty_Width_Hz: float = 0.2
    # 中文说明：静息段全局 Tdelay 搜索的主评分。AAE 更贴近最终误差指标；
    # ``std`` 保留旧逻辑用于复现实验，``mae`` 作为 ``aae`` 的兼容别名。
    Rest_Alignment_Score_Mode: str = "aae"
    # 中文说明：以下 time_bias_after_* 是自适应滤波后 HR 曲线的 post-hoc
    # 诊断对齐参数，只影响新增 posthoc 指标和手动重画，不进入 Optuna 搜索空间。
    Enable_Time_Bias_After: bool = True
    Time_Bias_After_Range_S: tuple[float, float] = (-5.0, 5.0)
    Time_Bias_After_Step_S: float = 1.0
    Time_Bias_After_Mode: str = "posthoc_oracle_alignment"
    handoff_wait_s: float = 5.0
    handoff_difference_bpm: float = 6.0
    handoff_confirm_windows: int = 2
    handoff_deadline_s: float = 20.0
    final_rest_up_limit_bpm: float = 1.5
    final_rest_up_step_bpm: float = 1.5
    final_rest_down_limit_bpm: float = 3.0
    final_rest_down_step_bpm: float = 1.5
    final_motion_up_limit_bpm: float = 5.5
    final_motion_up_step_bpm: float = 3.5
    final_motion_down_limit_bpm: float = 2.0
    final_motion_down_step_bpm: float = 1.5
    final_recovery_up_limit_bpm: float = 1.5
    final_recovery_up_step_bpm: float = 1.5
    final_recovery_down_limit_bpm: float = 3.5
    final_recovery_down_step_bpm: float = 3.0
    Recovery_Grace_S: float = 10.0
    Recovery_Diff_Bpm: float = 20.0
    Recovery_Cross_Diff_Bpm: float = 8.0
    alpha_u: float = 0.1
    M2: int = 3
    rff_D: int = 100
    rff_sigma_scale: float = 1.0
    rff_seed: int = 0
    # 中文说明：RFF-LMS 默认使用特征空间 NLMS，避免普通 LMS 在高维随机特征下
    # 因特征能量和大步长组合发散；字段不进入搜索空间，只作为工程固定策略。
    rff_update_mode: str = "nlms"
    rff_nlms_eps: float = 1e-9
    rff_leakage: float = 0.0
    rff_err_clip: float | None = None
    rff_theta_norm_guard: float | None = None
    klms_step_size: float = 0.05
    klms_sigma: float = 1.0
    # 中文说明：字段名沿用 klms_epsilon 以兼容旧 JSON；默认语义已改为
    # squared_distance / tap_dim 的归一化距离阈值。若要复现实验旧逻辑，可把
    # klms_distance_mode 设为 "absolute_squared"。
    klms_epsilon: float = 0.1
    klms_max_dictionary_size: int = 300
    klms_center_prune_policy: str = "freeze_new_centers"
    klms_distance_mode: str = "normalized"
    klms_normalized_update: bool = True
    klms_nlms_eps: float = 1e-6
    # 中文说明：PPG 输入策略和全局 objective/级联 guard 会改变实际 HR 结果，
    # 因此作为固定 trial 参数保存，并通过 cache_key 自动参与缓存隔离。
    ppg_input_transform: str = "log_absorbance"
    log_absorbance_baseline_mode: str = "rolling_median"
    log_absorbance_baseline_window_s: float = 5.0
    log_absorbance_eps: float = 1e-6
    log_absorbance_ratio_clip: tuple[float, float] = (1e-3, 1e3)
    global_objective_strategy: str = "current_global_adaptive"
    cascade_guard_policy: str = "none"
    cascade_guard_ratio_min: float = 0.05
    cascade_guard_ratio_max: float = 5.0
    cascade_guard_flat_std_eps: float = 1e-6
    cascade_guard_use_finite_zscore: bool = True
    # Enhanced spectrum tracker. These fixed fields are serialized and enter cache_key.
    tracker_mode: str = "enhanced"
    enable_directional_tracking: bool = True
    enable_dynamic_penalty: bool = True
    enable_continuity_protection: bool = True
    enable_low_lock_recovery: bool = False
    enable_high_lock_recovery: bool = False
    enable_post_motion_protection: bool = False
    # Phase-specific candidate tracking; no shared legacy override.
    motion_tracking_range_up_bpm: float = 35.0
    motion_tracking_range_down_bpm: float = 15.0
    motion_tracking_slew_limit_up_bpm: float = 5.5
    motion_tracking_slew_step_up_bpm: float = 3.5
    motion_tracking_slew_limit_down_bpm: float = 2.0
    motion_tracking_slew_step_down_bpm: float = 1.5
    recovery_tracking_range_up_bpm: float = 20.0
    recovery_tracking_range_down_bpm: float = 25.0
    recovery_tracking_slew_limit_up_bpm: float = 1.5
    recovery_tracking_slew_step_up_bpm: float = 1.5
    recovery_tracking_slew_limit_down_bpm: float = 3.5
    recovery_tracking_slew_step_down_bpm: float = 3.0
    candidate_peak_threshold_ratio: float = 0.30
    full_candidate_peak_threshold_ratio: float = 0.15
    low_lock_min_bpm: float = 50.0
    low_lock_max_bpm: float = 80.0
    low_lock_min_windows: int = 4
    low_lock_target_min_bpm: float = 90.0
    low_lock_min_jump_bpm: float = 20.0
    low_lock_min_amp_ratio: float = 0.45
    low_lock_candidate_stable_bpm: float = 10.0
    low_lock_confirm_windows: int = 3
    low_lock_step_bpm: float = 30.0
    high_lock_confirm_windows: int = 3
    high_lock_cooldown_windows: int = 4
    high_lock_min_gap_bpm: float = 20.0
    high_lock_min_amp_ratio: float = 0.45
    high_lock_candidate_min_bpm: float = 85.0
    high_lock_candidate_stable_bpm: float = 10.0
    high_lock_penalty_exclusion_bpm: float = 10.0
    high_lock_down_step_bpm: float = 20.0
    high_lock_up_step_bpm: float = 3.0
    # Isolated experimental policy parameters; the frozen core disables this mechanism.
    post_motion_guard_seconds: float | None = None
    post_motion_guard_min_elapsed_s: float = 5.0
    post_motion_guard_stable_windows: int = 3
    post_motion_guard_crossover_gap_bpm: float = 2.0
    post_motion_guard_upward_gap_bpm: float = 1.5
    post_motion_guard_fft_floor_bpm: float = 55.0
    post_motion_guard_recovery_step_up_bpm: float = 1.5
    post_motion_guard_recovery_step_down_bpm: float = 3.0
    post_motion_guard_rising_windows: int = 3
    post_motion_guard_rising_slope_bpm_per_window: float = 1.5
    post_motion_guard_rescue_gap_bpm: float = 20.0
    post_motion_guard_gap_rescue_enable: bool = False
    post_motion_guard_gap_rescue_windows: int = 4
    post_motion_guard_gap_rescue_min_hits: int = 3
    post_motion_guard_fft_stable_windows: int = 3
    post_motion_guard_fft_stable_bpm: float = 6.0

    def __post_init__(self) -> None:
        if self.hr_window_s != 8.0 or self.hr_step_s != 1.0:
            raise ValueError("HR observation is frozen at 8 s / 1 s")
        if self.evaluation_window_s != 8.0 or self.evaluation_step_s != 1.0:
            raise ValueError("evaluation observation is frozen at 8 s / 1 s")
        if self.memory_ms <= 0 or self.Fs_Target <= 0 or self.prewarm_s < 0:
            raise ValueError(
                "sampling, memory and prewarm must have valid physical units"
            )

    @property
    def history_tap_limit(self) -> int:
        """Positive-delay historical cap; forward K remains delay-derived."""
        return max(1, round(self.memory_ms * self.Fs_Target / 1000.0))

    def to_dict(self) -> dict[str, Any]:
        """Return parameter values as plain Python scalars."""

        return asdict(self)

    def cache_key(self) -> tuple[tuple[str, Any], ...]:
        """Return a stable cache key for one full mode/trial configuration.

        中文说明：RFF-LMS 的随机特征由 ``rff_seed`` 固定，因此 seed 必须进入缓存键，
        否则不同随机特征会误用同一组窗口结果。
        """

        return tuple(sorted(self.to_dict().items()))


# Literal snapshot: changing dataclass defaults must not change this preset.
CORE_PRESET_NAME = "core_enhanced_v1"
CORE_PRESET_VERSION = 1
REFERENCE_IMPLEMENTATION_SHA = "02d7e091219aeff78e82d53ae0069acf7bedb467"
CORE_FEATURE_FLAGS = MappingProxyType({
    "raw_imu_detector": True, "fixed_observation": True,
    "adaptive_filtering": True, "shared_soft_penalty": True,
    "continuity_protection": True, "directional_candidate_tracking": True,
    "simple_reset_fft_handoff": True, "phase_final_dynamics": True,
    "low_lock": False, "high_lock": False, "rise_lineage": False,
    "gap_rescue": False, "rising_rescue": False, "complex_shadow": False,
    "complex_post_motion_rescue": False, "experimental_rescue": False,
    "interface_quality_online": False,
})
_CORE_V1_VALUES = MappingProxyType({'C_scale': 1.2,
 'Enable_Time_Bias_After': True,
 'Fs_Target': 100,
 'K_max': 12,
 'Kstop': 0.3,
 'LMS_Mu_Base': 0.01,
 'LMS_Mu_Min': 1e-06,
 'M2': 3,
 'M_base': 2,
 'Recovery_Cross_Diff_Bpm': 8.0,
 'Recovery_Diff_Bpm': 20.0,
 'Recovery_Grace_S': 10.0,
 'Rest_Alignment_Score_Mode': 'aae',
 'Rest_HR_Band_BPM': (40.0, 180.0),
 'Rest_HR_Peak_Percent': 0.3,
 'Rest_HR_Slew_Limit_BPM': 6.0,
 'Rest_HR_Slew_Step_BPM': 4.0,
 'Rest_HR_Smooth_Method': 'median',
 'Rest_HR_Smooth_Win': 3,
 'Rest_HR_Spec_Penalty_Enable': True,
 'Rest_HR_Spec_Penalty_Weight': 0.2,
 'Rest_HR_Spec_Penalty_Width_Hz': 0.2,
 'Rest_HR_Track_Band_BPM': 30.0,
 'SSR_Fallback_To_FFT': True,
 'SSR_Grid_Resolution_BPM': 1.0,
 'SSR_Harmonic_Tol_BPM': 5.0,
 'SSR_Lambda': 0.15,
 'SSR_Num_Atoms': 5,
 'Spec_Penalty_Weight': 0.2,
 'Spec_Penalty_Width': 0.2,
 'Time_Bias_After_Mode': 'posthoc_oracle_alignment',
 'Time_Bias_After_Range_S': (-5.0, 5.0),
 'Time_Bias_After_Step_S': 1.0,
 'adaptive_filter': 'lms',
 'alpha_u': 0.1,
 'candidate_peak_threshold_ratio': 0.3,
 'cascade_guard_flat_std_eps': 1e-06,
 'cascade_guard_policy': 'none',
 'cascade_guard_ratio_max': 5.0,
 'cascade_guard_ratio_min': 0.05,
 'cascade_guard_use_finite_zscore': True,
 'delay_estimation_mode': 'envelope',
 'enable_continuity_protection': True,
 'enable_directional_tracking': True,
 'enable_dynamic_penalty': True,
 'enable_high_lock_recovery': False,
 'enable_low_lock_recovery': False,
 'enable_post_motion_protection': False,
 'evaluation_step_s': 1.0,
 'evaluation_window_s': 8.0,
 'final_motion_down_limit_bpm': 2.0,
 'final_motion_down_step_bpm': 1.5,
 'final_motion_up_limit_bpm': 5.5,
 'final_motion_up_step_bpm': 3.5,
 'final_recovery_down_limit_bpm': 3.5,
 'final_recovery_down_step_bpm': 3.0,
 'final_recovery_up_limit_bpm': 1.5,
 'final_recovery_up_step_bpm': 1.5,
 'final_rest_down_limit_bpm': 3.0,
 'final_rest_down_step_bpm': 1.5,
 'final_rest_up_limit_bpm': 1.5,
 'final_rest_up_step_bpm': 1.5,
 'full_candidate_peak_threshold_ratio': 0.15,
 'global_objective_strategy': 'current_global_adaptive',
 'handoff_confirm_windows': 2,
 'handoff_deadline_s': 20.0,
 'handoff_difference_bpm': 6.0,
 'handoff_wait_s': 5.0,
 'high_lock_candidate_min_bpm': 85.0,
 'high_lock_candidate_stable_bpm': 10.0,
 'high_lock_confirm_windows': 3,
 'high_lock_cooldown_windows': 4,
 'high_lock_down_step_bpm': 20.0,
 'high_lock_min_amp_ratio': 0.45,
 'high_lock_min_gap_bpm': 20.0,
 'high_lock_penalty_exclusion_bpm': 10.0,
 'high_lock_up_step_bpm': 3.0,
 'hr_range_hz': 0.4166666666666667,
 'hr_step_s': 1.0,
 'hr_window_s': 8.0,
 'klms_center_prune_policy': 'freeze_new_centers',
 'klms_distance_mode': 'normalized',
 'klms_epsilon': 0.1,
 'klms_max_dictionary_size': 300,
 'klms_nlms_eps': 1e-06,
 'klms_normalized_update': True,
 'klms_sigma': 1.0,
 'klms_step_size': 0.05,
 'log_absorbance_baseline_mode': 'rolling_median',
 'log_absorbance_baseline_window_s': 5.0,
 'log_absorbance_eps': 1e-06,
 'log_absorbance_ratio_clip': (0.001, 1000.0),
 'low_lock_candidate_stable_bpm': 10.0,
 'low_lock_confirm_windows': 3,
 'low_lock_max_bpm': 80.0,
 'low_lock_min_amp_ratio': 0.45,
 'low_lock_min_bpm': 50.0,
 'low_lock_min_jump_bpm': 20.0,
 'low_lock_min_windows': 4,
 'low_lock_step_bpm': 30.0,
 'low_lock_target_min_bpm': 90.0,
 'memory_ms': 160.0,
 'motion_policy': 'single_bout',
 'motion_tracking_range_down_bpm': 15.0,
 'motion_tracking_range_up_bpm': 35.0,
 'motion_tracking_slew_limit_down_bpm': 2.0,
 'motion_tracking_slew_limit_up_bpm': 5.5,
 'motion_tracking_slew_step_down_bpm': 1.5,
 'motion_tracking_slew_step_up_bpm': 3.5,
 'normalization_mode': 'minmax',
 'objective_mode': 'posthoc_aae',
 'post_motion_guard_crossover_gap_bpm': 2.0,
 'post_motion_guard_fft_floor_bpm': 55.0,
 'post_motion_guard_fft_stable_bpm': 6.0,
 'post_motion_guard_fft_stable_windows': 3,
 'post_motion_guard_gap_rescue_enable': False,
 'post_motion_guard_gap_rescue_min_hits': 3,
 'post_motion_guard_gap_rescue_windows': 4,
 'post_motion_guard_min_elapsed_s': 5.0,
 'post_motion_guard_recovery_step_down_bpm': 3.0,
 'post_motion_guard_recovery_step_up_bpm': 1.5,
 'post_motion_guard_rescue_gap_bpm': 20.0,
 'post_motion_guard_rising_slope_bpm_per_window': 1.5,
 'post_motion_guard_rising_windows': 3,
 'post_motion_guard_seconds': None,
 'post_motion_guard_stable_windows': 3,
 'post_motion_guard_upward_gap_bpm': 1.5,
 'postprocess_method': 'fft',
 'ppg_input_transform': 'log_absorbance',
 'prewarm_s': 0.0,
 'qc_policy': 'fallback_baseline',
 'recovery_tracking_range_down_bpm': 25.0,
 'recovery_tracking_range_up_bpm': 20.0,
 'recovery_tracking_slew_limit_down_bpm': 3.5,
 'recovery_tracking_slew_limit_up_bpm': 1.5,
 'recovery_tracking_slew_step_down_bpm': 3.0,
 'recovery_tracking_slew_step_up_bpm': 1.5,
 'rff_D': 100,
 'rff_err_clip': None,
 'rff_leakage': 0.0,
 'rff_nlms_eps': 1e-09,
 'rff_seed': 0,
 'rff_sigma_scale': 1.0,
 'rff_theta_norm_guard': None,
 'rff_update_mode': 'nlms',
 'slew_limit_bpm': 10,
 'slew_step_bpm': 7,
 'smooth_win_len': 7,
 'tracker_mode': 'enhanced',





})
CORE_SEARCH_FIELDS = frozenset({"Fs_Target", "memory_ms", "LMS_Mu_Base", "klms_step_size", "Spec_Penalty_Width"})
CORE_EXPERIMENT_FIELDS = frozenset({"adaptive_filter", "postprocess_method", "motion_policy", "ppg_input_transform"})
# Evaluation settings never enter sensor prediction. Seed is derived per RFF trial.
CORE_EVALUATION_FIELDS = frozenset({"objective_mode", "Enable_Time_Bias_After", "Time_Bias_After_Range_S", "Time_Bias_After_Step_S", "Time_Bias_After_Mode", "rff_seed"})


def resolve_core_config(values: ProtocolTrialParams | dict[str, Any] | None = None) -> ProtocolTrialParams:
    """Resolve the immutable v1 snapshot and reject unknown/frozen overrides."""
    if {item.name for item in fields(ProtocolTrialParams)} != set(_CORE_V1_VALUES):
        raise RuntimeError("Protocol parameters changed without an explicit frozen preset snapshot")
    supplied = values.to_dict() if isinstance(values, ProtocolTrialParams) else dict(values or {})
    supplied = {key: tuple(value) if isinstance(_CORE_V1_VALUES.get(key), tuple) and isinstance(value, list) else value
                for key, value in supplied.items()}
    unknown = set(supplied) - set(_CORE_V1_VALUES)
    if unknown:
        raise ValueError(f"Unknown core parameters: {sorted(unknown)}")
    mutable = CORE_SEARCH_FIELDS | CORE_EXPERIMENT_FIELDS | CORE_EVALUATION_FIELDS
    changed = [key for key, value in supplied.items() if key not in mutable and value != _CORE_V1_VALUES[key]]
    if changed:
        raise ValueError(f"Frozen core_enhanced_v1 parameters: {sorted(changed)}")
    resolved = dict(_CORE_V1_VALUES)
    resolved.update(supplied)
    if resolved["adaptive_filter"] not in {"lms", "volterra", "rff_lms", "klms"}:
        raise ValueError("Unsupported adaptive_filter")
    unused_step = "LMS_Mu_Base" if resolved["adaptive_filter"] == "klms" else "klms_step_size"
    if resolved[unused_step] != _CORE_V1_VALUES[unused_step]:
        raise ValueError(f"Frozen unused filter parameter: {unused_step}")
    if resolved["postprocess_method"] not in {"fft", "ssr"}:
        raise ValueError("Unsupported spectrum estimator")
    if resolved["motion_policy"] not in {"single_bout", "multi_bout"}:
        raise ValueError("Unsupported motion protocol")
    if resolved["ppg_input_transform"] not in {"raw_bandpass", "log_absorbance"}:
        raise ValueError("Unsupported PPG transform")
    return ProtocolTrialParams(**resolved)


def core_preset_metadata(params: ProtocolTrialParams) -> dict[str, Any]:
    """Serialize the complete frozen contract alongside effective values."""
    return {"preset_name": CORE_PRESET_NAME, "preset_version": CORE_PRESET_VERSION,
            "feature_flags": dict(CORE_FEATURE_FLAGS), "core_parameters": params.to_dict(),
            "motion_detector": asdict(DEFAULT_MOTION_DETECTOR),
            "reference_implementation_sha": REFERENCE_IMPLEMENTATION_SHA}
