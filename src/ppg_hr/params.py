"""Central protocol, detector, and concrete trial parameters.

中文说明：本模块只保存协议级枚举和默认参数，不直接读写数据。实验入口会
根据这些 dataclass 生成 Optuna 搜索空间，参数只在本模块定义。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
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
    outlier_minimum_samples: int = 3
    outlier_threshold: float = 3.0
    outlier_mad_scale: float = 1.4826
    missing_repair: str = "linear_interpolation_edge_hold"
    ud_calculation_mode: str = "smoothed"
    ud_smoothing_window_s: float = 0.1
    ud_denominator_eps: float = 1e-9
    thermal_voltage_scale: float = 1000.0
    minimum_bandpass_samples: int = 8
    minimum_cutoff_hz: float = 1e-3
    bandpass_pad_multiplier: int = 3


PHYSICAL4D_SETTINGS = Physical4DSettings()
DEFAULT_PREPROCESSING = PreprocessingParams()


@dataclass(frozen=True)
class NumericalParams:
    """Numerical controls formerly embedded in signal/FFT/filter helpers."""

    fft_min_length: int = 8192
    input_zscore_std_eps: float = 1e-12
    window_normalization_eps: float = 1e-12
    lms_nlms_eps: float = 1e-9
    volterra_nlms_eps: float = 1e-9
    volterra_alpha_floor: float = 1e-6
    rff_min_sigma: float = 1e-6
    rff_max_distance_pairs: int = 2048
    klms_min_sigma: float = 1e-12
    candidate_band_hz: tuple[float, float] = (0.5, 4.0)
    penalty_reference_band_hz: tuple[float, float] = (0.2, 5.0)
    penalty_challenger_min_amp_ratio: float = 0.45
    penalty_confidence_floor: float = 0.9
    penalty_harmonic_min_half_width_hz: float = 1.0 / 60.0
    high_lock_risk_near_motion_bpm: float = 8.0
    dual_reset_raw_persistence_windows: int = 3
    dual_reset_raw_band_hz: tuple[float, float] = (0.7, 4.0)
    posthoc_min_valid_windows: int = 2


DEFAULT_NUMERICS = NumericalParams()


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
    # Physical4D 主入口显式选择 reference_parity；legacy 仅用于旧记录兼容。
    cascade_mode: str = "legacy"
    reference_lag_ms: float = 200.0
    correlation_scale_lms: bool = True
    correlation_scale_nonlinear: bool = False
    spectral_penalty_policy: str = "soft_visible"
    spectral_penalty_policy_version: int = 1
    visibility_match_half_width_bpm: float = 3.5
    visibility_motion_invocations: int = 3
    visibility_edge_guard_bpm: float = 1.0
    enable_rise_lineage: bool = False
    enable_post_motion_recovery: bool = False
    enable_post_motion_guard: bool = False
    enable_dual_reset: bool = False
    rise_lineage_confirm_windows: int = 3
    dual_reset_confirm_windows: int = 3
    dual_reset_gap_bpm: float = 6.0
    dual_reset_mechanism: str = "trend_persistence"
    dual_reset_prior_half_life_s: float = 10.0
    dual_reset_qualification_windows: int = 4
    dual_reset_trajectory_tolerance_bpm: float = 6.0
    dual_reset_min_amp_ratio: float = 0.25
    dual_reset_max_held_previous: int = 0
    dual_reset_readiness_hits: int = 2
    dual_reset_controlled_reanchor: bool = True
    dual_reset_reanchor_prior_guard_bpm: float = 45.0
    dual_reset_reanchor_min_gap_bpm: float = 25.0
    dual_reset_prior_invalidation_enable: bool = False
    dual_reset_prior_invalidation_hits: int = 3
    dual_reset_prior_invalidation_gap_bpm: float = 40.0
    dual_reset_prior_invalidation_raw_decline_bpm: float = 0.5
    dual_reset_prior_invalidation_prior_decline_bpm: float = 0.5
    rise_lineage_max_age_windows: int = 6
    rise_lineage_max_downward_drift_bpm: float = 1.0
    rise_lineage_harmonic_multiple: float = 3.0

    def __post_init__(self) -> None:
        if self.spectral_penalty_policy not in {"soft_visible", "suppressed_protected_continuous_visibility"}:
            raise ValueError("Unknown spectral penalty policy")
        if self.spectral_penalty_policy_version != 1:
            raise ValueError('Unsupported spectral penalty policy version')
        if self.cascade_mode == 'reference_parity':
            if self.delay_estimation_mode not in {'direct', 'envelope'} or self.reference_lag_ms < 0:
                raise ValueError('Invalid reference correlation configuration')
            if self.adaptive_filter not in {'lms', 'volterra', 'rff_lms', 'klms'}:
                raise ValueError('Unsupported adaptive filter')
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
    if isinstance(values, ProtocolTrialParams) and values.cascade_mode == "reference_parity":
        return values
    supplied = values.to_dict() if isinstance(values, ProtocolTrialParams) else dict(values or {})
    supplied = {key: tuple(value) if isinstance(_CORE_V1_VALUES.get(key), tuple) and isinstance(value, list) else value
                for key, value in supplied.items()}
    extended_defaults = ProtocolTrialParams().to_dict()
    added = set(extended_defaults) - set(_CORE_V1_VALUES)
    unknown = set(supplied) - set(extended_defaults)
    if unknown:
        raise ValueError(f"Unknown core parameters: {sorted(unknown)}")
    mutable = CORE_SEARCH_FIELDS | CORE_EXPERIMENT_FIELDS | CORE_EVALUATION_FIELDS
    changed = [key for key, value in supplied.items() if key not in mutable and value != _CORE_V1_VALUES.get(key, extended_defaults[key])]
    if changed:
        raise ValueError(f"Frozen core_enhanced_v1 parameters: {sorted(changed)}")
    resolved = dict(_CORE_V1_VALUES)
    resolved.update({key: extended_defaults[key] for key in added})
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
    if params.cascade_mode == 'reference_parity':
        return {"preset_name": "reference_parity", "preset_version": 1,
                "feature_flags": {**{k: v for k, v in params.to_dict().items() if k.startswith('enable_')},
                    "advanced_visibility": params.spectral_penalty_policy == 'suppressed_protected_continuous_visibility'},
                "core_parameters": params.to_dict(),
                "reference_implementation_sha": REFERENCE_IMPLEMENTATION_SHA}
    """Serialize the complete frozen contract alongside effective values."""
    return {"preset_name": CORE_PRESET_NAME, "preset_version": CORE_PRESET_VERSION,
            "feature_flags": dict(CORE_FEATURE_FLAGS), "core_parameters": params.to_dict(),
            "motion_detector": asdict(DEFAULT_MOTION_DETECTOR),
            "reference_implementation_sha": REFERENCE_IMPLEMENTATION_SHA}


def reference_parity_params(**overrides: Any) -> ProtocolTrialParams:
    """Explicit main-path defaults; retained nonlinear kernels use their own grid."""
    values = ProtocolTrialParams().to_dict()
    values.update(cascade_mode="reference_parity", delay_estimation_mode="direct",
                  M_base=1, C_scale=1.0, K_max=16, smooth_win_len=5,
                  Spec_Penalty_Weight=0.4, rff_seed=42)
    values.update(overrides)
    return ProtocolTrialParams(**values)



# 固定参数目录说明；文字不进入数值配置hash。
PARAMETER_NOTES = {'ACCURACY_THRESHOLD_BPM': '绝对HR误差不超过该bpm阈值计为准确；评价scope由manifest固定。',
 'ALIGNMENT_FALLBACK': '比较窗口不足时使用Tdelay=0，记录原因后继续计算。',
 'ALIGNMENT_MIN_WINDOWS': 'Tdelay候选至少需要的有效静息比较窗口数。',
 'ALIGNMENT_TIE_RULE': '固定AAE、绝对时延、RMSE、候选顺序依次处理平局。',
 'CANDIDATE_PEAK_THRESHOLD_RATIO': '运动惩罚参考峰相对主峰门限；参考默认 0.30。',
 'CASCADE_MODE': 'Physical4D固定reference_parity；legacy仅兼容旧调用，不能进入新训练。',
 'CASCADE_TRAIN_BUDGETS': '按 scheme 覆盖 trial/repeat 预算；None 使用全局预算。',
 'CLEAN_OUTPUTS': '是否清空当前运行目录；续跑 checkpoint 时必须保持 False。',
 'CONNECTIVITY_MAX_ITERATIONS': '连通性检查每模式 trial 数；默认 1，避免误跑正式预算。',
 'CORRELATION_SCALE_LMS': 'LMS默认从基础步长减去abs(correlation)/100并限制到mu_min。',
 'CORRELATION_SCALE_NONLINEAR': '非线性滤波器可选相同相关缩放，默认关闭以保留现有内核。',
 'C_SCALE': '参考固定缩放1；reference-parity的M严格按floor(abs(shared delay))。',
 'DEBUG_MODE': '调试模式；会保留更多中间记录并隐式启用 stage JSON。',
 'DELAY_ESTIMATION_MODE': 'direct为主路径；envelope为可选Hilbert包络相关实验。',
 'DUAL_RESET_RAW_BAND_HZ': 'dual-reset使用的原始FFT候选合法频带，默认0.7–4Hz。',
 'ENABLE_DUAL_RESET': '启用纯PPG与Final弱先验两条独立reset轨迹及可信目标交接，默认关闭。',
 'ENABLE_POST_MOTION_GUARD': '启用动态运动后adaptive/reset-FFT来源保护与交接，默认关闭。',
 'ENABLE_POST_MOTION_RECOVERY': '进入recovery时清空adaptive追踪历史；baseline仍延续，默认关闭。',
 'ENABLE_RISE_LINEAGE': '启用有期限的上升候选谱峰lineage确认及重锚，默认关闭。',
 'ENABLE_TIME_BIAS_AFTER': '是否计算滤波后 post-hoc 时间偏移诊断。',
 'EVALUATION_STEP_S': '评价诊断步长1秒，冻结。',
 'EVALUATION_WINDOW_S': '评价诊断窗8秒，冻结；不控制预测网格。',
 'FFT_MIN_LENGTH': 'FFT最短长度；实际取不小于当前信号长度的2次幂。',
 'FS_ORIGIN': '原始采样率(Hz)；change6/Pydisplay 数据固定按 100 Hz 与序号校验。',
 'FS_TARGET': '当前坐标采样率Hz，仅Physical4D坐标提供。',
 'HR_STEP_S': 'HR窗口hop秒数；固定1秒。',
 'HR_WINDOW_S': 'HR观察窗秒数；固定8秒，不进入Physical4D。',
 'K_MAX': '兼容参数固定16；reference-parity实际K为ACC=1，HF/CF/UD=0。',
 'LOG_ABSORBANCE_BASELINE_MODE': 'log-absorbance 基线估计方法；rolling_median 抑制慢漂移。',
 'LOG_ABSORBANCE_BASELINE_WINDOW_S': '滚动基线窗口长度(秒)；越大越保留低频趋势。',
 'LOG_ABSORBANCE_EPS': 'log 比值的数值稳定下限，防止零值导致无穷大。',
 'LOG_ABSORBANCE_RATIO_CLIP': '进入 log 前的比值裁剪范围，防止极端异常值。',
 'M_BASE': '参考规则基准历史tap数；主路径非负shared delay实际M=1。',
 'NORMALIZATION_MODE': '窗口内reference归一化：minmax、zscore或none。',
 'NUM_SEED_POINTS': 'TPE 启动前的随机探索 trial 数；不得超过总 trial 预算。',
 'N_JOBS': '并行 worker 数；1 最稳定，增大时注意内存和可复现性。',
 'OUTLIER_MAD_SCALE': 'MAD转标准差的一致性系数；参考正态分布1.4826。',
 'OUTLIER_MINIMUM_SAMPLES': 'PPG移动中位数检测的最小样本数。',
 'OUTLIER_THRESHOLD': 'PPG毛刺检测阈值：局部偏差超过该倍数scaled MAD才修复。',
 'PARALLEL_REPEATS': '重复实验并行数；当前实际并行度由 n_jobs 优先控制。',
 'POSTHOC_MIN_VALID_WINDOWS': 'oracle bias候选至少需要的有效比较窗口数，默认2。',
 'PPG_INPUT_TRANSFORM': 'PPG 输入变换；默认 log_absorbance，也可选择后端支持的既有变换。',
 'PREWARM_S': '自适应前置上下文秒数；默认0，与8秒观察窗独立。',
 'QC_POLICY': '坏窗口策略；fallback_baseline 表示保留计算并回退 baseline HR。',
 'REFERENCE_LAG_MS': '每窗reference correlation搜索半宽，毫秒。',
 'REPLAY_METRIC_ATOL': '保存指标与复算指标的绝对容差；有效窗口数仍须完全相同。',
 'REST_ALIGNMENT_SCORE_MODE': 'Tdelay 评分指标；aae 为当前默认，std 用于兼容旧实验。',
 'REST_HR_BAND_BPM': '静息 HR 合法范围(bpm)；用于对齐阶段的频带限制。',
 'REST_HR_BAND_HZ': '由 REST_HR_BAND_BPM 换算的 Hz 频带，供频谱接口使用。',
 'REST_HR_KWARGS': '汇总静息 HR 绘图参数；键值直接传给第 3/4/6 节绘图接口。',
 'REST_HR_PEAK_PERCENT': '候选频谱峰相对主峰的最低幅值比例。',
 'REST_HR_SLEW_LIMIT_BPM': '静息 HR 允许直接接受的相邻变化阈值(bpm)。',
 'REST_HR_SLEW_STEP_BPM': '超过静息变化阈值时每窗口最大修正步长(bpm)。',
 'REST_HR_SMOOTH_METHOD': '静息 HR 平滑方法；median 对孤立异常峰更稳健。',
 'REST_HR_SMOOTH_WIN': '静息 HR 平滑窗口数；应使用正奇数以保持中心对齐。',
 'REST_HR_SPEC_PENALTY_ENABLE': '是否在静息 HR 提取时启用运动频率惩罚。',
 'REST_HR_SPEC_PENALTY_WEIGHT': '静息HR提取的惩罚保留比例；保持基准值，独立于主路径0.4。',
 'REST_HR_SPEC_PENALTY_WIDTH_HZ': '静息运动峰惩罚半宽(Hz)；决定受抑制频带范围。',
 'REST_HR_TRACK_BAND_BPM': '只读冻结的诊断追踪带宽(bpm)，不参与Physical4D搜索。',
 'RFF_SEED': '随机特征固定种子42；不同坐标不派生新seed。',
 'RISE_LINEAGE_CONFIRM_WINDOWS': '连续跟踪候选达到该窗口数后才允许lineage reanchor。',
 'RISE_LINEAGE_HARMONIC_MULTIPLE': '接近motion峰此谐波的lineage被阻止reanchor。',
 'RISE_LINEAGE_MAX_AGE_WINDOWS': 'lineage的最大追踪窗口寿命。',
 'RISE_LINEAGE_MAX_DOWNWARD_DRIFT_BPM': 'lineage相邻窗允许的最大向下漂移bpm。',
 'RUN_ALIGNMENT_DIAGNOSTICS': '是否运行第 3 节并写静息 Tdelay 对齐图。',
 'RUN_ALL_TRAIN': '是否运行第 7 节正式 200-trial 模式级优化。',
 'RUN_BATCH_REFERENCE_COMPARE': '是否运行第 15 节批量参考 scheme 对比。',
 'RUN_BATCH_TARGET_HR_REDRAW': '是否运行第 16 节并批量重绘当前受试者全部目标段 HR 图。',
 'RUN_CONNECTIVITY_CHECK': '是否运行第 8 节小预算模式连通性检查。',
 'RUN_CROSS_MOTION_SUMMARY': '是否运行第 12 节跨运动汇总。',
 'RUN_CROSS_STAGE7_REPLAY': '是否运行第 13 节跨 scheme 回放。',
 'RUN_CROSS_WINDOW_DIAGNOSTICS': '是否运行第 14 节跨 scheme 窗口诊断。',
 'RUN_FULLFIELD_PLOT': '是否运行第 4 节并写全段未对齐 PPG-HR 图。',
 'RUN_OUTPUT_CHECK': '是否运行第 9 节只读检查 Stage-6 和断点文件。',
 'RUN_PREPROCESS_PREVIEW': '是否在第 2 节完整预处理一个所选样本。',
 'RUN_RAW_PPG_PLOT': '是否运行第 6 节并写原始 PPG 双轴图。',
 'RUN_STAGE7_REPLAY': '是否运行第 10 节单样本 best-params 回放。',
 'RUN_STATIC_QC': '是否在第 1 节读取各文件前 10 秒并显示静态 QC。',
 'RUN_WINDOW_DIAGNOSTICS': '是否运行第 11 节窗口波形/频谱诊断。',
 'SAVE_STAGE_JSON': '是否保存详细 stage JSON；调试有用但会增加磁盘文件。',
 'SEARCH_SPACE': 'Optuna 离散候选集合；下方逐项列表可直接增删候选值。',
 'SEARCH_SPACE.FS_TARGET': '训练重采样率候选(Hz)；较低值更快，较高值时间分辨率更高。',
 'SMOOTH_WIN_LEN': 'baseline/adaptive HR序列的移动中位数窗口数；主路径固定5。',
 'SPECTRAL_PENALTY_POLICY': 'soft_visible默认保持候选可见并保护连续性；完整policy增加有期限的owner可见性。',
 'SPEC_PENALTY_WEIGHT': '运动频谱中心保留比例；固定0.4，越小抑制越强，不参与Physical4D。',
 'TDELAY_RANGE_S': '样本唯一固定系统时延的候选范围，秒。',
 'TDELAY_STEP_S': '样本Tdelay搜索步长，秒。',
 'TEST_GROUPS_PER_TYPE': 'split 模式每个运动类型的测试组数；all_train 下不使用。',
 'TIME_BIAS_AFTER_MODE': 'post-hoc 对齐语义；oracle 只用于诊断指标，不参与部署选择。',
 'TIME_BIAS_AFTER_RANGE_S': 'post-hoc 时间偏移搜索区间(秒)，按(最小,最大)设置。',
 'TIME_BIAS_AFTER_STEP_S': 'post-hoc 时间偏移搜索步长(秒)。',
 'TRACKING_RANGE_DOWN_BPM': '旧共享向下搜索范围兼容覆盖；None 表示使用分阶段参考参数。',
 'TRACKING_RANGE_UP_BPM': '旧共享向上搜索范围兼容覆盖；None 表示使用分阶段参考参数。',
 'TRACKING_SLEW_LIMIT_DOWN_BPM': '旧共享下行 limit 兼容覆盖；None 表示使用分阶段参考参数。',
 'TRACKING_SLEW_LIMIT_UP_BPM': '旧共享上行 limit 兼容覆盖；None 表示使用分阶段参考参数。',
 'TRACKING_SLEW_STEP_DOWN_BPM': '旧共享下行 step 兼容覆盖；None 表示使用分阶段参考参数。',
 'TRACKING_SLEW_STEP_UP_BPM': '旧共享上行 step 兼容覆盖；None 表示使用分阶段参考参数。',
 'TRAIN_HR_POSTPROCESS_METHOD': '训练 HR 后处理方法；fft 为默认，ssr 启用稀疏频谱重建。',
 'TRIAL_CACHE_MAX_ENTRIES': 'trial 结果内存缓存上限；越大越占内存。',
 'TRIAL_PARAM_OVERRIDES': '固定 trial 参数映射；覆盖默认值并进入元数据、缓存键和断点指纹。',
 'UD_CALCULATION_MODE': 'UD 公式输入模式；raw 直接使用原始 Uh/Uc，smoothed 先分别平滑四路电压再计算。',
 'UD_SMOOTHING_WINDOW_S': '模式 smoothed 的中心移动中位数窗口(秒)；100 Hz 下 0.1 秒约为 10 点。',
 'VAL_GROUPS_PER_TYPE': 'split 模式每个运动类型的验证组数；all_train 下不使用。',
 'VISIBILITY_EDGE_GUARD_BPM': '候选排除频带的边缘保护宽度，默认1bpm。',
 'VISIBILITY_MATCH_HALF_WIDTH_BPM': 'owner与当前候选的匹配半宽，参考固定3.5bpm。',
 'VISIBILITY_MOTION_INVOCATIONS': 'owner取得后最多3次motion调用可越过排除；不按秒或所有窗口计数。'}
