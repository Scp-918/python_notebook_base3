"""Build the complete change6 single-subject experiment notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "run_batch_adaptive_protocol.ipynb"


def code(source: str) -> nbformat.NotebookNode:
    """Create one normalized code cell."""

    return nbformat.v4.new_code_cell(source.strip() + "\n")


def markdown(source: str) -> nbformat.NotebookNode:
    """Create one normalized Markdown cell."""

    return nbformat.v4.new_markdown_cell(source.strip() + "\n")


def section(number: int, title: str, description: str, source: str) -> list[nbformat.NotebookNode]:
    """Create one documented numbered section and its executable cell."""

    return [markdown(f"## {number}. {title}\n\n{description}"), code(source)]


def build_notebook() -> nbformat.NotebookNode:
    """Return the change6 notebook without writing to disk."""

    cells: list[nbformat.NotebookNode] = [
        markdown(
            """
# change6 单受试者自适应 PPG-HR 协议

本 Notebook 只接受一个 `SUBJECT_DIR`。文件名必须为
`{subject}_{motion}_{index}_sensor.csv` 与 `{subject}_{motion}_{index}_HRdata.csv`，
其中 motion 为 `write/gripper/run/rope`。标定固定读取 `SUBJECT_DIR.parent / "ck.mat"`。

训练、写图、回放和诊断默认全部关闭。请先完成第 1–2 节的只读检查，再按需打开对应
`RUN_*` 开关。所有结果和模式级断点写入
`SUBJECT_DIR.parent / "outputs" / "{subject}__{run_signature}"`。
"""
        ),
        markdown(
            """
## 0. 初始化、分层配置与安全开关

本节分成四个配置代码单元：核心路径与实验组合、预处理与对齐、滤波与增强追踪、
执行与输出。首个代码单元保留 tracker 总开关和六项独立开关。
"""
        ),
        code(
            r'''
# 用途：导入公共接口，设置单受试者路径、实验组合、优化预算和增强追踪开关。
# 输入：环境变量 PPG_SUBJECT_DIR；也可以在本单元临时修改 SUBJECT_DIR。
# 输出：仅定义变量，不创建输出目录。
# 是否写文件：否。
# 耗时风险：无；本单元不会读取 CSV、创建 Optuna study 或启动训练。
from pathlib import Path
import os
import sys

import pandas as pd
from IPython.display import Image, display

PROJECT_ROOT = Path.cwd().resolve()
if PROJECT_ROOT.name.lower() == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ppg_hr.params import CascadeScheme, MotionType, TargetScope
from ppg_hr.experimental.batch_pairing import discover_sample_pairs_with_unpaired
from ppg_hr.experimental.preprocess_protocol import load_and_preprocess_protocol
from ppg_hr.experimental.protocol_outputs import (
    plot_raw_ppg_and_unaligned_hr_by_motion_type,
    plot_rest_alignment_diagnostics_by_motion_type,
    plot_unaligned_fullfield_ppg_hr_by_motion_type,
)
from ppg_hr.experimental.protocol_search_space import default_protocol_search_space
from ppg_hr.experimental.qc import quality_filter_sample
from ppg_hr.experimental.run_batch_protocol import (
    build_cross_motion_summary_table,
    build_subject_output_dir,
    plot_window_diagnostics_from_records,
    replay_best_record_hr_curves,
    run_batch_adaptive_protocol,
    run_batch_reference_compare,
)
from ppg_hr.experimental.segmentation import detect_activity_segments
from ppg_hr.preprocess.calibration import load_subject_calibration

# 唯一数据输入。不要把本机绝对路径提交到仓库。
SUBJECT_DIR = Path(os.environ.get("PPG_SUBJECT_DIR", PROJECT_ROOT / "subject_data")).resolve()
SUBJECT_NAME = SUBJECT_DIR.name
OUTPUT_ROOT = SUBJECT_DIR.parent / "outputs"
MOTION_TYPES = tuple(item.value for item in MotionType)
assert MOTION_TYPES == ("write", "gripper", "run", "rope")

TARGET_SCOPES = [TargetScope.MOTION_POST10.value]
CASCADE_SCHEMES = [
    CascadeScheme.ACC.value,
    CascadeScheme.HF2.value,
    CascadeScheme.UD2.value,
    CascadeScheme.ACC_HF2.value,
    CascadeScheme.HF2_CF2.value,
    CascadeScheme.ACC_UD2.value,
]
ADAPTIVE_FILTERS = ["lms"]
OBJECTIVE_MODE = "posthoc_aae"
DATA_SPLIT_MODE = "all_train"
MAX_ITERATIONS = 200
NUM_REPEATS = 1
RANDOM_STATE = 42

# 总开关：legacy 完全使用旧追踪；enhanced 再由下面六个布尔开关逐项控制。
TRACKER_MODE = "enhanced"
ENABLE_DIRECTIONAL_TRACKING = True
ENABLE_DYNAMIC_PENALTY = True
ENABLE_CONTINUITY_PROTECTION = True
ENABLE_LOW_LOCK_RECOVERY = True
ENABLE_HIGH_LOCK_RECOVERY = True
ENABLE_POST_MOTION_PROTECTION = True

RUN_OUTPUT_DIR = build_subject_output_dir(
    subject_dir=SUBJECT_DIR,
    target_scopes=TARGET_SCOPES,
    cascade_schemes=CASCADE_SCHEMES,
    adaptive_filters=ADAPTIVE_FILTERS,
    objective_mode=OBJECTIVE_MODE,
    data_split_mode=DATA_SPLIT_MODE,
    tw_f_s=0.0,
    postprocess_method="fft",
)
RESULTS_ROOT = RUN_OUTPUT_DIR
print("SUBJECT_DIR:", SUBJECT_DIR)
print("RUN_OUTPUT_DIR:", RUN_OUTPUT_DIR)
'''
        ),
        code(
            r'''
# 用途：配置采样、预处理、静息 HR、Tdelay、post-hoc 对齐和恢复策略。
# 输入：这些值会进入预览、训练、缓存键、结果元数据和 replay 参数恢复。
# 输出：固定参数变量；不计算信号。
# 是否写文件：否。
# 耗时风险：无。
FS_ORIGIN = 100
FS_TARGET = 100
TW = 8
TW_F = 0.0
NORMALIZATION_MODE = "minmax"
QC_POLICY = "fallback_baseline"
DELAY_ESTIMATION_MODE = "envelope"  # 可选 envelope/direct
ALIGNMENT_TW = 8.0
ALIGNMENT_STEP_S = 1.0

REST_HR_BAND_BPM = (40.0, 180.0)
REST_HR_BAND_HZ = tuple(value / 60.0 for value in REST_HR_BAND_BPM)
REST_HR_TRACK_BAND_BPM = 30.0
REST_HR_SLEW_LIMIT_BPM = 6.0
REST_HR_SLEW_STEP_BPM = 4.0
REST_HR_SMOOTH_METHOD = "median"
REST_HR_SMOOTH_WIN = 3
REST_HR_PEAK_PERCENT = 0.3
REST_HR_SPEC_PENALTY_ENABLE = True
REST_HR_SPEC_PENALTY_WEIGHT = 0.2
REST_HR_SPEC_PENALTY_WIDTH_HZ = 0.2
REST_ALIGNMENT_SCORE_MODE = "aae"

ENABLE_TIME_BIAS_AFTER = True
TIME_BIAS_AFTER_RANGE_S = (-5.0, 5.0)
TIME_BIAS_AFTER_STEP_S = 1.0
TIME_BIAS_AFTER_MODE = "posthoc_oracle_alignment"
RECOVERY_GRACE_S = 10.0
RECOVERY_DIFF_BPM = 20.0
RECOVERY_CROSS_DIFF_BPM = 8.0

PPG_INPUT_TRANSFORM = "log_absorbance"
LOG_ABSORBANCE_BASELINE_MODE = "rolling_median"
LOG_ABSORBANCE_BASELINE_WINDOW_S = 5.0
LOG_ABSORBANCE_EPS = 1e-6
LOG_ABSORBANCE_RATIO_CLIP = (1e-3, 1e3)
GLOBAL_OBJECTIVE_STRATEGY = "current_global_adaptive"

REST_HR_KWARGS = {
    "hr_band_hz": REST_HR_BAND_HZ,
    "track_band_bpm": REST_HR_TRACK_BAND_BPM,
    "slew_limit_bpm": REST_HR_SLEW_LIMIT_BPM,
    "slew_step_bpm": REST_HR_SLEW_STEP_BPM,
    "smooth_method": REST_HR_SMOOTH_METHOD,
    "smooth_win": REST_HR_SMOOTH_WIN,
    "peak_percent": REST_HR_PEAK_PERCENT,
    "spec_penalty_enable": REST_HR_SPEC_PENALTY_ENABLE,
    "spec_penalty_weight": REST_HR_SPEC_PENALTY_WEIGHT,
    "spec_penalty_width_hz": REST_HR_SPEC_PENALTY_WIDTH_HZ,
}
'''
        ),
        code(
            r'''
# 用途：配置 HR 后处理、四种自适应滤波器、级联保护和增强追踪全部固定阈值。
# 输入：固定值覆盖 Optuna 解码后的同名字段；搜索参数范围由 SEARCH_SPACE 控制。
# 输出：滤波与追踪配置变量。
# 是否写文件：否。
# 耗时风险：无；启用 SSR、Volterra、RFF-LMS 或 KLMS 会增加后续训练耗时。
TRAIN_HR_POSTPROCESS_METHOD = "fft"  # 可选 fft/ssr
REDRAW_HR_POSTPROCESS_METHOD = None   # None 表示按训练记录恢复
SSR_NUM_ATOMS = 5
SSR_LAMBDA = 0.15
SSR_HARMONIC_TOL_BPM = 5.0
SSR_FALLBACK_TO_FFT = True
SSR_GRID_RESOLUTION_BPM = 1.0

LMS_MU_BASE = 0.01
LMS_MU_MIN = 1e-6
VOLTERRA_ALPHA_U = 0.1
VOLTERRA_M2 = 3
RFF_D = 100
RFF_SIGMA_SCALE = 1.0
RFF_SIGMA = 1.0  # 仅用于兼容缺少 sigma_scale 的旧记录
RFF_UPDATE_MODE = "nlms"
RFF_NLMS_EPS = 1e-9
RFF_LEAKAGE = 0.0
RFF_ERR_CLIP = None
RFF_THETA_NORM_GUARD = None
KLMS_STEP_SIZE = 0.05
KLMS_SIGMA = 1.0
KLMS_EPSILON = 0.1
KLMS_MAX_DICTIONARY_SIZE = 300
KLMS_CENTER_PRUNE_POLICY = "freeze_new_centers"
KLMS_DISTANCE_MODE = "normalized"
KLMS_NORMALIZED_UPDATE = True
KLMS_NLMS_EPS = 1e-6

CASCADE_GUARD_POLICY = "none"
CASCADE_GUARD_RATIO_MIN = 0.05
CASCADE_GUARD_RATIO_MAX = 5.0
CASCADE_GUARD_FLAT_STD_EPS = 1e-6
CASCADE_GUARD_USE_FINITE_ZSCORE = True

TRACKING_RANGE_UP_BPM = 25.0
TRACKING_RANGE_DOWN_BPM = 25.0
TRACKING_SLEW_LIMIT_UP_BPM = 10.0
TRACKING_SLEW_STEP_UP_BPM = 7.0
TRACKING_SLEW_LIMIT_DOWN_BPM = 10.0
TRACKING_SLEW_STEP_DOWN_BPM = 7.0
LOW_LOCK_MIN_BPM = 50.0
LOW_LOCK_MAX_BPM = 80.0
LOW_LOCK_MIN_WINDOWS = 4
LOW_LOCK_TARGET_MIN_BPM = 90.0
LOW_LOCK_MIN_JUMP_BPM = 20.0
LOW_LOCK_MIN_AMP_RATIO = 0.45
LOW_LOCK_CANDIDATE_STABLE_BPM = 10.0
LOW_LOCK_CONFIRM_WINDOWS = 3
LOW_LOCK_STEP_BPM = 30.0
HIGH_LOCK_CONFIRM_WINDOWS = 3
HIGH_LOCK_COOLDOWN_WINDOWS = 4
HIGH_LOCK_MIN_GAP_BPM = 20.0
HIGH_LOCK_MIN_AMP_RATIO = 0.45
HIGH_LOCK_CANDIDATE_MIN_BPM = 85.0
HIGH_LOCK_CANDIDATE_STABLE_BPM = 10.0
HIGH_LOCK_PENALTY_EXCLUSION_BPM = 10.0
HIGH_LOCK_DOWN_STEP_BPM = 20.0
HIGH_LOCK_UP_STEP_BPM = 3.0
POST_MOTION_GUARD_SECONDS = 10.0
POST_MOTION_GUARD_MIN_ELAPSED_S = 5.0
POST_MOTION_GUARD_STABLE_WINDOWS = 3
POST_MOTION_GUARD_CROSSOVER_GAP_BPM = 2.0
POST_MOTION_GUARD_UPWARD_GAP_BPM = 1.5
POST_MOTION_GUARD_FFT_FLOOR_BPM = 55.0
POST_MOTION_GUARD_RECOVERY_STEP_UP_BPM = 1.5
POST_MOTION_GUARD_RECOVERY_STEP_DOWN_BPM = 3.0
POST_MOTION_GUARD_RESCUE_GAP_BPM = 20.0
POST_MOTION_GUARD_GAP_RESCUE_ENABLE = True
POST_MOTION_GUARD_GAP_RESCUE_WINDOWS = 4
POST_MOTION_GUARD_GAP_RESCUE_MIN_HITS = 3
POST_MOTION_GUARD_FFT_STABLE_WINDOWS = 3
POST_MOTION_GUARD_FFT_STABLE_BPM = 6.0
'''
        ),
        code(
            r'''
# 用途：汇总固定 trial 参数、搜索空间、并行/缓存策略和各章节安全开关。
# 输入：前面三个配置单元；修改任一固定参数会进入结果元数据和 checkpoint 指纹。
# 输出：SEARCH_SPACE、TRIAL_PARAM_OVERRIDES 和运行控制变量。
# 是否写文件：否。
# 耗时风险：无；所有昂贵或写文件的 RUN_* 开关默认 False。
SEARCH_SPACE = default_protocol_search_space()
NUM_SEED_POINTS = 10
N_JOBS = 1
PARALLEL_REPEATS = 1
TRIAL_CACHE_MAX_ENTRIES = 128
SAVE_STAGE_JSON = False
DEBUG_MODE = False
CLEAN_OUTPUTS = False
VAL_GROUPS_PER_TYPE = 1
TEST_GROUPS_PER_TYPE = 1
CASCADE_TRAIN_BUDGETS = None

TRIAL_PARAM_OVERRIDES = {
    "Fs_Target": FS_TARGET,
    "TW_F": TW_F,
    "normalization_mode": NORMALIZATION_MODE,
    "postprocess_method": TRAIN_HR_POSTPROCESS_METHOD,
    "SSR_Num_Atoms": SSR_NUM_ATOMS,
    "SSR_Lambda": SSR_LAMBDA,
    "SSR_Harmonic_Tol_BPM": SSR_HARMONIC_TOL_BPM,
    "SSR_Fallback_To_FFT": SSR_FALLBACK_TO_FFT,
    "SSR_Grid_Resolution_BPM": SSR_GRID_RESOLUTION_BPM,
    "LMS_Mu_Base": LMS_MU_BASE,
    "LMS_Mu_Min": LMS_MU_MIN,
    "qc_policy": QC_POLICY,
    "delay_estimation_mode": DELAY_ESTIMATION_MODE,
    "Alignment_TW": ALIGNMENT_TW,
    "Alignment_Step": ALIGNMENT_STEP_S,
    "Rest_HR_Band_BPM": REST_HR_BAND_BPM,
    "Rest_HR_Track_Band_BPM": REST_HR_TRACK_BAND_BPM,
    "Rest_HR_Slew_Limit_BPM": REST_HR_SLEW_LIMIT_BPM,
    "Rest_HR_Slew_Step_BPM": REST_HR_SLEW_STEP_BPM,
    "Rest_HR_Smooth_Method": REST_HR_SMOOTH_METHOD,
    "Rest_HR_Smooth_Win": REST_HR_SMOOTH_WIN,
    "Rest_HR_Peak_Percent": REST_HR_PEAK_PERCENT,
    "Rest_HR_Spec_Penalty_Enable": REST_HR_SPEC_PENALTY_ENABLE,
    "Rest_HR_Spec_Penalty_Weight": REST_HR_SPEC_PENALTY_WEIGHT,
    "Rest_HR_Spec_Penalty_Width_Hz": REST_HR_SPEC_PENALTY_WIDTH_HZ,
    "Rest_Alignment_Score_Mode": REST_ALIGNMENT_SCORE_MODE,
    "Enable_Time_Bias_After": ENABLE_TIME_BIAS_AFTER,
    "Time_Bias_After_Range_S": TIME_BIAS_AFTER_RANGE_S,
    "Time_Bias_After_Step_S": TIME_BIAS_AFTER_STEP_S,
    "Time_Bias_After_Mode": TIME_BIAS_AFTER_MODE,
    "Recovery_Grace_S": RECOVERY_GRACE_S,
    "Recovery_Diff_Bpm": RECOVERY_DIFF_BPM,
    "Recovery_Cross_Diff_Bpm": RECOVERY_CROSS_DIFF_BPM,
    "alpha_u": VOLTERRA_ALPHA_U,
    "M2": VOLTERRA_M2,
    "rff_D": RFF_D,
    "rff_sigma_scale": RFF_SIGMA_SCALE,
    "rff_sigma": RFF_SIGMA,
    "rff_update_mode": RFF_UPDATE_MODE,
    "rff_nlms_eps": RFF_NLMS_EPS,
    "rff_leakage": RFF_LEAKAGE,
    "rff_err_clip": RFF_ERR_CLIP,
    "rff_theta_norm_guard": RFF_THETA_NORM_GUARD,
    "klms_step_size": KLMS_STEP_SIZE,
    "klms_sigma": KLMS_SIGMA,
    "klms_epsilon": KLMS_EPSILON,
    "klms_max_dictionary_size": KLMS_MAX_DICTIONARY_SIZE,
    "klms_center_prune_policy": KLMS_CENTER_PRUNE_POLICY,
    "klms_distance_mode": KLMS_DISTANCE_MODE,
    "klms_normalized_update": KLMS_NORMALIZED_UPDATE,
    "klms_nlms_eps": KLMS_NLMS_EPS,
    "ppg_input_transform": PPG_INPUT_TRANSFORM,
    "log_absorbance_baseline_mode": LOG_ABSORBANCE_BASELINE_MODE,
    "log_absorbance_baseline_window_s": LOG_ABSORBANCE_BASELINE_WINDOW_S,
    "log_absorbance_eps": LOG_ABSORBANCE_EPS,
    "log_absorbance_ratio_clip": LOG_ABSORBANCE_RATIO_CLIP,
    "global_objective_strategy": GLOBAL_OBJECTIVE_STRATEGY,
    "cascade_guard_policy": CASCADE_GUARD_POLICY,
    "cascade_guard_ratio_min": CASCADE_GUARD_RATIO_MIN,
    "cascade_guard_ratio_max": CASCADE_GUARD_RATIO_MAX,
    "cascade_guard_flat_std_eps": CASCADE_GUARD_FLAT_STD_EPS,
    "cascade_guard_use_finite_zscore": CASCADE_GUARD_USE_FINITE_ZSCORE,
    "tracker_mode": TRACKER_MODE,
    "enable_directional_tracking": ENABLE_DIRECTIONAL_TRACKING,
    "enable_dynamic_penalty": ENABLE_DYNAMIC_PENALTY,
    "enable_continuity_protection": ENABLE_CONTINUITY_PROTECTION,
    "enable_low_lock_recovery": ENABLE_LOW_LOCK_RECOVERY,
    "enable_high_lock_recovery": ENABLE_HIGH_LOCK_RECOVERY,
    "enable_post_motion_protection": ENABLE_POST_MOTION_PROTECTION,
    "tracking_range_up_bpm": TRACKING_RANGE_UP_BPM,
    "tracking_range_down_bpm": TRACKING_RANGE_DOWN_BPM,
    "tracking_slew_limit_up_bpm": TRACKING_SLEW_LIMIT_UP_BPM,
    "tracking_slew_step_up_bpm": TRACKING_SLEW_STEP_UP_BPM,
    "tracking_slew_limit_down_bpm": TRACKING_SLEW_LIMIT_DOWN_BPM,
    "tracking_slew_step_down_bpm": TRACKING_SLEW_STEP_DOWN_BPM,
    "low_lock_min_bpm": LOW_LOCK_MIN_BPM,
    "low_lock_max_bpm": LOW_LOCK_MAX_BPM,
    "low_lock_min_windows": LOW_LOCK_MIN_WINDOWS,
    "low_lock_target_min_bpm": LOW_LOCK_TARGET_MIN_BPM,
    "low_lock_min_jump_bpm": LOW_LOCK_MIN_JUMP_BPM,
    "low_lock_min_amp_ratio": LOW_LOCK_MIN_AMP_RATIO,
    "low_lock_candidate_stable_bpm": LOW_LOCK_CANDIDATE_STABLE_BPM,
    "low_lock_confirm_windows": LOW_LOCK_CONFIRM_WINDOWS,
    "low_lock_step_bpm": LOW_LOCK_STEP_BPM,
    "high_lock_confirm_windows": HIGH_LOCK_CONFIRM_WINDOWS,
    "high_lock_cooldown_windows": HIGH_LOCK_COOLDOWN_WINDOWS,
    "high_lock_min_gap_bpm": HIGH_LOCK_MIN_GAP_BPM,
    "high_lock_min_amp_ratio": HIGH_LOCK_MIN_AMP_RATIO,
    "high_lock_candidate_min_bpm": HIGH_LOCK_CANDIDATE_MIN_BPM,
    "high_lock_candidate_stable_bpm": HIGH_LOCK_CANDIDATE_STABLE_BPM,
    "high_lock_penalty_exclusion_bpm": HIGH_LOCK_PENALTY_EXCLUSION_BPM,
    "high_lock_down_step_bpm": HIGH_LOCK_DOWN_STEP_BPM,
    "high_lock_up_step_bpm": HIGH_LOCK_UP_STEP_BPM,
    "post_motion_guard_seconds": POST_MOTION_GUARD_SECONDS,
    "post_motion_guard_min_elapsed_s": POST_MOTION_GUARD_MIN_ELAPSED_S,
    "post_motion_guard_stable_windows": POST_MOTION_GUARD_STABLE_WINDOWS,
    "post_motion_guard_crossover_gap_bpm": POST_MOTION_GUARD_CROSSOVER_GAP_BPM,
    "post_motion_guard_upward_gap_bpm": POST_MOTION_GUARD_UPWARD_GAP_BPM,
    "post_motion_guard_fft_floor_bpm": POST_MOTION_GUARD_FFT_FLOOR_BPM,
    "post_motion_guard_recovery_step_up_bpm": POST_MOTION_GUARD_RECOVERY_STEP_UP_BPM,
    "post_motion_guard_recovery_step_down_bpm": POST_MOTION_GUARD_RECOVERY_STEP_DOWN_BPM,
    "post_motion_guard_rescue_gap_bpm": POST_MOTION_GUARD_RESCUE_GAP_BPM,
    "post_motion_guard_gap_rescue_enable": POST_MOTION_GUARD_GAP_RESCUE_ENABLE,
    "post_motion_guard_gap_rescue_windows": POST_MOTION_GUARD_GAP_RESCUE_WINDOWS,
    "post_motion_guard_gap_rescue_min_hits": POST_MOTION_GUARD_GAP_RESCUE_MIN_HITS,
    "post_motion_guard_fft_stable_windows": POST_MOTION_GUARD_FFT_STABLE_WINDOWS,
    "post_motion_guard_fft_stable_bpm": POST_MOTION_GUARD_FFT_STABLE_BPM,
}

RUN_STATIC_QC = True
RUN_PREPROCESS_PREVIEW = True
RUN_ALIGNMENT_DIAGNOSTICS = False
RUN_FULLFIELD_PLOT = False
RUN_RAW_PPG_PLOT = False
RUN_ALL_TRAIN = False
RUN_CONNECTIVITY_CHECK = False
RUN_OUTPUT_CHECK = False
CONNECTIVITY_MAX_ITERATIONS = 1
'''
        ),
    ]

    cells += section(
        1,
        "严格配对、标定和静态 QC",
        "只读取当前受试者目录和上一级 `ck.mat`。QC 坏样本仍会保留并进入正式流程。",
        r'''
# 用途：发现完整配对、解析 MATLAB 标定，并汇总每个样本的只读 QC。
# 输入：SUBJECT_DIR、FS_ORIGIN。
# 输出：discovery、calibration、qc_preview；仅显示，不落盘。
# 是否写文件：否。
# 耗时风险：低；每个传感器文件只读取 QC 所需的前 10 秒。
discovery = None
calibration = None
qc_preview = pd.DataFrame()
datasets = {}
if not SUBJECT_DIR.is_dir():
    print("请设置 PPG_SUBJECT_DIR 或修改 SUBJECT_DIR：", SUBJECT_DIR)
else:
    calibration_messages = []
    calibration = load_subject_calibration(SUBJECT_DIR, on_log=calibration_messages.append)
    discovery = discover_sample_pairs_with_unpaired(SUBJECT_DIR)
    print(*calibration_messages, sep="\n")
    print("完整配对：", len(discovery.pairs))
    print("运动计数：", {
        motion: sum(pair.motion_type == motion for pair in discovery.pairs)
        for motion in MOTION_TYPES
    })
    if discovery.unpaired:
        display(pd.DataFrame([issue.__dict__ for issue in discovery.unpaired]))
    if RUN_STATIC_QC:
        qc_results = [
            quality_filter_sample(
                pair.sensor_csv,
                fs=FS_ORIGIN,
                group_id=pair.motion_id,
                motion_type=pair.motion_type,
                ref_csv=pair.ref_csv,
            )
            for pair in discovery.pairs
        ]
        qc_preview = pd.DataFrame([item.to_dict() for item in qc_results])
        display(qc_preview)
''',
    )
    cells += section(
        2,
        "单样本预处理、分段和对齐预览",
        "通过 motion/index 选择一个样本，检查 CF2、HF2、HF2comp、UD2、ACC、QC 和运动分段。",
        r'''
# 用途：只读预处理选定样本，并显示新协议通道、标定元数据和分段摘要。
# 输入：PREVIEW_MOTION_TYPE/PREVIEW_MOTION_INDEX，以及第 1 节的 discovery/calibration。
# 输出：preview_pair、preview_dataset、preview_segment；不保存图表。
# 是否写文件：否。
# 耗时风险：中；会完整读取并预处理一个传感器/HR 配对。
PREVIEW_MOTION_TYPE = "write"
PREVIEW_MOTION_INDEX = 1
preview_pair = None
preview_dataset = None
preview_segment = None
if RUN_PREPROCESS_PREVIEW and discovery is not None:
    matches = [
        pair for pair in discovery.pairs
        if pair.motion_type == PREVIEW_MOTION_TYPE and pair.motion_index == PREVIEW_MOTION_INDEX
    ]
    if len(matches) != 1:
        print("预览样本必须唯一，当前匹配数：", len(matches))
    else:
        preview_pair = matches[0]
        preview_dataset = load_and_preprocess_protocol(
            preview_pair.sensor_csv,
            preview_pair.ref_csv,
            fs_origin=FS_ORIGIN,
            calibration=calibration,
        )
        datasets[preview_pair.motion_id] = preview_dataset
        preview_segment = detect_activity_segments(
            preview_dataset.accx,
            preview_dataset.accy,
            preview_dataset.accz,
            preview_dataset.fs,
            TW=TW,
        )
        display(preview_dataset.to_frame().head())
        print("样本：", preview_pair.sample_id)
        print("标定/序号元数据：", preview_dataset.source_metadata)
        print("分段：", preview_segment)
else:
    print("RUN_PREPROCESS_PREVIEW=False 或尚未发现配对，跳过预览。")
''',
    )
    cells += section(
        3,
        "输出静息段全局 Tdelay 对齐诊断图",
        "按运动类型生成静息段参考 HR 与 PPG-HR 对齐图；该诊断不改变训练参数。",
        r'''
# 用途：为所有已预处理样本绘制静息段 Tdelay 对齐诊断。
# 输入：discovery、datasets 和 REST_HR_KWARGS。
# 输出：RUN_OUTPUT_DIR/alignment_diagnostics 下的 PNG。
# 是否写文件：是，仅在 RUN_ALIGNMENT_DIAGNOSTICS=True 时。
# 耗时风险：中；若 datasets 未缓存，会预处理全部配对。
alignment_paths = {}
if RUN_ALIGNMENT_DIAGNOSTICS and discovery is not None:
    for pair in discovery.pairs:
        if pair.motion_id not in datasets:
            datasets[pair.motion_id] = load_and_preprocess_protocol(
                pair.sensor_csv, pair.ref_csv, fs_origin=FS_ORIGIN, calibration=calibration
            )
    alignment_paths = plot_rest_alignment_diagnostics_by_motion_type(
        discovery.pairs,
        datasets,
        RUN_OUTPUT_DIR / "alignment_diagnostics",
        fs_target=FS_TARGET,
        TW=TW,
        alignment_TW=ALIGNMENT_TW,
        alignment_step_s=ALIGNMENT_STEP_S,
        rest_hr_kwargs=REST_HR_KWARGS,
        alignment_score_mode=REST_ALIGNMENT_SCORE_MODE,
    )
    print(alignment_paths)
else:
    print("RUN_ALIGNMENT_DIAGNOSTICS=False，跳过写图。")
''',
    )
    cells += section(
        4,
        "输出全段未对齐 PPG-HR 测试图",
        "按 canonical 运动顺序绘制全段 PPG-HR 与参考 HR，用于训练前人工检查。",
        r'''
# 用途：绘制所有运动类型的全段、未对齐 PPG-HR 检查图。
# 输入：discovery、datasets 和静息 HR 追踪参数。
# 输出：RUN_OUTPUT_DIR/fullfield_ppg_hr 下的 PNG。
# 是否写文件：是，仅在 RUN_FULLFIELD_PLOT=True 时。
# 耗时风险：中；可能预处理全部配对并执行窗口 FFT。
fullfield_paths = {}
if RUN_FULLFIELD_PLOT and discovery is not None:
    for pair in discovery.pairs:
        if pair.motion_id not in datasets:
            datasets[pair.motion_id] = load_and_preprocess_protocol(
                pair.sensor_csv, pair.ref_csv, fs_origin=FS_ORIGIN, calibration=calibration
            )
    fullfield_paths = plot_unaligned_fullfield_ppg_hr_by_motion_type(
        discovery.pairs,
        datasets,
        RUN_OUTPUT_DIR / "fullfield_ppg_hr",
        fs_target=FS_TARGET,
        TW=TW,
        step_s=ALIGNMENT_STEP_S,
        **REST_HR_KWARGS,
    )
    print(fullfield_paths)
else:
    print("RUN_FULLFIELD_PLOT=False，跳过写图。")
''',
    )
    cells += section(
        5,
        "封装训练函数",
        "所有训练入口统一调用该函数，确保输出目录、固定参数、搜索空间和断点配置一致。",
        r'''
# 用途：定义统一训练包装器；第 7、8 节只选择预算和组合。
# 输入：第 0 节全部配置及 SUBJECT_DIR。
# 输出：调用时返回 BatchProtocolResult。
# 是否写文件：定义函数时不写；实际调用会写入 RUN_OUTPUT_DIR。
# 耗时风险：定义无风险；调用会运行 Optuna，请只在明确打开训练开关时执行。
def run_training_cell(*, max_iterations, target_scopes, cascade_schemes, adaptive_filters):
    return run_batch_adaptive_protocol(
        subject_dir=SUBJECT_DIR,
        output_root=RUN_OUTPUT_DIR,
        max_iterations=max_iterations,
        num_repeats=NUM_REPEATS,
        random_state=RANDOM_STATE,
        num_seed_points=NUM_SEED_POINTS,
        fs_origin=FS_ORIGIN,
        parallel_repeats=PARALLEL_REPEATS,
        n_jobs=N_JOBS,
        trial_cache_max_entries=TRIAL_CACHE_MAX_ENTRIES,
        save_stage_json=SAVE_STAGE_JSON,
        debug_mode=DEBUG_MODE,
        search_space=SEARCH_SPACE,
        trial_param_overrides=TRIAL_PARAM_OVERRIDES,
        target_scopes=target_scopes,
        cascade_schemes=cascade_schemes,
        adaptive_filters=adaptive_filters,
        objective_mode=OBJECTIVE_MODE,
        data_split_mode=DATA_SPLIT_MODE,
        delay_estimation_mode=DELAY_ESTIMATION_MODE,
        val_groups_per_type=VAL_GROUPS_PER_TYPE,
        test_groups_per_type=TEST_GROUPS_PER_TYPE,
        cascade_train_budgets=CASCADE_TRAIN_BUDGETS,
        project_root=SUBJECT_DIR.parent,
        clean_outputs=CLEAN_OUTPUTS,
    )
''',
    )
    cells += section(
        6,
        "输出静息段原始 PPG 与 PPG-HR 双 y 轴图",
        "检查原始 PPG 波形与窗口 HR 的时间对应关系，不参与训练目标。",
        r'''
# 用途：绘制各运动类型静息段的原始 PPG 与未对齐 HR 双轴图。
# 输入：discovery、datasets、FS_ORIGIN 和 REST_HR_KWARGS。
# 输出：RUN_OUTPUT_DIR/raw_ppg_dual_axis 下的 PNG。
# 是否写文件：是，仅在 RUN_RAW_PPG_PLOT=True 时。
# 耗时风险：中；可能预处理全部配对并执行窗口 FFT。
raw_ppg_paths = {}
if RUN_RAW_PPG_PLOT and discovery is not None:
    for pair in discovery.pairs:
        if pair.motion_id not in datasets:
            datasets[pair.motion_id] = load_and_preprocess_protocol(
                pair.sensor_csv, pair.ref_csv, fs_origin=FS_ORIGIN, calibration=calibration
            )
    raw_ppg_paths = plot_raw_ppg_and_unaligned_hr_by_motion_type(
        discovery.pairs,
        datasets,
        RUN_OUTPUT_DIR / "raw_ppg_dual_axis",
        fs_target=FS_TARGET,
        fs_origin=FS_ORIGIN,
        TW=TW,
        step_s=ALIGNMENT_STEP_S,
        **REST_HR_KWARGS,
    )
    print(raw_ppg_paths)
else:
    print("RUN_RAW_PPG_PLOT=False，跳过写图。")
''',
    )
    cells += section(
        7,
        "all_train 正式模式级优化",
        "同一运动类型全部 index 共同形成一次 objective；不同运动分别保存参数和断点。",
        r'''
# 用途：运行 change6 默认正式训练：all_train + motion_post10 + posthoc_aae。
# 输入：六种 scheme、默认 LMS、每模式 200 trials/1 repeat。
# 输出：RUN_OUTPUT_DIR 下的 QC、图表、Stage-6 记录、history 和模式 checkpoint。
# 是否写文件：是，仅在 RUN_ALL_TRAIN=True 时。
# 耗时风险：高；真实数据会对每个运动/模式执行完整 Optuna 优化。
result_all_train = None
if RUN_ALL_TRAIN:
    result_all_train = run_training_cell(
        max_iterations=MAX_ITERATIONS,
        target_scopes=TARGET_SCOPES,
        cascade_schemes=CASCADE_SCHEMES,
        adaptive_filters=ADAPTIVE_FILTERS,
    )
    RESULTS_ROOT = result_all_train.output_root
    print("训练结果：", RESULTS_ROOT)
else:
    print("RUN_ALL_TRAIN=False，未创建 Optuna study。")
''',
    )
    cells += section(
        8,
        "当前配置的模式连通性检查",
        "模式数量按已发现运动数 × scope × scheme × filter 动态计算，不再使用旧的 84 模式。",
        r'''
# 用途：用极小 trial 数检查当前 motion/scope/scheme/filter 组合能否完整连通。
# 输入：CONNECTIVITY_MAX_ITERATIONS 和当前实验组合。
# 输出：与正式训练共用模式级目录和配置指纹；建议使用独立 scheme 子集测试。
# 是否写文件：是，仅在 RUN_CONNECTIVITY_CHECK=True 时。
# 耗时风险：中到高；即使每模式 1 trial，也会处理所有已选模式和全部数据。
motion_count = len({pair.motion_type for pair in discovery.pairs}) if discovery is not None else 0
configured_mode_count = motion_count * len(TARGET_SCOPES) * len(CASCADE_SCHEMES) * len(ADAPTIVE_FILTERS)
print("当前配置模式数：", configured_mode_count)
result_connectivity = None
if RUN_CONNECTIVITY_CHECK:
    result_connectivity = run_training_cell(
        max_iterations=CONNECTIVITY_MAX_ITERATIONS,
        target_scopes=TARGET_SCOPES,
        cascade_schemes=CASCADE_SCHEMES,
        adaptive_filters=ADAPTIVE_FILTERS,
    )
    RESULTS_ROOT = result_connectivity.output_root
else:
    print("RUN_CONNECTIVITY_CHECK=False，跳过连通性检查。")
''',
    )
    cells += section(
        9,
        "模式级输出与断点检查",
        "只读取已有结果，核对四类运动目录、checkpoint、history、Stage-6 CSV/JSON 和失败记录。",
        r'''
# 用途：检查 RESULTS_ROOT 下是否具备可恢复、可回放的完整模式级记录。
# 输入：RESULTS_ROOT；默认等于确定性的 RUN_OUTPUT_DIR。
# 输出：output_inventory DataFrame，仅显示文件状态。
# 是否写文件：否。
# 耗时风险：低；不会重新训练或改写 checkpoint。
output_inventory = pd.DataFrame()
if RUN_OUTPUT_CHECK:
    rows = []
    for motion in MOTION_TYPES:
        motion_dir = Path(RESULTS_ROOT) / "motion_types" / motion
        required = [
            "_checkpoint.json",
            "best_params_and_alignment.csv",
            "best_metrics.csv",
            "motion_frequency_and_params.csv",
            "full_report.json",
        ]
        rows.append({
            "motion_type": motion,
            "motion_dir": str(motion_dir),
            **{name: (motion_dir / name).exists() for name in required},
            "history_count": len(list(motion_dir.glob("modes/*/history.csv"))) if motion_dir.exists() else 0,
        })
    output_inventory = pd.DataFrame(rows)
    display(output_inventory)
else:
    print("RUN_OUTPUT_CHECK=False，跳过结果目录检查。")
''',
    )

    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {
        "display_name": "PPG_sensor_env",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["language_info"] = {"name": "python", "version": "3.10"}
    return notebook


def main() -> None:
    """Write the generated notebook to its tracked location."""

    nbformat.write(build_notebook(), NOTEBOOK)


if __name__ == "__main__":
    main()
