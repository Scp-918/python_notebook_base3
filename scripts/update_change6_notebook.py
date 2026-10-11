"""Build the complete change6 single-subject experiment notebook."""

from __future__ import annotations

import re
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "run_batch_adaptive_protocol.ipynb"

CONFIG_LINE_RE = re.compile(
    r'^\s*(?:[A-Z][A-Z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?\s*=|"[A-Za-z_][A-Za-z0-9_]*"\s*:)'
)

CONFIG_COMMENTS = {'FS_ORIGIN': '原始采样率(Hz)；change6/Pydisplay 数据固定按 100 Hz 与序号校验。',
 'UD_CALCULATION_MODE': 'UD 公式输入模式；raw 直接使用原始 Uh/Uc，smoothed 先分别平滑四路电压再计算。',
 'UD_SMOOTHING_WINDOW_S': '模式 smoothed 的中心移动中位数窗口(秒)；100 Hz 下 0.1 秒约为 10 点。',
 'FS_TARGET': '预览绘图目标采样率(Hz)；训练候选由 SEARCH_SPACE.Fs_Target 单独控制。',
 'hr_window_s': '预览分段窗口长度(秒)；HR observation固定8秒，step固定1秒，状态以center判断。',
 'prewarm_s': '自适应滤波前置收敛上下文(秒)；0 表示不额外增加前置上下文。',
 'NORMALIZATION_MODE': '参考通道归一化方式；minmax 为当前默认和推荐配置。',
 'QC_POLICY': '坏窗口策略；fallback_baseline 表示保留计算并回退 baseline HR。',
 'DELAY_ESTIMATION_MODE': '参考通道时延估计方式；envelope 为默认，direct 保留直接相关模式。',
 'EVALUATION_WINDOW_S': '评价诊断窗8秒，冻结；不控制预测网格。',
 'EVALUATION_STEP_S': '评价诊断步长1秒，冻结。',
 'REST_HR_BAND_BPM': '静息 HR 合法范围(bpm)；用于对齐阶段的频带限制。',
 'REST_HR_BAND_HZ': '由 REST_HR_BAND_BPM 换算的 Hz 频带，供频谱接口使用。',
 'REST_HR_TRACK_BAND_BPM': '只读冻结的诊断追踪带宽(bpm)，不参与BO。',
 'REST_HR_SLEW_LIMIT_BPM': '静息 HR 允许直接接受的相邻变化阈值(bpm)。',
 'REST_HR_SLEW_STEP_BPM': '超过静息变化阈值时每窗口最大修正步长(bpm)。',
 'REST_HR_SMOOTH_METHOD': '静息 HR 平滑方法；median 对孤立异常峰更稳健。',
 'REST_HR_SMOOTH_WIN': '静息 HR 平滑窗口数；应使用正奇数以保持中心对齐。',
 'REST_HR_PEAK_PERCENT': '候选频谱峰相对主峰的最低幅值比例。',
 'REST_HR_SPEC_PENALTY_ENABLE': '是否在静息 HR 提取时启用运动频率惩罚。',
 'REST_HR_SPEC_PENALTY_WEIGHT': '保留比例；越小抑制越强，核心固定0.2。',
 'REST_HR_SPEC_PENALTY_WIDTH_HZ': '静息运动峰惩罚半宽(Hz)；决定受抑制频带范围。',
 'REST_ALIGNMENT_SCORE_MODE': 'Tdelay 评分指标；aae 为当前默认，std 用于兼容旧实验。',
 'ENABLE_TIME_BIAS_AFTER': '是否计算滤波后 post-hoc 时间偏移诊断。',
 'TIME_BIAS_AFTER_RANGE_S': 'post-hoc 时间偏移搜索区间(秒)，按(最小,最大)设置。',
 'TIME_BIAS_AFTER_STEP_S': 'post-hoc 时间偏移搜索步长(秒)。',
 'TIME_BIAS_AFTER_MODE': 'post-hoc 对齐语义；oracle 只用于诊断指标，不参与部署选择。',
 'PPG_INPUT_TRANSFORM': 'PPG 输入变换；默认 log_absorbance，也可选择后端支持的既有变换。',
 'LOG_ABSORBANCE_BASELINE_MODE': 'log-absorbance 基线估计方法；rolling_median 抑制慢漂移。',
 'LOG_ABSORBANCE_BASELINE_WINDOW_S': '滚动基线窗口长度(秒)；越大越保留低频趋势。',
 'LOG_ABSORBANCE_EPS': 'log 比值的数值稳定下限，防止零值导致无穷大。',
 'LOG_ABSORBANCE_RATIO_CLIP': '进入 log 前的比值裁剪范围，防止极端异常值。',
 'REST_HR_KWARGS': '汇总静息 HR 绘图参数；键值直接传给第 3/4/6 节绘图接口。',
 'TRAIN_HR_POSTPROCESS_METHOD': '训练 HR 后处理方法；fft 为默认，ssr 启用稀疏频谱重建。',
 'SPEC_PENALTY_WEIGHT': '通用运动频谱惩罚权重；越大越排斥参考通道峰。',
 'TRACKING_RANGE_UP_BPM': '旧共享向上搜索范围兼容覆盖；None 表示使用分阶段参考参数。',
 'TRACKING_RANGE_DOWN_BPM': '旧共享向下搜索范围兼容覆盖；None 表示使用分阶段参考参数。',
 'TRACKING_SLEW_LIMIT_UP_BPM': '旧共享上行 limit 兼容覆盖；None 表示使用分阶段参考参数。',
 'TRACKING_SLEW_STEP_UP_BPM': '旧共享上行 step 兼容覆盖；None 表示使用分阶段参考参数。',
 'TRACKING_SLEW_LIMIT_DOWN_BPM': '旧共享下行 limit 兼容覆盖；None 表示使用分阶段参考参数。',
 'TRACKING_SLEW_STEP_DOWN_BPM': '旧共享下行 step 兼容覆盖；None 表示使用分阶段参考参数。',
 'CANDIDATE_PEAK_THRESHOLD_RATIO': '运动惩罚参考峰相对主峰门限；参考默认 0.30。',
 'SEARCH_SPACE': 'Optuna 离散候选集合；下方逐项列表可直接增删候选值。',
 'SEARCH_SPACE.Fs_Target': '训练重采样率候选(Hz)；较低值更快，较高值时间分辨率更高。',
 'NUM_SEED_POINTS': 'TPE 启动前的随机探索 trial 数；不得超过总 trial 预算。',
 'N_JOBS': '并行 worker 数；1 最稳定，增大时注意内存和可复现性。',
 'PARALLEL_REPEATS': '重复实验并行数；当前实际并行度由 n_jobs 优先控制。',
 'TRIAL_CACHE_MAX_ENTRIES': 'trial 结果内存缓存上限；越大越占内存。',
 'SAVE_STAGE_JSON': '是否保存详细 stage JSON；调试有用但会增加磁盘文件。',
 'DEBUG_MODE': '调试模式；会保留更多中间记录并隐式启用 stage JSON。',
 'CLEAN_OUTPUTS': '是否清空当前运行目录；续跑 checkpoint 时必须保持 False。',
 'VAL_GROUPS_PER_TYPE': 'split 模式每个运动类型的验证组数；all_train 下不使用。',
 'TEST_GROUPS_PER_TYPE': 'split 模式每个运动类型的测试组数；all_train 下不使用。',
 'CASCADE_TRAIN_BUDGETS': '按 scheme 覆盖 trial/repeat 预算；None 使用全局预算。',
 'TRIAL_PARAM_OVERRIDES': '固定 trial 参数映射；覆盖默认值并进入元数据、缓存键和断点指纹。',
 'RUN_STATIC_QC': '是否在第 1 节读取各文件前 10 秒并显示静态 QC。',
 'RUN_PREPROCESS_PREVIEW': '是否在第 2 节完整预处理一个所选样本。',
 'RUN_ALIGNMENT_DIAGNOSTICS': '是否运行第 3 节并写静息 Tdelay 对齐图。',
 'RUN_FULLFIELD_PLOT': '是否运行第 4 节并写全段未对齐 PPG-HR 图。',
 'RUN_RAW_PPG_PLOT': '是否运行第 6 节并写原始 PPG 双轴图。',
 'RUN_ALL_TRAIN': '是否运行第 7 节正式 200-trial 模式级优化。',
 'RUN_CONNECTIVITY_CHECK': '是否运行第 8 节小预算模式连通性检查。',
 'RUN_OUTPUT_CHECK': '是否运行第 9 节只读检查 Stage-6 和断点文件。',
 'RUN_STAGE7_REPLAY': '是否运行第 10 节单样本 best-params 回放。',
 'RUN_WINDOW_DIAGNOSTICS': '是否运行第 11 节窗口波形/频谱诊断。',
 'RUN_CROSS_MOTION_SUMMARY': '是否运行第 12 节跨运动汇总。',
 'RUN_CROSS_STAGE7_REPLAY': '是否运行第 13 节跨 scheme 回放。',
 'RUN_CROSS_WINDOW_DIAGNOSTICS': '是否运行第 14 节跨 scheme 窗口诊断。',
 'RUN_BATCH_REFERENCE_COMPARE': '是否运行第 15 节批量参考 scheme 对比。',
 'RUN_BATCH_TARGET_HR_REDRAW': '是否运行第 16 节并批量重绘当前受试者全部目标段 HR 图。',
 'CONNECTIVITY_MAX_ITERATIONS': '连通性检查每模式 trial 数；默认 1，避免误跑正式预算。'}


def code(source: str) -> nbformat.NotebookNode:
    """Create one normalized code cell."""

    return nbformat.v4.new_code_cell(source.strip() + "\n")


def config_code(source: str) -> nbformat.NotebookNode:
    """Create a config cell and require an inline explanation on every option."""

    annotated: list[str] = []
    for raw_line in source.strip().splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if CONFIG_LINE_RE.match(line) and "#" not in line:
            if stripped.startswith('"'):
                value_name = stripped.split(":", 1)[1].split(",", 1)[0].strip()
                lookup = value_name
            else:
                lookup = stripped.split("=", 1)[0].strip()
            comment = CONFIG_COMMENTS.get(lookup)
            if comment is None:
                raise ValueError(f"Missing inline config explanation for {lookup}: {stripped}")
            line = f"{line}  # {comment}"
        annotated.append(line)
    return nbformat.v4.new_code_cell("\n".join(annotated) + "\n")


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
执行与输出。所有机制开关由冻结preset显式保存；Notebook不提供rescue覆盖入口。


冻结配置通过 `resolve_core_config()` 解析，preset_name=core_enhanced_v1、preset_version=1。
未知字段、旧字段及冻结机制覆盖会报错。metadata记录完整有效参数、显式feature flags和参考SHA；
batch fingerprint同时包含schema与输入签名，配置变化不得复用旧checkpoint。

| 配置 | 单位 / 默认值 | 角色与约束 |
|---|---|---|
| hr_window_s / hr_step_s | 秒 / 8 / 1 | 固定observation，全部状态使用center；不参与BO |
| Fs_Target | Hz / 固定验收100；BO25或50 | 只改变算法采样域，不改变source IMU和原始PPG |
| memory_ms | ms / 160；BO40/80/120/160/200 | 派生正时延history_tap_limit=round(memory_ms*Fs/1000)；实际M/K仍按延迟估计，K_max=12不代表统一物理时长 |
| prewarm_s | 秒 / 0 | 独立前置上下文，冻结；不承担HR窗长角色 |
| LMS_Mu_Base / klms_step_size | 无量纲 / .01 / .05 | 搜索滤波器步长；RFF用LMS字段，KLMS只搜索自己的step |
| Spec_Penalty_Width / Weight | Hz半宽 / .2；保留比例 / .2 | 只搜索width；weight越小抑制越强，confidence共用，区域外权重1 |
| M_base / C_scale / K_max / Kstop | 2 / 1.2 / 12 / .3 | 滤波器内部映射冻结，不为对齐参考项目而强行重解释 |
| motion_policy | single_bout | detector保留所有runs；policy最长段（等长最早）或multi_bout全部保留 |
| reference_groups_order | ordered tuple | 唯一外部真源；ACC+HF和HF+ACC不同实验，组内固定顺序 |
| target_scope | motion_post10 | 控制实际adaptive范围；motion_only恢复立即reset-FFT，post10最迟范围结束切换 |

实验轴为filter、FFT/SSR、reference groups、protocol、scope及PPG transform。
`('ACC',)`→accx/accy/accz；`('ACC','HF')`→accx/accy/accz/hf1/hf2；
`('HF','ACC')`→hf1/hf2/accx/accy/accz。canonical key由registry产生，关联输出目录、缓存、记录与回放。

四层职责：未惩罚candidate spectrum（FFT/SSR）→共享soft penalty与candidate tracker→
简单adaptive→reset-FFT source handoff→phase-aware Final dynamics。tracker第一次slew保护峰连续性；
Final第二次slew保护跨phase/source拼接，上一Final跨phase延续。rest up1.5/1.5、down3/1.5；
motion up5.5/3.5、down2/1.5；recovery up1.5/1.5、down3.5/3（limit/step，bpm）。
恢复等待5秒，差值≤6bpm连续2有效窗切换，最迟20秒有效reset-FFT切换；episode内锁存，下一motion重置。
low/high-lock、rise-lineage、gap/rising rescue、complex shadow和interface-quality在线介入显式关闭。

raw-source detector：100Hz六轴四阶零相位带通，ACC .5–5Hz、Gyro .5–10Hz，分别magnitude；
8/1秒STD(ddof=1)，前30秒baseline，阈值max(2.5*baseline_std,.05*max_window_score)，任一路严格超过即motion。
先桥接≤3窗内部false gap，再删除<5窗true run。记录raw和selected两套结果；run为首true center到下一false center半开区间。
每窗记录episode和恢复时间；首段前rest，每段后直到下一motion或结束均recovery。

信号血缘：ppg_acquired_raw（source网格、缺口NaN、毛刺原值）→ppg_clean（missing/spike repair）→
ppg_algorithm_input（bandpass/log_absorbance及目标域resample）。原始PPG与raw Uh/Uc、标定UD calculation、时间和QC
组成独立InterfaceQualityInput；adaptive UD使用另一个预处理副本。quality暂未实现，不接收adaptive output或参考HR。
零相位、全记录阈值及前视滤波使本实验不声称严格实时因果。
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

from ppg_hr.params import (ReferenceConfig, MotionType, TargetScope, resolve_core_config, core_preset_metadata)
from ppg_hr.experimental.cascade_solver import run_protocol_trial
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
    redraw_subject_target_hr_curves,
    run_batch_adaptive_protocol,
    run_batch_reference_compare,
)
from ppg_hr.experimental.segmentation import apply_motion_protocol, detect_source_imu_segments
from ppg_hr.preprocess.calibration import load_subject_calibration

# 唯一数据输入。不要把本机绝对路径提交到仓库。
SUBJECT_DIR = Path(os.environ.get("PPG_SUBJECT_DIR", PROJECT_ROOT / "subject_data/lzj")).resolve()
SUBJECT_NAME = SUBJECT_DIR.name
OUTPUT_ROOT = SUBJECT_DIR.parent / "outputs"
MOTION_TYPES = tuple(item.value for item in MotionType)
assert MOTION_TYPES == ("write", "gripper", "run", "rope")

TARGET_SCOPES = [TargetScope.MOTION_POST10.value]
REFERENCE_GROUP_ORDERS = [
    ('ACC',),
    ('HF',),
    ('UD',),
    ('ACC', 'HF'),
    ('HF', 'CF'),
    ('ACC', 'UD'),
]
ADAPTIVE_FILTERS = ["lms"]
OBJECTIVE_MODE = "posthoc_aae"
DATA_SPLIT_MODE = "all_train"
MAX_ITERATIONS = 200
NUM_REPEATS = 1
RANDOM_STATE = 42
CORE_PARAMS = resolve_core_config()
prewarm_s = CORE_PARAMS.prewarm_s
UD_CALCULATION_MODE = "smoothed"  # raw=原始 Uh/Uc 直接计算；smoothed=四路电压分别平滑后计算，默认为模式2。
UD_SMOOTHING_WINDOW_S = 0.1  # 模式2中心移动中位数窗口(秒)；100 Hz 下 0.1 秒约为 10 点，可调。
TRAIN_HR_POSTPROCESS_METHOD = "fft"  # 可选 fft/ssr，并进入运行目录名

# 核心增强追踪开关来自冻结preset；此处只配置独立protocol policy。
MOTION_POLICY = "single_bout"  # protocol选择策略；检测器始终返回全部runs。

RUN_OUTPUT_DIR = build_subject_output_dir(
    subject_dir=SUBJECT_DIR,
    target_scopes=TARGET_SCOPES,
    reference_group_orders=REFERENCE_GROUP_ORDERS,
    adaptive_filters=ADAPTIVE_FILTERS,
    objective_mode=OBJECTIVE_MODE,
    data_split_mode=DATA_SPLIT_MODE,
    prewarm_s=prewarm_s,
    postprocess_method=TRAIN_HR_POSTPROCESS_METHOD,
)
RESULTS_ROOT = RUN_OUTPUT_DIR
print("SUBJECT_DIR:", SUBJECT_DIR)
print("RUN_OUTPUT_DIR:", RUN_OUTPUT_DIR)
'''
        ),
        config_code(
            r'''
# 用途：配置采样、预处理、静息 HR、Tdelay、post-hoc 对齐和恢复策略。
# 输入：这些值会进入预览、训练、缓存键、结果元数据和 replay 参数恢复。
# 输出：固定参数变量；不计算信号。
# 是否写文件：否。
# 耗时风险：无。
FS_ORIGIN = 100
FS_TARGET = 100
hr_window_s = 8
NORMALIZATION_MODE = CORE_PARAMS.normalization_mode
QC_POLICY = CORE_PARAMS.qc_policy
DELAY_ESTIMATION_MODE = CORE_PARAMS.delay_estimation_mode  # core固定envelope；仅延迟估计，不读取参考HR。
EVALUATION_WINDOW_S = CORE_PARAMS.evaluation_window_s
EVALUATION_STEP_S = CORE_PARAMS.evaluation_step_s

REST_HR_BAND_BPM = CORE_PARAMS.Rest_HR_Band_BPM
REST_HR_BAND_HZ = tuple(value / 60.0 for value in REST_HR_BAND_BPM)
REST_HR_TRACK_BAND_BPM = CORE_PARAMS.Rest_HR_Track_Band_BPM
REST_HR_SLEW_LIMIT_BPM = CORE_PARAMS.Rest_HR_Slew_Limit_BPM
REST_HR_SLEW_STEP_BPM = CORE_PARAMS.Rest_HR_Slew_Step_BPM
REST_HR_SMOOTH_METHOD = CORE_PARAMS.Rest_HR_Smooth_Method
REST_HR_SMOOTH_WIN = CORE_PARAMS.Rest_HR_Smooth_Win
REST_HR_PEAK_PERCENT = CORE_PARAMS.Rest_HR_Peak_Percent
REST_HR_SPEC_PENALTY_ENABLE = CORE_PARAMS.Rest_HR_Spec_Penalty_Enable
REST_HR_SPEC_PENALTY_WEIGHT = CORE_PARAMS.Rest_HR_Spec_Penalty_Weight
REST_HR_SPEC_PENALTY_WIDTH_HZ = CORE_PARAMS.Rest_HR_Spec_Penalty_Width_Hz
REST_ALIGNMENT_SCORE_MODE = "aae"

ENABLE_TIME_BIAS_AFTER = True
TIME_BIAS_AFTER_RANGE_S = (-5.0, 5.0)
TIME_BIAS_AFTER_STEP_S = 1.0
TIME_BIAS_AFTER_MODE = "posthoc_oracle_alignment"

PPG_INPUT_TRANSFORM = "log_absorbance"
LOG_ABSORBANCE_BASELINE_MODE = "rolling_median"
LOG_ABSORBANCE_BASELINE_WINDOW_S = 5.0
LOG_ABSORBANCE_EPS = 1e-6
LOG_ABSORBANCE_RATIO_CLIP = (1e-3, 1e3)

REST_HR_KWARGS = {
    "hr_band_bpm": REST_HR_BAND_BPM,
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
        config_code(
            r'''
# 用途：查看冻结核心，并构造固定参数验收配置，避免用 BO 掩盖结构问题。
# 输入：实验轴与 params.py 的完整 v1 快照。
# 输出：FIXED_CORE_PARAMS、PRESET_METADATA；只读展示，不运行算法。
# 是否写文件：否。
# 耗时风险：无；四层流程与所有 rescue 开关都在 preset metadata 中可检查。
REDRAW_HR_POSTPROCESS_METHOD = None  # replay 默认恢复记录的实际 estimator。
FIXED_REFERENCE_GROUPS_ORDER = ("ACC", "HF")  # 单组例：( "UD", )；反向例：( "HF", "ACC" )。
FIXED_CORE_PARAMS = resolve_core_config({  # 固定验收配置；所有其余参数来自v1快照。
    "Fs_Target": 100,  # 固定验收使用100Hz；BO另选25/50，detector仍在source域。
    "memory_ms": 160,  # 历史正延迟上限约160ms；M/K仍由延迟映射，K并未物理统一。
    "LMS_Mu_Base": 0.01,  # 固定验收步长；不使用BO寻找更好指标。
    "postprocess_method": TRAIN_HR_POSTPROCESS_METHOD,  # FFT/SSR仅改变谱估计器。
    "ppg_input_transform": PPG_INPUT_TRANSFORM,  # 从clean PPG生成algorithm input。
    "motion_policy": MOTION_POLICY,  # single_bout最长段，multi_bout保留全部。
})
PRESET_METADATA = core_preset_metadata(FIXED_CORE_PARAMS)  # 核查preset/version、flags、核心参数和参考SHA。
display(PRESET_METADATA)
'''
        ),
        config_code(
            r'''
# 用途：汇总固定 trial 参数、搜索空间、并行/缓存策略和各章节安全开关。
# 输入：前面三个配置单元；修改任一固定参数会进入结果元数据和 checkpoint 指纹。
# 输出：SEARCH_SPACE、TRIAL_PARAM_OVERRIDES 和运行控制变量。
# 是否写文件：否。
# 耗时风险：无；所有昂贵或写文件的 RUN_* 开关默认 False。
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
    "postprocess_method": TRAIN_HR_POSTPROCESS_METHOD,
    "ppg_input_transform": PPG_INPUT_TRANSFORM,
    "motion_policy": MOTION_POLICY,  # 单/多bout只控制protocol选择，不修改detector。
    "Enable_Time_Bias_After": ENABLE_TIME_BIAS_AFTER,
    "Time_Bias_After_Range_S": TIME_BIAS_AFTER_RANGE_S,
    "Time_Bias_After_Step_S": TIME_BIAS_AFTER_STEP_S,
    "Time_Bias_After_Mode": TIME_BIAS_AFTER_MODE,
}
SEARCH_SPACE = default_protocol_search_space()  # 每滤波器只有四维；候选列表集中于params.py。
for filter_name in ADAPTIVE_FILTERS:
    print(filter_name, {name: SEARCH_SPACE.options(name, adaptive_filter=filter_name)
                        for name in SEARCH_SPACE.names_for_filter(filter_name)})

RUN_STATIC_QC = False
RUN_PREPROCESS_PREVIEW = False
RUN_ALIGNMENT_DIAGNOSTICS = False
RUN_FULLFIELD_PLOT = False
RUN_RAW_PPG_PLOT = False
RUN_ALL_TRAIN = False
RUN_CONNECTIVITY_CHECK = False
RUN_OUTPUT_CHECK = False
RUN_STAGE7_REPLAY = False
RUN_WINDOW_DIAGNOSTICS = False
RUN_CROSS_MOTION_SUMMARY = False
RUN_CROSS_STAGE7_REPLAY = False
RUN_CROSS_WINDOW_DIAGNOSTICS = False
RUN_BATCH_REFERENCE_COMPARE = False
RUN_BATCH_TARGET_HR_REDRAW = False
CONNECTIVITY_MAX_ITERATIONS = 1

RUN_FIXED_CORE = False  # 先固定运行并检查边界/曲线/metadata，再决定是否打开BO。
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
        """通过 motion/index 选择一个样本，检查 CF2、HF、HFcomp、UD、ACC、QC 和运动分段。

本节涉及的参数和可配置项如下：

| 参数 | 含义 | 可配置值与注意事项 |
| --- | --- | --- |
| `RUN_PREPROCESS_PREVIEW` | 本节总开关 | `True` 时读取并预处理一个配对；`False` 时只打印跳过信息。它不会创建 Optuna study，也不会写结果文件。 |
| `PREVIEW_MOTION_TYPE` | 要预览的运动类型 | 只能选择 `write/gripper/run/rope` 之一，并且必须在当前受试者目录中存在。 |
| `PREVIEW_MOTION_INDEX` | 同一运动类型下的测试序号 | 使用正整数，例如 `1`。motion 与 index 必须唯一匹配一个严格配对。 |
| `SUBJECT_DIR` | 当前唯一受试者目录 | 在第 0 节或环境变量 `PPG_SUBJECT_DIR` 中设置；本节不会扫描其他受试者。 |
| `calibration` | 当前受试者的标定系数 | 不是可搜索超参数；固定由第 1 节从 `SUBJECT_DIR.parent / "ck.mat"` 解析并校验。 |
| `FS_ORIGIN` | 原始传感器采样率 | change6 数据协议固定为 100 Hz；通常保持 `100`，修改后必须与采集序号和真实采样率一致。 |
| `hr_window_s` | 本节运动分段的窗口长度，单位秒 | 固定 `8` 秒，步长固定 `1` 秒；detector 与正式 HR observation 都使用此时间网格，但独立计算。 |

本节完成信号预处理和运动分段，并加载参考 HR，为后续对齐诊断准备数据；真正的全局
Tdelay 搜索和对齐图在第 3 节执行。`PPG_INPUT_TRANSFORM` 则在正式求解窗口内生效，
不会改变本节展示的原始预处理通道。""",
        r'''
# 用途：只读预处理选定样本，并显示新协议通道、标定元数据和分段摘要。
# 输入：PREVIEW_MOTION_TYPE/PREVIEW_MOTION_INDEX，以及第 1 节的 discovery/calibration。
# 输出：preview_pair、preview_dataset、preview_segment；不保存图表。
# 是否写文件：否。
# 耗时风险：中；会完整读取并预处理一个传感器/HR 配对。
# RUN_PREPROCESS_PREVIEW: True 执行本节；False 安全跳过，不读取完整样本。
# PREVIEW_MOTION_TYPE: 仅允许 write/gripper/run/rope，必须与当前受试者文件名一致。
PREVIEW_MOTION_TYPE = "write"
# PREVIEW_MOTION_INDEX: 同一 motion 下的正整数测试序号；必须唯一匹配一个配对。
PREVIEW_MOTION_INDEX = 1
# SUBJECT_DIR: 唯一受试者目录，已在第 0 节解析；本节不会切换或扫描其他受试者。
# calibration: 第 1 节从 SUBJECT_DIR.parent/ck.mat 得到，不允许在此硬编码标定数值。
# FS_ORIGIN: 原始采样率，change6/Pydisplay 协议固定按 100 Hz 和序号字段校验。
# hr_window_s: detector 和 observation 固定8秒/1秒；prewarm、memory 不改变窗口数量。
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
            ud_calculation_mode=UD_CALCULATION_MODE,
            ud_smoothing_window_s=UD_SMOOTHING_WINDOW_S,
        )
        datasets[preview_pair.motion_id] = preview_dataset
        preview_segment = apply_motion_protocol(detect_source_imu_segments(preview_dataset.source_imu), MOTION_POLICY)
        display(preview_dataset.to_frame().head())
        print("样本：", preview_pair.sample_id)
        print("标定/序号元数据：", preview_dataset.source_metadata)
        print("分段：", preview_segment)
else:
    print("RUN_PREPROCESS_PREVIEW=False 或尚未发现配对，跳过预览。")
''',
    )
    cells.append(markdown('''
### 固定参数验收：先查看曲线与状态，再考虑搜索

输入是第2节的 preview_dataset。求解内部先取不含参考HR的 sensor_view；完整预测完成后才计算误差。
输出保存固定配置、各窗口状态、候选来源和Final。window_start_s / window_center_s / window_end_s
分别描述8秒窗的边界与中心，phase、scope和handoff都以center判断。参考覆盖不足只使误差为NaN。
'''))
    cells.append(code(r'''
# 用途：固定100Hz、160ms、LMS步长0.01运行核心，不创建Optuna study。
# 输入：preview_dataset（先打开RUN_PREPROCESS_PREVIEW）。
# 输出：fixed_core/窗口CSV和完整preset JSON，检查中心时间、phase、source与handoff。
# 是否写文件：是，仅RUN_FIXED_CORE=True时。
# 耗时风险：中；完整求解一个记录。
fixed_result = None
if RUN_FIXED_CORE:
    if preview_dataset is None:
        raise ValueError("请先运行第2节预览")
    fixed_result = run_protocol_trial(preview_dataset, FIXED_REFERENCE_GROUPS_ORDER,
                                      TARGET_SCOPES[0], FIXED_CORE_PARAMS, collect_frame=True)
    if not fixed_result.success:
        raise RuntimeError(fixed_result.reason)
    fixed_dir = RUN_OUTPUT_DIR / "fixed_core"
    fixed_dir.mkdir(parents=True, exist_ok=True)
    fixed_result.frame.to_csv(fixed_dir / "windows.csv", index=False)
    import json
    (fixed_dir / "preset.json").write_text(json.dumps(PRESET_METADATA, indent=2), encoding="utf-8")
    display(fixed_result.frame.head())
    print("Scope Final AAE:", fixed_result.final_aae_bpm, "检测/选择:", fixed_result.segment_info)
else:
    print("RUN_FIXED_CORE=False，跳过固定求解。")
'''))
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
                pair.sensor_csv,
                pair.ref_csv,
                fs_origin=FS_ORIGIN,
                calibration=calibration,
                ud_calculation_mode=UD_CALCULATION_MODE,
                ud_smoothing_window_s=UD_SMOOTHING_WINDOW_S,
            )
    alignment_paths = plot_rest_alignment_diagnostics_by_motion_type(
        discovery.pairs,
        datasets,
        RUN_OUTPUT_DIR / "alignment_diagnostics",
        fs_target=FS_TARGET,
        hr_window_s=hr_window_s,
        evaluation_window_s=EVALUATION_WINDOW_S,
        evaluation_step_s=EVALUATION_STEP_S,
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
                pair.sensor_csv,
                pair.ref_csv,
                fs_origin=FS_ORIGIN,
                calibration=calibration,
                ud_calculation_mode=UD_CALCULATION_MODE,
                ud_smoothing_window_s=UD_SMOOTHING_WINDOW_S,
            )
    fullfield_paths = plot_unaligned_fullfield_ppg_hr_by_motion_type(
        discovery.pairs,
        datasets,
        RUN_OUTPUT_DIR / "fullfield_ppg_hr",
        fs_target=FS_TARGET,
        hr_window_s=hr_window_s,
        step_s=EVALUATION_STEP_S,
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
def notebook_progress(info):
    """每个运动-模式完成或从断点恢复后，仅打印一次最终 post-hoc 指标。"""
    if info.get("stage") != "optimization_mode_completed":
        return
    state = "断点恢复" if info.get("resumed") else ("训练完成" if info.get("success") else "训练失败")
    current = f"{int(info.get('mode_current', 0))}/{int(info.get('mode_total', 0))}"
    mode = (
        f"{info.get('motion_type')} / {info.get('target_scope_value')} / "
        f"{info.get('reference_order_key')} / {str(info.get('adaptive_filter')).upper()}"
    )
    if info.get("success"):
        print(
            f"[{current}] {state}: {mode} | "
            f"final post-hoc AAE={float(info.get('posthoc_final_aae_bpm')):.3f} bpm, "
            f"accuracy={float(info.get('posthoc_final_acc_pct')):.2f}% | "
            f"ACC(同参数) post-hoc AAE={float(info.get('acc_compare_posthoc_aae_bpm')):.3f} bpm, "
            f"accuracy={float(info.get('acc_compare_posthoc_accuracy_pct')):.2f}% | "
            f"耗时={float(info.get('training_elapsed_s', 0.0)):.1f}s"
        )
    else:
        print(f"[{current}] {state}: {mode} | 原因={info.get('reason', '')} | 耗时={float(info.get('training_elapsed_s', 0.0)):.1f}s")


def run_training_cell(*, max_iterations, target_scopes, reference_group_orders, adaptive_filters):
    return run_batch_adaptive_protocol(
        subject_dir=SUBJECT_DIR,
        output_root=RUN_OUTPUT_DIR,
        max_iterations=max_iterations,
        num_repeats=NUM_REPEATS,
        random_state=RANDOM_STATE,
        num_seed_points=NUM_SEED_POINTS,
        fs_origin=FS_ORIGIN,
        ud_calculation_mode=UD_CALCULATION_MODE,
        ud_smoothing_window_s=UD_SMOOTHING_WINDOW_S,
        parallel_repeats=PARALLEL_REPEATS,
        n_jobs=N_JOBS,
        trial_cache_max_entries=TRIAL_CACHE_MAX_ENTRIES,
        save_stage_json=SAVE_STAGE_JSON,
        debug_mode=DEBUG_MODE,
        search_space=SEARCH_SPACE,
        trial_param_overrides=TRIAL_PARAM_OVERRIDES,
        target_scopes=target_scopes,
        reference_group_orders=reference_group_orders,
        adaptive_filters=adaptive_filters,
        objective_mode=OBJECTIVE_MODE,
        data_split_mode=DATA_SPLIT_MODE,
        delay_estimation_mode=DELAY_ESTIMATION_MODE,
        val_groups_per_type=VAL_GROUPS_PER_TYPE,
        test_groups_per_type=TEST_GROUPS_PER_TYPE,
        cascade_train_budgets=CASCADE_TRAIN_BUDGETS,
        project_root=SUBJECT_DIR.parent,
        clean_outputs=CLEAN_OUTPUTS,
        progress_callback=notebook_progress,
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
                pair.sensor_csv,
                pair.ref_csv,
                fs_origin=FS_ORIGIN,
                calibration=calibration,
                ud_calculation_mode=UD_CALCULATION_MODE,
                ud_smoothing_window_s=UD_SMOOTHING_WINDOW_S,
            )
    raw_ppg_paths = plot_raw_ppg_and_unaligned_hr_by_motion_type(
        discovery.pairs,
        datasets,
        RUN_OUTPUT_DIR / "raw_ppg_dual_axis",
        fs_target=FS_TARGET,
        fs_origin=FS_ORIGIN,
        hr_window_s=hr_window_s,
        step_s=EVALUATION_STEP_S,
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
        reference_group_orders=REFERENCE_GROUP_ORDERS,
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
configured_mode_count = motion_count * len(TARGET_SCOPES) * len(REFERENCE_GROUP_ORDERS) * len(ADAPTIVE_FILTERS)
print("当前配置模式数：", configured_mode_count)
result_connectivity = None
if RUN_CONNECTIVITY_CHECK:
    result_connectivity = run_training_cell(
        max_iterations=CONNECTIVITY_MAX_ITERATIONS,
        target_scopes=TARGET_SCOPES,
        reference_group_orders=REFERENCE_GROUP_ORDERS,
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
    cells += section(
        10,
        "对单个样本执行 best-params HR 曲线回放",
        "从当前受试者的配对结果选择文件，从 Stage-6 记录恢复参数，不重新执行 Optuna。",
        r'''
# 用途：按 motion/index 选择当前受试者样本，回放完整级联和 guard 后两套 HR 曲线。
# 输入：RESULTS_ROOT、REPLAY_MOTION_TYPE/INDEX、scope/filter/scheme 选择器。
# 输出：RESULTS_ROOT/replay 下的 CSV 和 PNG。
# 是否写文件：是，仅在 RUN_STAGE7_REPLAY=True 时。
# 耗时风险：中；重新执行一次已保存参数的信号滤波，但不会训练或创建 study。
REPLAY_MOTION_TYPE = "write"
REPLAY_MOTION_INDEX = 1
REPLAY_TARGET_SCOPE = TargetScope.MOTION_POST10.value
REPLAY_ADAPTIVE_FILTER = "lms"
REPLAY_REFERENCE_ORDER_KEY = ReferenceConfig(('ACC',)).reference_order_key
REPLAY_ADAPTIVE_DATA_TYPE = ""
REPLAY_prewarm_s = prewarm_s
REPLAY_SPLIT = ""
REPLAY_MODE = ""
REPLAY_OUTPUT_DIR = Path(RESULTS_ROOT) / "replay"

def select_subject_pair(motion_type, motion_index):
    """Return one strict pair from the current subject discovery result."""
    if discovery is None:
        raise RuntimeError("请先运行第 1 节完成受试者文件发现。")
    matches = [
        pair for pair in discovery.pairs
        if pair.motion_type == str(motion_type) and pair.motion_index == int(motion_index)
    ]
    if len(matches) != 1:
        raise ValueError(f"样本选择必须唯一，当前匹配数={len(matches)}: {motion_type}/{motion_index}")
    return matches[0]

replay_paths = {}
replay_pair = None
if RUN_STAGE7_REPLAY:
    replay_pair = select_subject_pair(REPLAY_MOTION_TYPE, REPLAY_MOTION_INDEX)
    replay_paths = replay_best_record_hr_curves(
        signal_csv=replay_pair.sensor_csv,
        ref_csv=replay_pair.ref_csv,
        results_root=RESULTS_ROOT,
        output_dir=REPLAY_OUTPUT_DIR,
        motion_type=REPLAY_MOTION_TYPE,
        split=REPLAY_SPLIT,
        mode=REPLAY_MODE,
        target_scope=REPLAY_TARGET_SCOPE,
        adaptive_filter=REPLAY_ADAPTIVE_FILTER,
        adaptive_data_type=REPLAY_ADAPTIVE_DATA_TYPE,
        reference_config=REPLAY_REFERENCE_ORDER_KEY,
        prewarm_s=REPLAY_prewarm_s,
        fs_origin=FS_ORIGIN,
        ud_calculation_mode=UD_CALCULATION_MODE,
        ud_smoothing_window_s=UD_SMOOTHING_WINDOW_S,
        postprocess_method_override=REDRAW_HR_POSTPROCESS_METHOD,
    )
    print(replay_paths)
    display(Image(filename=str(replay_paths["plot"])))
else:
    print("RUN_STAGE7_REPLAY=False，跳过单样本回放。")
''',
    )
    cells += section(
        11,
        "对单个样本输出窗口级波形与频谱诊断",
        "复用第 10 节的样本和参数选择器，查看指定传感器时间窗口的输入、频谱、权重和追踪状态。",
        r'''
# 用途：从 Stage-6 参数记录重建一个窗口的完整诊断图和状态摘要。
# 输入：第 10 节 REPLAY_* 配置以及 DIAGNOSTIC_WINDOW_START_S。
# 输出：RESULTS_ROOT/window_diagnostics 下的图表和诊断信息。
# 是否写文件：是，仅在 RUN_WINDOW_DIAGNOSTICS=True 时。
# 耗时风险：中；执行单样本回放和单窗口诊断，不启动 Optuna。
DIAGNOSTIC_OUTPUT_DIR = Path(RESULTS_ROOT) / "window_diagnostics"
DIAGNOSTIC_WINDOW_START_S = 80.0
window_diagnostics = {}
if RUN_WINDOW_DIAGNOSTICS:
    diagnostic_pair = select_subject_pair(REPLAY_MOTION_TYPE, REPLAY_MOTION_INDEX)
    window_diagnostics = plot_window_diagnostics_from_records(
        signal_csv=diagnostic_pair.sensor_csv,
        ref_csv=diagnostic_pair.ref_csv,
        results_root=RESULTS_ROOT,
        output_dir=DIAGNOSTIC_OUTPUT_DIR,
        window_start_s=DIAGNOSTIC_WINDOW_START_S,
        motion_type=REPLAY_MOTION_TYPE,
        split=REPLAY_SPLIT,
        mode=REPLAY_MODE,
        target_scope=REPLAY_TARGET_SCOPE,
        adaptive_filter=REPLAY_ADAPTIVE_FILTER,
        adaptive_data_type=REPLAY_ADAPTIVE_DATA_TYPE,
        reference_config=REPLAY_REFERENCE_ORDER_KEY,
        prewarm_s=REPLAY_prewarm_s,
        fs_origin=FS_ORIGIN,
        ud_calculation_mode=UD_CALCULATION_MODE,
        ud_smoothing_window_s=UD_SMOOTHING_WINDOW_S,
        postprocess_method_override=REDRAW_HR_POSTPROCESS_METHOD,
    )
    print(window_diagnostics)
else:
    print("RUN_WINDOW_DIAGNOSTICS=False，跳过窗口诊断。")
''',
    )
    cells += section(
        12,
        "输出指定模式的跨运动汇总表",
        "只读取四类运动已有 Stage-6 紧凑记录，并按 write/gripper/run/rope 顺序汇总。",
        r'''
# 用途：比较同一 filter/scheme/scope 在四类运动上的 AAE、准确率和最终来源分布。
# 输入：RESULTS_ROOT 和 SUMMARY_* 模式选择器。
# 输出：RESULTS_ROOT/summary_tables 下的 CSV。
# 是否写文件：是，仅在 RUN_CROSS_MOTION_SUMMARY=True 时。
# 耗时风险：低；只读取结果记录，不回放、不训练。
SUMMARY_ADAPTIVE_FILTER = "lms"
SUMMARY_REFERENCE_ORDER_KEY = ReferenceConfig(('ACC',)).reference_order_key
SUMMARY_ADAPTIVE_DATA_TYPE = ""
SUMMARY_TARGET_SCOPE = TargetScope.MOTION_POST10.value
SUMMARY_prewarm_s = prewarm_s
SUMMARY_TABLE_OUTPUT_DIR = Path(RESULTS_ROOT) / "summary_tables"
summary_path = None
if RUN_CROSS_MOTION_SUMMARY:
    summary_path = build_cross_motion_summary_table(
        table_output_dir=SUMMARY_TABLE_OUTPUT_DIR,
        results_root=RESULTS_ROOT,
        adaptive_filter=SUMMARY_ADAPTIVE_FILTER,
        adaptive_data_type=SUMMARY_ADAPTIVE_DATA_TYPE,
        reference_config=SUMMARY_REFERENCE_ORDER_KEY,
        target_scope=SUMMARY_TARGET_SCOPE,
        prewarm_s=SUMMARY_prewarm_s,
    )
    print(summary_path)
    display(pd.read_csv(summary_path))
else:
    print("RUN_CROSS_MOTION_SUMMARY=False，跳过跨运动汇总。")
''',
    )
    cells += section(
        13,
        "跨 scheme 参数来源与实际滤波回放",
        "使用 scheme A 保存的最优参数运行 scheme B，检查参考通道架构变化的影响。",
        r'''
# 用途：从一个 scheme 恢复参数，但用另一个 change6 scheme 实际执行滤波。
# 输入：CROSS_PARAM_SCHEME、CROSS_APPLIED_SCHEME 和当前受试者 motion/index。
# 输出：RESULTS_ROOT/cross_replay 下的 CSV 和 PNG。
# 是否写文件：是，仅在 RUN_CROSS_STAGE7_REPLAY=True 时。
# 耗时风险：中；执行两种 guard 变体的单样本回放，不启动 Optuna。
CROSS_MOTION_TYPE = "write"
CROSS_MOTION_INDEX = 1
CROSS_TARGET_SCOPE = TargetScope.MOTION_POST10.value
CROSS_ADAPTIVE_FILTER = "lms"
CROSS_PARAM_SCHEME = ReferenceConfig(('HF',)).reference_order_key
CROSS_APPLIED_SCHEME = ReferenceConfig(('ACC',)).reference_order_key
CROSS_prewarm_s = prewarm_s
CROSS_OUTPUT_DIR = Path(RESULTS_ROOT) / "cross_replay"
cross_replay_paths = {}
if RUN_CROSS_STAGE7_REPLAY:
    cross_pair = select_subject_pair(CROSS_MOTION_TYPE, CROSS_MOTION_INDEX)
    cross_replay_paths = replay_best_record_hr_curves(
        signal_csv=cross_pair.sensor_csv,
        ref_csv=cross_pair.ref_csv,
        results_root=RESULTS_ROOT,
        output_dir=CROSS_OUTPUT_DIR,
        motion_type=CROSS_MOTION_TYPE,
        target_scope=CROSS_TARGET_SCOPE,
        adaptive_filter=CROSS_ADAPTIVE_FILTER,
        reference_config=CROSS_PARAM_SCHEME,
        override_reference_config=CROSS_APPLIED_SCHEME,
        prewarm_s=CROSS_prewarm_s,
        fs_origin=FS_ORIGIN,
        ud_calculation_mode=UD_CALCULATION_MODE,
        ud_smoothing_window_s=UD_SMOOTHING_WINDOW_S,
        postprocess_method_override=REDRAW_HR_POSTPROCESS_METHOD,
    )
    print(cross_replay_paths)
else:
    print("RUN_CROSS_STAGE7_REPLAY=False，跳过跨 scheme 回放。")
''',
    )
    cells += section(
        14,
        "跨 scheme 窗口级诊断",
        "对第 13 节的参数来源 scheme 与实际运行 scheme 生成相同窗口的波形、频谱和状态诊断。",
        r'''
# 用途：定位跨 scheme 回放在指定窗口中的通道、频谱峰和追踪选择差异。
# 输入：第 13 节 CROSS_* 配置和 CROSS_WINDOW_START_S。
# 输出：RESULTS_ROOT/cross_window_diagnostics 下的诊断图和摘要。
# 是否写文件：是，仅在 RUN_CROSS_WINDOW_DIAGNOSTICS=True 时。
# 耗时风险：中；执行单样本、单窗口诊断，不启动 Optuna。
CROSS_DIAGNOSTIC_OUTPUT_DIR = Path(RESULTS_ROOT) / "cross_window_diagnostics"
CROSS_WINDOW_START_S = 80.0
cross_window_diagnostics = {}
if RUN_CROSS_WINDOW_DIAGNOSTICS:
    cross_diagnostic_pair = select_subject_pair(CROSS_MOTION_TYPE, CROSS_MOTION_INDEX)
    cross_window_diagnostics = plot_window_diagnostics_from_records(
        signal_csv=cross_diagnostic_pair.sensor_csv,
        ref_csv=cross_diagnostic_pair.ref_csv,
        results_root=RESULTS_ROOT,
        output_dir=CROSS_DIAGNOSTIC_OUTPUT_DIR,
        window_start_s=CROSS_WINDOW_START_S,
        motion_type=CROSS_MOTION_TYPE,
        target_scope=CROSS_TARGET_SCOPE,
        adaptive_filter=CROSS_ADAPTIVE_FILTER,
        reference_config=CROSS_PARAM_SCHEME,
        override_reference_config=CROSS_APPLIED_SCHEME,
        prewarm_s=CROSS_prewarm_s,
        fs_origin=FS_ORIGIN,
        ud_calculation_mode=UD_CALCULATION_MODE,
        ud_smoothing_window_s=UD_SMOOTHING_WINDOW_S,
        postprocess_method_override=REDRAW_HR_POSTPROCESS_METHOD,
    )
    print(cross_window_diagnostics)
else:
    print("RUN_CROSS_WINDOW_DIAGNOSTICS=False，跳过跨 scheme 窗口诊断。")
''',
    )
    cells += section(
        15,
        "批量参考通道 scheme 对比",
        "对一个运动类型的既有 split 记录复用同一套最优参数，比较参数来源 scheme 与实际参考 scheme。",
        r'''
# 用途：批量比较同一参数在两个 change6 scheme 下的最终与 adaptive 指标。
# 输入：REFERENCE_COMPARE_* 配置及对应 motion 的 best_params_and_alignment.csv。
# 输出：RESULTS_ROOT/batch_reference_compare 下的 CSV。
# 是否写文件：是，仅在 RUN_BATCH_REFERENCE_COMPARE=True 时。
# 耗时风险：中到高；逐样本复评估两套 scheme，但不会运行 Optuna。
REFERENCE_COMPARE_MOTION_TYPE = "write"
REFERENCE_COMPARE_PARAM_SCHEME = ReferenceConfig(('HF',)).reference_order_key
REFERENCE_COMPARE_APPLIED_SCHEME = ReferenceConfig(('ACC',)).reference_order_key
REFERENCE_COMPARE_TARGET_SCOPE = TargetScope.MOTION_POST10.value
REFERENCE_COMPARE_ADAPTIVE_FILTER = "lms"
REFERENCE_COMPARE_BEST_PARAMS_CSV = (
    Path(RESULTS_ROOT)
    / "motion_types"
    / REFERENCE_COMPARE_MOTION_TYPE
    / "best_params_and_alignment.csv"
)
REFERENCE_COMPARE_OUTPUT_DIR = Path(RESULTS_ROOT) / "batch_reference_compare"
reference_compare_paths = {}
if RUN_BATCH_REFERENCE_COMPARE:
    reference_compare_paths = run_batch_reference_compare(
        motion_type=REFERENCE_COMPARE_MOTION_TYPE,
        best_param_source_reference_config=REFERENCE_COMPARE_PARAM_SCHEME,
        actual_reference_reference_config=REFERENCE_COMPARE_APPLIED_SCHEME,
        target_scope=REFERENCE_COMPARE_TARGET_SCOPE,
        adaptive_filter=REFERENCE_COMPARE_ADAPTIVE_FILTER,
        best_param_csv_path=REFERENCE_COMPARE_BEST_PARAMS_CSV,
        output_dir=REFERENCE_COMPARE_OUTPUT_DIR,
        fs_origin=FS_ORIGIN,
        ud_calculation_mode=UD_CALCULATION_MODE,
        ud_smoothing_window_s=UD_SMOOTHING_WINDOW_S,
    )
    print(reference_compare_paths)
    display(pd.read_csv(reference_compare_paths["csv"]))
else:
    print("RUN_BATCH_REFERENCE_COMPARE=False，跳过批量参考通道对比。")
''',
    )
    cells += section(
        16,
        "批量重绘当前受试者全部目标段真实 HR—预测 HR",
        "选择一个已训练 scheme，按 write/gripper/run/rope 和 index 顺序回放全部样本；只绘制 post-hoc 对齐参考 HR 与 final HR。",
        r'''
# 用途：一次调用批量输出当前受试者全部 motion/index 的目标段真实 HR—最终预测 HR 图。
# 输入：SUBJECT_DIR、RESULTS_ROOT，以及下面单个可配置模式；读取已有 Stage-6 最优参数。
# 输出：每样本 PNG/CSV 和 batch_target_hr_manifest.csv，并在下方显示汇总表。
# 是否写文件：是，仅在 RUN_BATCH_TARGET_HR_REDRAW=True 时；不会创建 Optuna study。
# 耗时风险：中到高；每个样本回放一次滤波流程，但不训练、不搜索超参数。
BATCH_TARGET_SCOPE = TargetScope.MOTION_POST10.value  # 目标段；默认运动段至运动结束后 10 秒，仅控制评价范围。
BATCH_TARGET_REFERENCE_ORDER_KEY = ReferenceConfig(('ACC',)).reference_order_key  # 每次只回放一个 scheme；可改为其余五种 change6 scheme。
BATCH_TARGET_ADAPTIVE_FILTER = "volterra"  # 选择已训练滤波器；应与 Stage-6 最优记录完全匹配。
BATCH_TARGET_DATA_SPLIT_MODE = "all_train"  # 选择记录中的 mode；默认同 motion 全 index 联合训练。
BATCH_TARGET_prewarm_s = prewarm_s  # 匹配最优记录的前置收敛上下文；None 表示不按 prewarm_s 筛选。
BATCH_TARGET_OUTPUT_DIR = Path(RESULTS_ROOT) / "batch_target_hr_curves"  # 输出基目录；后端会追加模式签名。
batch_target_manifest = pd.DataFrame()
if RUN_BATCH_TARGET_HR_REDRAW:
    batch_target_manifest = redraw_subject_target_hr_curves(
        subject_dir=SUBJECT_DIR,
        results_root=RESULTS_ROOT,
        target_scope=BATCH_TARGET_SCOPE,
        reference_config=BATCH_TARGET_REFERENCE_ORDER_KEY,
        adaptive_filter=BATCH_TARGET_ADAPTIVE_FILTER,
        data_split_mode=BATCH_TARGET_DATA_SPLIT_MODE,
        prewarm_s=BATCH_TARGET_prewarm_s,
        output_dir=BATCH_TARGET_OUTPUT_DIR,
        fs_origin=FS_ORIGIN,
        ud_calculation_mode=UD_CALCULATION_MODE,
        ud_smoothing_window_s=UD_SMOOTHING_WINDOW_S,
        postprocess_method_override=REDRAW_HR_POSTPROCESS_METHOD,
    )
    success_count = int((batch_target_manifest["status"] == "ok").sum())
    failed_count = int((batch_target_manifest["status"] != "ok").sum())
    print(f"批量目标段重绘完成：成功 {success_count}，失败 {failed_count}")
    print("manifest:", batch_target_manifest.attrs.get("manifest_path"))
    display(batch_target_manifest)
else:
    print("RUN_BATCH_TARGET_HR_REDRAW=False，跳过全受试者目标段批量重绘。")
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
