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

CONFIG_COMMENTS = {
    # Sampling, preprocessing, alignment, post-hoc alignment and recovery.
    "FS_ORIGIN": "原始采样率(Hz)；change6/Pydisplay 数据固定按 100 Hz 与序号校验。",
    "FS_TARGET": "预览绘图目标采样率(Hz)；训练候选由 SEARCH_SPACE.Fs_Target 单独控制。",
    "TW": "预览分段窗口长度(秒)；训练窗口候选由 SEARCH_SPACE.TW 控制。",
    "TW_F": "自适应滤波前置收敛上下文(秒)；0 表示不额外增加前置上下文。",
    "NORMALIZATION_MODE": "参考通道归一化方式；minmax 为当前默认和推荐配置。",
    "QC_POLICY": "坏窗口策略；fallback_baseline 表示保留计算并回退 baseline HR。",
    "DELAY_ESTIMATION_MODE": "参考通道时延估计方式；envelope 为默认，direct 保留直接相关模式。",
    "ALIGNMENT_TW": "静息段全局 Tdelay 提取 HR 的窗口长度(秒)，不等同训练 TW。",
    "ALIGNMENT_STEP_S": "静息对齐窗口步长(秒)；越小时间分辨率越高、计算越慢。",
    "REST_HR_BAND_BPM": "静息 HR 合法范围(bpm)；用于对齐阶段的频带限制。",
    "REST_HR_BAND_HZ": "由 REST_HR_BAND_BPM 换算的 Hz 频带，供频谱接口使用。",
    "REST_HR_TRACK_BAND_BPM": "静息 HR 相邻窗口搜索带宽(bpm)；训练候选由搜索空间控制。",
    "REST_HR_SLEW_LIMIT_BPM": "静息 HR 允许直接接受的相邻变化阈值(bpm)。",
    "REST_HR_SLEW_STEP_BPM": "超过静息变化阈值时每窗口最大修正步长(bpm)。",
    "REST_HR_SMOOTH_METHOD": "静息 HR 平滑方法；median 对孤立异常峰更稳健。",
    "REST_HR_SMOOTH_WIN": "静息 HR 平滑窗口数；应使用正奇数以保持中心对齐。",
    "REST_HR_PEAK_PERCENT": "候选频谱峰相对主峰的最低幅值比例。",
    "REST_HR_SPEC_PENALTY_ENABLE": "是否在静息 HR 提取时启用运动频率惩罚。",
    "REST_HR_SPEC_PENALTY_WEIGHT": "静息运动频率惩罚权重；越大越抑制运动相关峰。",
    "REST_HR_SPEC_PENALTY_WIDTH_HZ": "静息运动峰惩罚半宽(Hz)；决定受抑制频带范围。",
    "REST_ALIGNMENT_SCORE_MODE": "Tdelay 评分指标；aae 为当前默认，std 用于兼容旧实验。",
    "ENABLE_TIME_BIAS_AFTER": "是否计算滤波后 post-hoc 时间偏移诊断。",
    "TIME_BIAS_AFTER_RANGE_S": "post-hoc 时间偏移搜索区间(秒)，按(最小,最大)设置。",
    "TIME_BIAS_AFTER_STEP_S": "post-hoc 时间偏移搜索步长(秒)。",
    "TIME_BIAS_AFTER_MODE": "post-hoc 对齐语义；oracle 只用于诊断指标，不参与部署选择。",
    "RECOVERY_GRACE_S": "运动结束后的恢复宽限时间(秒)。",
    "RECOVERY_DIFF_BPM": "恢复阶段候选与连续轨迹允许的最大差值(bpm)。",
    "RECOVERY_CROSS_DIFF_BPM": "恢复阶段跨来源切换允许的差值阈值(bpm)。",
    "PPG_INPUT_TRANSFORM": "PPG 输入变换；默认 log_absorbance，也可选择后端支持的既有变换。",
    "LOG_ABSORBANCE_BASELINE_MODE": "log-absorbance 基线估计方法；rolling_median 抑制慢漂移。",
    "LOG_ABSORBANCE_BASELINE_WINDOW_S": "滚动基线窗口长度(秒)；越大越保留低频趋势。",
    "LOG_ABSORBANCE_EPS": "log 比值的数值稳定下限，防止零值导致无穷大。",
    "LOG_ABSORBANCE_RATIO_CLIP": "进入 log 前的比值裁剪范围，防止极端异常值。",
    "GLOBAL_OBJECTIVE_STRATEGY": "全局目标组合策略；current_global_adaptive 沿用当前融合语义。",
    "REST_HR_KWARGS": "汇总静息 HR 绘图参数；键值直接传给第 3/4/6 节绘图接口。",
    # HR postprocessing and fixed adaptive-filter controls.
    "TRAIN_HR_POSTPROCESS_METHOD": "训练 HR 后处理方法；fft 为默认，ssr 启用稀疏频谱重建。",
    "SSR_NUM_ATOMS": "SSR 稀疏表示使用的原子数；越大表达力和计算量越高。",
    "SSR_LAMBDA": "SSR 稀疏正则权重；越大产生越稀疏的频谱解。",
    "SSR_HARMONIC_TOL_BPM": "SSR 谐波匹配容差(bpm)。",
    "SSR_FALLBACK_TO_FFT": "SSR 失败时是否回退既有 FFT 跟踪器。",
    "SSR_GRID_RESOLUTION_BPM": "SSR 心率搜索网格分辨率(bpm)；越小越精细但更慢。",
    "LMS_MU_MIN": "LMS/Volterra/RFF-LMS 更新步长下限，避免步长退化为零。",
    "TRACKING_SMOOTH_WIN_LEN": "最终 HR 轨迹平滑窗口数；使用正奇数。",
    "SPEC_PENALTY_WEIGHT": "通用运动频谱惩罚权重；越大越排斥参考通道峰。",
    "RFF_SIGMA": "旧记录缺少 sigma_scale 时使用的固定 RBF sigma 兼容值。",
    "RFF_UPDATE_MODE": "RFF 权重更新方式；nlms 按特征能量归一化，稳定性更好。",
    "RFF_NLMS_EPS": "RFF-NLMS 分母稳定项，防止特征能量接近零。",
    "RFF_LEAKAGE": "RFF 权重泄漏系数；0 表示不衰减历史权重。",
    "RFF_ERR_CLIP": "RFF 更新误差裁剪阈值；None 表示不额外裁剪。",
    "RFF_THETA_NORM_GUARD": "RFF 权重范数保护阈值；None 表示关闭该保护。",
    "KLMS_MAX_DICTIONARY_SIZE": "KLMS 字典中心上限；越大拟合能力和内存占用越高。",
    "KLMS_CENTER_PRUNE_POLICY": "KLMS 达到字典上限后的策略；freeze_new_centers 停止加中心。",
    "KLMS_DISTANCE_MODE": "KLMS 新颖度距离语义；normalized 可减弱 tap 维数影响。",
    "KLMS_NORMALIZED_UPDATE": "是否按核特征能量归一化 KLMS 更新。",
    "KLMS_NLMS_EPS": "KLMS 归一化更新的分母稳定项。",
    # Cascade guard and enhanced tracking.
    "CASCADE_GUARD_POLICY": "级联保护策略；none 关闭，rms_guard 按级输出 RMS 决定回退。",
    "CASCADE_GUARD_RATIO_MIN": "当前级/上一级 RMS 比值下限，低于时判定异常。",
    "CASCADE_GUARD_RATIO_MAX": "当前级/上一级 RMS 比值上限，高于时判定异常。",
    "CASCADE_GUARD_FLAT_STD_EPS": "近似平坦输出的标准差阈值。",
    "CASCADE_GUARD_USE_FINITE_ZSCORE": "是否把有限 z-score 检查纳入级联保护。",
    "TRACKER_MODE": "频谱追踪总模式；legacy 保留旧算法，enhanced 启用六项增强开关。",
    "ENABLE_DIRECTIONAL_TRACKING": "是否启用上行/下行不同搜索范围和 slew 限制。",
    "ENABLE_DYNAMIC_PENALTY": "是否按置信度和谐波存在性动态调整频谱惩罚。",
    "ENABLE_CONTINUITY_PROTECTION": "是否用连续性与挑战峰确认保护当前锁定。",
    "ENABLE_LOW_LOCK_RECOVERY": "是否启用低锁定确认和恢复；实际仅对 LMS 生效。",
    "ENABLE_HIGH_LOCK_RECOVERY": "是否启用高锁定风险确认、恢复和冷却。",
    "ENABLE_POST_MOTION_PROTECTION": "是否启用运动结束后 reset-FFT、交叉和 gap rescue 保护。",
    "TRACKING_RANGE_UP_BPM": "旧共享向上搜索范围兼容覆盖；None 表示使用分阶段参考参数。",
    "TRACKING_RANGE_DOWN_BPM": "旧共享向下搜索范围兼容覆盖；None 表示使用分阶段参考参数。",
    "TRACKING_SLEW_LIMIT_UP_BPM": "旧共享上行 limit 兼容覆盖；None 表示使用分阶段参考参数。",
    "TRACKING_SLEW_STEP_UP_BPM": "旧共享上行 step 兼容覆盖；None 表示使用分阶段参考参数。",
    "TRACKING_SLEW_LIMIT_DOWN_BPM": "旧共享下行 limit 兼容覆盖；None 表示使用分阶段参考参数。",
    "TRACKING_SLEW_STEP_DOWN_BPM": "旧共享下行 step 兼容覆盖；None 表示使用分阶段参考参数。",
    "MOTION_TRACKING_RANGE_UP_BPM": "adaptive 运动段向上候选搜索范围(bpm)。",
    "MOTION_TRACKING_RANGE_DOWN_BPM": "adaptive 运动段向下候选搜索范围(bpm)。",
    "MOTION_TRACKING_SLEW_LIMIT_UP_BPM": "adaptive 运动段上行可直接接受阈值(bpm)。",
    "MOTION_TRACKING_SLEW_STEP_UP_BPM": "adaptive 运动段上行超限时单窗口推进步长(bpm)。",
    "MOTION_TRACKING_SLEW_LIMIT_DOWN_BPM": "adaptive 运动段下行可直接接受阈值(bpm)。",
    "MOTION_TRACKING_SLEW_STEP_DOWN_BPM": "adaptive 运动段下行超限时单窗口推进步长(bpm)。",
    "RECOVERY_TRACKING_RANGE_UP_BPM": "adaptive 恢复段向上候选搜索范围(bpm)。",
    "RECOVERY_TRACKING_RANGE_DOWN_BPM": "adaptive 恢复段向下候选搜索范围(bpm)。",
    "RECOVERY_TRACKING_SLEW_LIMIT_UP_BPM": "adaptive 恢复段上行可直接接受阈值(bpm)。",
    "RECOVERY_TRACKING_SLEW_STEP_UP_BPM": "adaptive 恢复段上行超限时单窗口推进步长(bpm)。",
    "RECOVERY_TRACKING_SLEW_LIMIT_DOWN_BPM": "adaptive 恢复段下行可直接接受阈值(bpm)。",
    "RECOVERY_TRACKING_SLEW_STEP_DOWN_BPM": "adaptive 恢复段下行超限时单窗口推进步长(bpm)。",
    "CANDIDATE_PEAK_THRESHOLD_RATIO": "运动惩罚参考峰相对主峰门限；参考默认 0.30。",
    "FULL_CANDIDATE_PEAK_THRESHOLD_RATIO": "PPG 完整候选相对主峰门限；参考默认 0.15。",
    "LOW_LOCK_MIN_BPM": "低锁定判定区间下限(bpm)。",
    "LOW_LOCK_MAX_BPM": "低锁定判定区间上限(bpm)。",
    "LOW_LOCK_MIN_WINDOWS": "进入低锁定风险前要求连续命中的窗口数。",
    "LOW_LOCK_TARGET_MIN_BPM": "低锁定恢复候选的最低目标心率(bpm)。",
    "LOW_LOCK_MIN_JUMP_BPM": "恢复候选相对低锁定轨迹的最小上跳幅度(bpm)。",
    "LOW_LOCK_MIN_AMP_RATIO": "低锁定恢复候选相对主峰的最低幅值比例。",
    "LOW_LOCK_CANDIDATE_STABLE_BPM": "低锁定恢复候选跨窗口稳定容差(bpm)。",
    "LOW_LOCK_CONFIRM_WINDOWS": "接受低锁定恢复候选前的确认窗口数。",
    "LOW_LOCK_STEP_BPM": "低锁定恢复时单窗口允许的最大上调步长(bpm)。",
    "HIGH_LOCK_CONFIRM_WINDOWS": "确认高锁定风险所需的连续窗口数。",
    "HIGH_LOCK_COOLDOWN_WINDOWS": "高锁定恢复后的冷却窗口数，防止立即反复触发。",
    "HIGH_LOCK_MIN_GAP_BPM": "高锁定轨迹与较低挑战峰之间的最小间隔(bpm)。",
    "HIGH_LOCK_MIN_AMP_RATIO": "高锁定恢复挑战峰相对主峰的最低幅值比例。",
    "HIGH_LOCK_CANDIDATE_MIN_BPM": "高锁定恢复候选允许的最低心率(bpm)。",
    "HIGH_LOCK_CANDIDATE_STABLE_BPM": "高锁定恢复候选跨窗口稳定容差(bpm)。",
    "HIGH_LOCK_PENALTY_EXCLUSION_BPM": "挑战峰附近免受运动惩罚影响的频带宽度(bpm)。",
    "HIGH_LOCK_DOWN_STEP_BPM": "高锁定恢复时单窗口最大下降步长(bpm)。",
    "HIGH_LOCK_UP_STEP_BPM": "高锁定冷却期内单窗口最大反向上升步长(bpm)。",
    "POST_MOTION_GUARD_SECONDS": "旧固定 timeout 兼容值；None 表示动态保护不按时间硬退出。",
    "POST_MOTION_GUARD_MIN_ELAPSED_S": "运动后允许稳定交叉前的最短等待时间(秒)。",
    "POST_MOTION_GUARD_STABLE_WINDOWS": "运动后候选稳定交叉所需的连续窗口数。",
    "POST_MOTION_GUARD_CROSSOVER_GAP_BPM": "稳定交叉时两来源允许的最大差值(bpm)。",
    "POST_MOTION_GUARD_UPWARD_GAP_BPM": "运动后向上切换要求的最小优势差值(bpm)。",
    "POST_MOTION_GUARD_FFT_FLOOR_BPM": "reset-FFT 候选允许的最低心率(bpm)。",
    "POST_MOTION_GUARD_RECOVERY_STEP_UP_BPM": "运动后保护期每窗口最大上调步长(bpm)。",
    "POST_MOTION_GUARD_RECOVERY_STEP_DOWN_BPM": "运动后保护期每窗口最大下调步长(bpm)。",
    "POST_MOTION_GUARD_RISING_WINDOWS": "rising rescue 检查的连续 adaptive 窗口数。",
    "POST_MOTION_GUARD_RISING_SLOPE_BPM_PER_WINDOW": "rising rescue 要求的每窗口最低上升幅度(bpm)。",
    "POST_MOTION_GUARD_RESCUE_GAP_BPM": "触发大间隔 rescue 的来源差值阈值(bpm)。",
    "POST_MOTION_GUARD_GAP_RESCUE_ENABLE": "是否启用运动后大间隔多数命中救援。",
    "POST_MOTION_GUARD_GAP_RESCUE_WINDOWS": "gap rescue 统计使用的最近窗口数。",
    "POST_MOTION_GUARD_GAP_RESCUE_MIN_HITS": "gap rescue 在统计窗口内要求的最少命中数。",
    "POST_MOTION_GUARD_FFT_STABLE_WINDOWS": "reset-FFT 被视为稳定前要求的连续窗口数。",
    "POST_MOTION_GUARD_FFT_STABLE_BPM": "reset-FFT 跨窗口稳定容差(bpm)。",
    # Search-space rows.
    "SEARCH_SPACE": "Optuna 离散候选集合；下方逐项列表可直接增删候选值。",
    "SEARCH_SPACE.Fs_Target": "训练重采样率候选(Hz)；较低值更快，较高值时间分辨率更高。",
    "SEARCH_SPACE.TW": "训练窗口长度候选(秒)；影响频率分辨率、窗口数和响应速度。",
    "SEARCH_SPACE.Kstop": "级联停止阈值候选；控制继续增加参考级的条件。",
    "SEARCH_SPACE.max_order": "每级最大 tap 阶数候选；越大建模能力和计算量越高。",
    "SEARCH_SPACE.M_base": "级联基础 tap 数候选。",
    "SEARCH_SPACE.C_scale": "级联阶数增长比例候选。",
    "SEARCH_SPACE.K_max": "级联 tap/阶数上限候选。",
    "SEARCH_SPACE.Spec_Penalty_Width": "运动频谱惩罚带宽候选(Hz)。",
    "SEARCH_SPACE.hr_range_hz": "相邻 HR 搜索范围候选(Hz)，列表由 bpm 换算。",
    "SEARCH_SPACE.slew_limit_bpm": "HR 轨迹直接接受变化阈值候选(bpm)。",
    "SEARCH_SPACE.slew_step_bpm": "超限时 HR 每窗口最大修正步长候选(bpm)。",
    "SEARCH_SPACE.Rest_HR_Track_Band_BPM": "静息 HR 搜索带宽候选(bpm)。",
    "SEARCH_SPACE.Rest_HR_Slew_Limit_BPM": "静息 HR 直接接受变化阈值候选(bpm)。",
    "SEARCH_SPACE.Rest_HR_Slew_Step_BPM": "静息 HR 超限修正步长候选(bpm)。",
    "SEARCH_SPACE.LMS_Mu_Base": "LMS/Volterra 基础步长候选。",
    "SEARCH_SPACE.RFF_LMS_Mu_Base": "RFF-LMS 特征空间更新步长候选。",
    "SEARCH_SPACE.alpha_u": "Volterra 非线性项混合系数候选。",
    "SEARCH_SPACE.M2": "Volterra 二阶记忆长度候选。",
    "SEARCH_SPACE.rff_D": "RFF 随机特征维数候选；越大越慢且占用更多内存。",
    "SEARCH_SPACE.rff_sigma_scale": "基于 tap robust 距离的 RFF sigma 缩放候选。",
    "SEARCH_SPACE.klms_step_size": "KLMS 核权重更新步长候选。",
    "SEARCH_SPACE.klms_sigma": "KLMS RBF 核宽度候选。",
    "SEARCH_SPACE.klms_epsilon": "KLMS 新中心新颖度阈值候选。",
    # Execution, cache, split and safety switches.
    "NUM_SEED_POINTS": "TPE 启动前的随机探索 trial 数；不得超过总 trial 预算。",
    "N_JOBS": "并行 worker 数；1 最稳定，增大时注意内存和可复现性。",
    "PARALLEL_REPEATS": "重复实验并行数；当前实际并行度由 n_jobs 优先控制。",
    "TRIAL_CACHE_MAX_ENTRIES": "trial 结果内存缓存上限；越大越占内存。",
    "SAVE_STAGE_JSON": "是否保存详细 stage JSON；调试有用但会增加磁盘文件。",
    "DEBUG_MODE": "调试模式；会保留更多中间记录并隐式启用 stage JSON。",
    "CLEAN_OUTPUTS": "是否清空当前运行目录；续跑 checkpoint 时必须保持 False。",
    "VAL_GROUPS_PER_TYPE": "split 模式每个运动类型的验证组数；all_train 下不使用。",
    "TEST_GROUPS_PER_TYPE": "split 模式每个运动类型的测试组数；all_train 下不使用。",
    "CASCADE_TRAIN_BUDGETS": "按 scheme 覆盖 trial/repeat 预算；None 使用全局预算。",
    "TRIAL_PARAM_OVERRIDES": "固定 trial 参数映射；覆盖默认值并进入元数据、缓存键和断点指纹。",
    "RUN_STATIC_QC": "是否在第 1 节读取各文件前 10 秒并显示静态 QC。",
    "RUN_PREPROCESS_PREVIEW": "是否在第 2 节完整预处理一个所选样本。",
    "RUN_ALIGNMENT_DIAGNOSTICS": "是否运行第 3 节并写静息 Tdelay 对齐图。",
    "RUN_FULLFIELD_PLOT": "是否运行第 4 节并写全段未对齐 PPG-HR 图。",
    "RUN_RAW_PPG_PLOT": "是否运行第 6 节并写原始 PPG 双轴图。",
    "RUN_ALL_TRAIN": "是否运行第 7 节正式 200-trial 模式级优化。",
    "RUN_CONNECTIVITY_CHECK": "是否运行第 8 节小预算模式连通性检查。",
    "RUN_OUTPUT_CHECK": "是否运行第 9 节只读检查 Stage-6 和断点文件。",
    "RUN_STAGE7_REPLAY": "是否运行第 10 节单样本 best-params 回放。",
    "RUN_WINDOW_DIAGNOSTICS": "是否运行第 11 节窗口波形/频谱诊断。",
    "RUN_CROSS_MOTION_SUMMARY": "是否运行第 12 节跨运动汇总。",
    "RUN_CROSS_STAGE7_REPLAY": "是否运行第 13 节跨 scheme 回放。",
    "RUN_CROSS_WINDOW_DIAGNOSTICS": "是否运行第 14 节跨 scheme 窗口诊断。",
    "RUN_BATCH_REFERENCE_COMPARE": "是否运行第 15 节批量参考 scheme 对比。",
    "CONNECTIVITY_MAX_ITERATIONS": "连通性检查每模式 trial 数；默认 1，避免误跑正式预算。",
}


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
TW_F = 0.0
TRAIN_HR_POSTPROCESS_METHOD = "fft"  # 可选 fft/ssr，并进入运行目录名

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
    tw_f_s=TW_F,
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
TW = 8
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
        config_code(
            r'''
# 用途：配置 HR 后处理、四种自适应滤波器、级联保护和增强追踪全部固定阈值。
# 输入：固定值覆盖 Optuna 解码后的同名字段；搜索参数范围由 SEARCH_SPACE 控制。
# 输出：滤波与追踪配置变量。
# 是否写文件：否。
# 耗时风险：无；启用 SSR、Volterra、RFF-LMS 或 KLMS 会增加后续训练耗时。
REDRAW_HR_POSTPROCESS_METHOD = None   # None 表示按训练记录恢复
SSR_NUM_ATOMS = 5
SSR_LAMBDA = 0.15
SSR_HARMONIC_TOL_BPM = 5.0
SSR_FALLBACK_TO_FFT = True
SSR_GRID_RESOLUTION_BPM = 1.0

LMS_MU_MIN = 1e-6
TRACKING_SMOOTH_WIN_LEN = 7
SPEC_PENALTY_WEIGHT = 0.4
RFF_SIGMA = 1.0  # 仅用于兼容缺少 sigma_scale 的旧记录
RFF_UPDATE_MODE = "nlms"
RFF_NLMS_EPS = 1e-9
RFF_LEAKAGE = 0.0
RFF_ERR_CLIP = None
RFF_THETA_NORM_GUARD = None
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

TRACKING_RANGE_UP_BPM = None
TRACKING_RANGE_DOWN_BPM = None
TRACKING_SLEW_LIMIT_UP_BPM = None
TRACKING_SLEW_STEP_UP_BPM = None
TRACKING_SLEW_LIMIT_DOWN_BPM = None
TRACKING_SLEW_STEP_DOWN_BPM = None
MOTION_TRACKING_RANGE_UP_BPM = 35.0
MOTION_TRACKING_RANGE_DOWN_BPM = 15.0
MOTION_TRACKING_SLEW_LIMIT_UP_BPM = 5.5
MOTION_TRACKING_SLEW_STEP_UP_BPM = 3.5
MOTION_TRACKING_SLEW_LIMIT_DOWN_BPM = 2.0
MOTION_TRACKING_SLEW_STEP_DOWN_BPM = 1.5
RECOVERY_TRACKING_RANGE_UP_BPM = 20.0
RECOVERY_TRACKING_RANGE_DOWN_BPM = 25.0
RECOVERY_TRACKING_SLEW_LIMIT_UP_BPM = 1.5
RECOVERY_TRACKING_SLEW_STEP_UP_BPM = 1.5
RECOVERY_TRACKING_SLEW_LIMIT_DOWN_BPM = 3.5
RECOVERY_TRACKING_SLEW_STEP_DOWN_BPM = 3.0
CANDIDATE_PEAK_THRESHOLD_RATIO = 0.30
FULL_CANDIDATE_PEAK_THRESHOLD_RATIO = 0.15
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
POST_MOTION_GUARD_SECONDS = None
POST_MOTION_GUARD_MIN_ELAPSED_S = 5.0
POST_MOTION_GUARD_STABLE_WINDOWS = 3
POST_MOTION_GUARD_CROSSOVER_GAP_BPM = 2.0
POST_MOTION_GUARD_UPWARD_GAP_BPM = 1.5
POST_MOTION_GUARD_FFT_FLOOR_BPM = 55.0
POST_MOTION_GUARD_RECOVERY_STEP_UP_BPM = 1.5
POST_MOTION_GUARD_RECOVERY_STEP_DOWN_BPM = 3.0
POST_MOTION_GUARD_RISING_WINDOWS = 3
POST_MOTION_GUARD_RISING_SLOPE_BPM_PER_WINDOW = 1.5
POST_MOTION_GUARD_RESCUE_GAP_BPM = 20.0
POST_MOTION_GUARD_GAP_RESCUE_ENABLE = True
POST_MOTION_GUARD_GAP_RESCUE_WINDOWS = 4
POST_MOTION_GUARD_GAP_RESCUE_MIN_HITS = 3
POST_MOTION_GUARD_FFT_STABLE_WINDOWS = 3
POST_MOTION_GUARD_FFT_STABLE_BPM = 6.0

# Optuna 离散搜索空间。需要收缩或扩展候选值时直接修改对应列表。
SEARCH_SPACE = default_protocol_search_space()
SEARCH_SPACE.Fs_Target = [25, 50]
SEARCH_SPACE.TW = [6, 8, 10]
SEARCH_SPACE.Kstop = [0.2, 0.3, 0.5]
SEARCH_SPACE.max_order = [8, 12, 16, 20]
SEARCH_SPACE.M_base = [1, 2]
SEARCH_SPACE.C_scale = [0.6, 0.9, 1.2, 1.5]
SEARCH_SPACE.K_max = [8, 12, 16, 20, 30]
SEARCH_SPACE.Spec_Penalty_Width = [0.1, 0.2, 0.3]
SEARCH_SPACE.hr_range_hz = [value / 60.0 for value in (20, 25, 30, 35, 40)]
SEARCH_SPACE.slew_limit_bpm = [8, 10, 12, 14]
SEARCH_SPACE.slew_step_bpm = [5, 7, 9]
SEARCH_SPACE.Rest_HR_Track_Band_BPM = [20.0, 30.0, 60.0, 80.0]
SEARCH_SPACE.Rest_HR_Slew_Limit_BPM = [1.0, 3.0, 6.0, 8.0]
SEARCH_SPACE.Rest_HR_Slew_Step_BPM = [0.5, 2.0, 4.0]
SEARCH_SPACE.LMS_Mu_Base = [0.004, 0.006, 0.008]
SEARCH_SPACE.RFF_LMS_Mu_Base = [0.001, 0.002, 0.004, 0.006]
SEARCH_SPACE.alpha_u = [0.005, 0.01, 0.03, 0.05, 0.1]
SEARCH_SPACE.M2 = [2, 3]
SEARCH_SPACE.rff_D = [50, 100, 200]
SEARCH_SPACE.rff_sigma_scale = [0.5, 1.0, 2.0, 4.0]
SEARCH_SPACE.klms_step_size = [0.005, 0.01, 0.02, 0.05]
SEARCH_SPACE.klms_sigma = [0.5, 1.0, 2.0, 5.0]
SEARCH_SPACE.klms_epsilon = [0.005, 0.01, 0.02, 0.05, 0.1]
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
    "TW_F": TW_F,
    "normalization_mode": NORMALIZATION_MODE,
    "smooth_win_len": TRACKING_SMOOTH_WIN_LEN,
    "Spec_Penalty_Weight": SPEC_PENALTY_WEIGHT,
    "postprocess_method": TRAIN_HR_POSTPROCESS_METHOD,
    "SSR_Num_Atoms": SSR_NUM_ATOMS,
    "SSR_Lambda": SSR_LAMBDA,
    "SSR_Harmonic_Tol_BPM": SSR_HARMONIC_TOL_BPM,
    "SSR_Fallback_To_FFT": SSR_FALLBACK_TO_FFT,
    "SSR_Grid_Resolution_BPM": SSR_GRID_RESOLUTION_BPM,
    "LMS_Mu_Min": LMS_MU_MIN,
    "qc_policy": QC_POLICY,
    "delay_estimation_mode": DELAY_ESTIMATION_MODE,
    "Alignment_TW": ALIGNMENT_TW,
    "Alignment_Step": ALIGNMENT_STEP_S,
    "Rest_HR_Band_BPM": REST_HR_BAND_BPM,
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
    "rff_sigma": RFF_SIGMA,
    "rff_update_mode": RFF_UPDATE_MODE,
    "rff_nlms_eps": RFF_NLMS_EPS,
    "rff_leakage": RFF_LEAKAGE,
    "rff_err_clip": RFF_ERR_CLIP,
    "rff_theta_norm_guard": RFF_THETA_NORM_GUARD,
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
    "motion_tracking_range_up_bpm": MOTION_TRACKING_RANGE_UP_BPM,
    "motion_tracking_range_down_bpm": MOTION_TRACKING_RANGE_DOWN_BPM,
    "motion_tracking_slew_limit_up_bpm": MOTION_TRACKING_SLEW_LIMIT_UP_BPM,
    "motion_tracking_slew_step_up_bpm": MOTION_TRACKING_SLEW_STEP_UP_BPM,
    "motion_tracking_slew_limit_down_bpm": MOTION_TRACKING_SLEW_LIMIT_DOWN_BPM,
    "motion_tracking_slew_step_down_bpm": MOTION_TRACKING_SLEW_STEP_DOWN_BPM,
    "recovery_tracking_range_up_bpm": RECOVERY_TRACKING_RANGE_UP_BPM,
    "recovery_tracking_range_down_bpm": RECOVERY_TRACKING_RANGE_DOWN_BPM,
    "recovery_tracking_slew_limit_up_bpm": RECOVERY_TRACKING_SLEW_LIMIT_UP_BPM,
    "recovery_tracking_slew_step_up_bpm": RECOVERY_TRACKING_SLEW_STEP_UP_BPM,
    "recovery_tracking_slew_limit_down_bpm": RECOVERY_TRACKING_SLEW_LIMIT_DOWN_BPM,
    "recovery_tracking_slew_step_down_bpm": RECOVERY_TRACKING_SLEW_STEP_DOWN_BPM,
    "candidate_peak_threshold_ratio": CANDIDATE_PEAK_THRESHOLD_RATIO,
    "full_candidate_peak_threshold_ratio": FULL_CANDIDATE_PEAK_THRESHOLD_RATIO,
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
    "post_motion_guard_rising_windows": POST_MOTION_GUARD_RISING_WINDOWS,
    "post_motion_guard_rising_slope_bpm_per_window": POST_MOTION_GUARD_RISING_SLOPE_BPM_PER_WINDOW,
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
RUN_STAGE7_REPLAY = False
RUN_WINDOW_DIAGNOSTICS = False
RUN_CROSS_MOTION_SUMMARY = False
RUN_CROSS_STAGE7_REPLAY = False
RUN_CROSS_WINDOW_DIAGNOSTICS = False
RUN_BATCH_REFERENCE_COMPARE = False
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
        """通过 motion/index 选择一个样本，检查 CF2、HF2、HF2comp、UD2、ACC、QC 和运动分段。

本节涉及的参数和可配置项如下：

| 参数 | 含义 | 可配置值与注意事项 |
| --- | --- | --- |
| `RUN_PREPROCESS_PREVIEW` | 本节总开关 | `True` 时读取并预处理一个配对；`False` 时只打印跳过信息。它不会创建 Optuna study，也不会写结果文件。 |
| `PREVIEW_MOTION_TYPE` | 要预览的运动类型 | 只能选择 `write/gripper/run/rope` 之一，并且必须在当前受试者目录中存在。 |
| `PREVIEW_MOTION_INDEX` | 同一运动类型下的测试序号 | 使用正整数，例如 `1`。motion 与 index 必须唯一匹配一个严格配对。 |
| `SUBJECT_DIR` | 当前唯一受试者目录 | 在第 0 节或环境变量 `PPG_SUBJECT_DIR` 中设置；本节不会扫描其他受试者。 |
| `calibration` | 当前受试者的标定系数 | 不是可搜索超参数；固定由第 1 节从 `SUBJECT_DIR.parent / "ck.mat"` 解析并校验。 |
| `FS_ORIGIN` | 原始传感器采样率 | change6 数据协议固定为 100 Hz；通常保持 `100`，修改后必须与采集序号和真实采样率一致。 |
| `TW` | 本节运动分段的窗口长度，单位秒 | 预览默认 `8` 秒。这里只影响预览分段；正式训练中的 trial 窗长由 `SEARCH_SPACE.TW` 候选列表采样。 |

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
# TW: 仅用于本节 detect_activity_segments 的分段窗口；正式训练由 SEARCH_SPACE.TW 采样。
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
REPLAY_CASCADE_SCHEME = CascadeScheme.ACC.value
REPLAY_ADAPTIVE_DATA_TYPE = ""
REPLAY_TW_F = TW_F
REPLAY_SPLIT = ""
REPLAY_MODE = ""
REPLAY_GUARD_RATIO_MIN_OVERRIDE = None
REPLAY_GUARD_RATIO_MAX_OVERRIDE = None
REPLAY_GUARD_FLAT_STD_EPS_OVERRIDE = None
REPLAY_GUARD_USE_FINITE_ZSCORE_OVERRIDE = None
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
        cascade_scheme=REPLAY_CASCADE_SCHEME,
        TW_F=REPLAY_TW_F,
        fs_origin=FS_ORIGIN,
        guard_ratio_min_override=REPLAY_GUARD_RATIO_MIN_OVERRIDE,
        guard_ratio_max_override=REPLAY_GUARD_RATIO_MAX_OVERRIDE,
        guard_flat_std_eps_override=REPLAY_GUARD_FLAT_STD_EPS_OVERRIDE,
        guard_use_finite_zscore_override=REPLAY_GUARD_USE_FINITE_ZSCORE_OVERRIDE,
        postprocess_method_override=REDRAW_HR_POSTPROCESS_METHOD,
    )
    print(replay_paths)
    display(Image(filename=str(replay_paths["plot_guarded"])))
else:
    print("RUN_STAGE7_REPLAY=False，跳过单样本回放。")
''',
    )
    cells += section(
        11,
        "对单个样本输出窗口级波形与频谱诊断",
        "复用第 10 节的样本和参数选择器，查看指定对齐窗口的输入、频谱、权重和追踪状态。",
        r'''
# 用途：从 Stage-6 参数记录重建一个窗口的完整诊断图和状态摘要。
# 输入：第 10 节 REPLAY_* 配置以及 MANUAL_ALIGNED_FFT_START_S。
# 输出：RESULTS_ROOT/window_diagnostics 下的图表和诊断信息。
# 是否写文件：是，仅在 RUN_WINDOW_DIAGNOSTICS=True 时。
# 耗时风险：中；执行单样本回放和单窗口诊断，不启动 Optuna。
DIAGNOSTIC_OUTPUT_DIR = Path(RESULTS_ROOT) / "window_diagnostics"
MANUAL_ALIGNED_FFT_START_S = 80.0
window_diagnostics = {}
if RUN_WINDOW_DIAGNOSTICS:
    diagnostic_pair = select_subject_pair(REPLAY_MOTION_TYPE, REPLAY_MOTION_INDEX)
    window_diagnostics = plot_window_diagnostics_from_records(
        signal_csv=diagnostic_pair.sensor_csv,
        ref_csv=diagnostic_pair.ref_csv,
        results_root=RESULTS_ROOT,
        output_dir=DIAGNOSTIC_OUTPUT_DIR,
        aligned_fft_start_s=MANUAL_ALIGNED_FFT_START_S,
        motion_type=REPLAY_MOTION_TYPE,
        split=REPLAY_SPLIT,
        mode=REPLAY_MODE,
        target_scope=REPLAY_TARGET_SCOPE,
        adaptive_filter=REPLAY_ADAPTIVE_FILTER,
        adaptive_data_type=REPLAY_ADAPTIVE_DATA_TYPE,
        cascade_scheme=REPLAY_CASCADE_SCHEME,
        TW_F=REPLAY_TW_F,
        fs_origin=FS_ORIGIN,
        guard_ratio_min_override=REPLAY_GUARD_RATIO_MIN_OVERRIDE,
        guard_ratio_max_override=REPLAY_GUARD_RATIO_MAX_OVERRIDE,
        guard_flat_std_eps_override=REPLAY_GUARD_FLAT_STD_EPS_OVERRIDE,
        guard_use_finite_zscore_override=REPLAY_GUARD_USE_FINITE_ZSCORE_OVERRIDE,
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
SUMMARY_CASCADE_SCHEME = CascadeScheme.ACC.value
SUMMARY_ADAPTIVE_DATA_TYPE = ""
SUMMARY_TARGET_SCOPE = TargetScope.MOTION_POST10.value
SUMMARY_TW_F = TW_F
SUMMARY_TABLE_OUTPUT_DIR = Path(RESULTS_ROOT) / "summary_tables"
summary_path = None
if RUN_CROSS_MOTION_SUMMARY:
    summary_path = build_cross_motion_summary_table(
        table_output_dir=SUMMARY_TABLE_OUTPUT_DIR,
        results_root=RESULTS_ROOT,
        adaptive_filter=SUMMARY_ADAPTIVE_FILTER,
        adaptive_data_type=SUMMARY_ADAPTIVE_DATA_TYPE,
        cascade_scheme=SUMMARY_CASCADE_SCHEME,
        target_scope=SUMMARY_TARGET_SCOPE,
        TW_F=SUMMARY_TW_F,
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
CROSS_PARAM_SCHEME = CascadeScheme.HF2.value
CROSS_APPLIED_SCHEME = CascadeScheme.ACC.value
CROSS_TW_F = TW_F
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
        cascade_scheme=CROSS_PARAM_SCHEME,
        override_cascade_scheme=CROSS_APPLIED_SCHEME,
        TW_F=CROSS_TW_F,
        fs_origin=FS_ORIGIN,
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
# 输入：第 13 节 CROSS_* 配置和 CROSS_ALIGNED_FFT_START_S。
# 输出：RESULTS_ROOT/cross_window_diagnostics 下的诊断图和摘要。
# 是否写文件：是，仅在 RUN_CROSS_WINDOW_DIAGNOSTICS=True 时。
# 耗时风险：中；执行单样本、单窗口诊断，不启动 Optuna。
CROSS_DIAGNOSTIC_OUTPUT_DIR = Path(RESULTS_ROOT) / "cross_window_diagnostics"
CROSS_ALIGNED_FFT_START_S = 80.0
cross_window_diagnostics = {}
if RUN_CROSS_WINDOW_DIAGNOSTICS:
    cross_diagnostic_pair = select_subject_pair(CROSS_MOTION_TYPE, CROSS_MOTION_INDEX)
    cross_window_diagnostics = plot_window_diagnostics_from_records(
        signal_csv=cross_diagnostic_pair.sensor_csv,
        ref_csv=cross_diagnostic_pair.ref_csv,
        results_root=RESULTS_ROOT,
        output_dir=CROSS_DIAGNOSTIC_OUTPUT_DIR,
        aligned_fft_start_s=CROSS_ALIGNED_FFT_START_S,
        motion_type=CROSS_MOTION_TYPE,
        target_scope=CROSS_TARGET_SCOPE,
        adaptive_filter=CROSS_ADAPTIVE_FILTER,
        cascade_scheme=CROSS_PARAM_SCHEME,
        override_cascade_scheme=CROSS_APPLIED_SCHEME,
        TW_F=CROSS_TW_F,
        fs_origin=FS_ORIGIN,
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
REFERENCE_COMPARE_PARAM_SCHEME = CascadeScheme.HF2.value
REFERENCE_COMPARE_APPLIED_SCHEME = CascadeScheme.ACC.value
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
        best_param_source_cascade_scheme=REFERENCE_COMPARE_PARAM_SCHEME,
        actual_reference_cascade_scheme=REFERENCE_COMPARE_APPLIED_SCHEME,
        target_scope=REFERENCE_COMPARE_TARGET_SCOPE,
        adaptive_filter=REFERENCE_COMPARE_ADAPTIVE_FILTER,
        best_param_csv_path=REFERENCE_COMPARE_BEST_PARAMS_CSV,
        output_dir=REFERENCE_COMPARE_OUTPUT_DIR,
        fs_origin=FS_ORIGIN,
    )
    print(reference_compare_paths)
    display(pd.read_csv(reference_compare_paths["csv"]))
else:
    print("RUN_BATCH_REFERENCE_COMPARE=False，跳过批量参考通道对比。")
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
