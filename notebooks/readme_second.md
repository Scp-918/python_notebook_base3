# 第二阶段批量自适应滤波协议说明

当前工程根目录统一为：

```text
D:\python_notebook_base
```

Notebook：

```text
D:\python_notebook_base\notebooks\run_batch_adaptive_protocol.ipynb
```

核心路径：

```python
PROJECT_ROOT = Path(r"D:\python_notebook_base")
SRC_DIR = PROJECT_ROOT / "src"
TESTDATA_DIR = PROJECT_ROOT / "testdata"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
```

## 数据命名与运动类型

传感器文件和参考 HR 文件必须成对出现：

```text
multi_<运动类型拼音+数字编号>.csv
multi_<运动类型拼音+数字编号>_ref.csv
```

当前合法运动类型：

```text
tiaosheng, wanju, fuwo, kaihe, bobi
```

示例：

```text
multi_kaihe1.csv
motion_type = "kaihe"
motion_index = 1
motion_id/group_id = "kaihe1"
```

未配对 CSV 会写入 `unpaired_samples.csv`，不会进入 QC、预处理或训练。

## QC 规则

QC 只检查 Ut1/Ut2 前 10 秒，采样率按 100 Hz：

- 对 Ut1/Ut2 分别做 4 阶多项式基线拟合；
- 原始信号减去基线得到高频残差；
- 任一路残差 STD > 2.5 mV，判坏；
- 两路 STD 比例 > 3，判坏；
- 大于 3 倍 STD 的离群点比例若一者大于另一者 3 倍以上，且两路离群点比例不都同时小于 3%，判坏。

`bad_samples.csv` 记录：

```text
group_id, motion_type, data_file, ref_file, reason,
std_ut1, std_ut2,
outlier_count_ut1, outlier_count_ut2,
outlier_ratio_ut1, outlier_ratio_ut2
```

## 预处理

`preprocess_protocol.py` 的协议流程：

- 时间轴按 `fs_origin` 从 0 重建；
- 缺失值线性/近邻插值；
- PPG 毛刺处理；
- `CF1 = Uc1 / (Ut1 - Uc1)`；
- `CF2 = Uc2 / (Ut2 - Uc2)`；
- 输出 13 路协议信号：

```text
ppg_green, ppg_red, ppg_ir,
hf1, hf2,
cf1, cf2,
accx, accy, accz,
gyrox, gyroy, gyroz
```

分类型带通和重采样：

- PPG：0.5-5 Hz；
- HF：0.1-5 Hz；
- CF：0.1-5 Hz；
- ACC：0.5-10 Hz；
- Gyro：0.5-10 Hz；
- 使用 `filtfilt` 零相位滤波；
- 使用 `resample_poly` 重采样到 `Fs_Target`。

主流程会在 `outputs/<run_name>/signal_figures` 下输出每组运动的完整原始/清洗后 13 路信号图，以及运动段 13 路带通信号图。

## 运动分段与 TargetScope

运动分段规则：

- 三轴 ACC 合模长；
- 前 30 秒为校准期；
- 阈值为前 30 秒合模长 STD 的 3 倍；
- `TW` 秒窗口，1 秒步长；
- 连续 6 个静息窗 -> 连续 6 个运动窗定位运动开始；
- 连续 6 个运动窗 -> 连续 6 个静息窗定位运动结束。

支持 3 个 TargetScope：

```text
motion_only      = 只包含 motion
motion_recovery  = motion + recovery
motion_post10    = motion + 运动结束后 10 秒；不足 10 秒则到文件末尾
```

## PPG 与参考 HR 对齐

对齐搜索：

全局 Tdelay 使用静息段绿光 PPG 与参考 HR 搜索。
当前实现已将全局对齐窗口与后续训练窗口解耦：

- Alignment_TW：只用于静息段 PPG-HR 提取和全局 Tdelay 搜索，默认 8 秒，不参与贝叶斯优化。
- TW：只用于后续自适应滤波训练/验证/测试切窗，可继续参与贝叶斯优化。

因此同一 Fs_Target、同一 Alignment_TW 下，TW = 6、8、10 秒会复用同一个全局 Tdelay；
后续训练窗口数量、window_centers_s 与参考 HR 半窗校正仍随各自的 TW 改变。

delay_s > 0：
    认为传感器侧信号滞后参考 HR；
    所有传感器通道统一左移；
    实现方式是裁掉前 delay_s 对应的样本数。

delay_s < 0：
    认为传感器侧信号提前参考 HR；
    所有传感器通道统一右移；
    实现方式是在每个通道前端补该通道首样本值 arr[0]，尾部截断相同样本数；
    不补数字 0。

参考 HR：
    心率带参考 HR 为 1 Hz；
    不进行 0.1 秒级前移；
    不通过高频插值实现负 Tdelay；
    仍保留 TW/2 半窗补偿。

`alignment_info` 保存：

```text
std_by_delay, best_tdelay_s, ref_shift_s, num_windows
```

如果静息段太短、可比较窗口少于 2 个，流程不会崩溃；对应样本/模式会在 summary 中记录失败原因。

## 自适应滤波器

当前 adaptive_filter 只允许：

```text
lms
volterra
rff_lms
```

不再使用其他旧的非线性滤波器命名或搜索项。

LMS 步长统一使用：

```text
mu = max(mu_min, LMS_Mu_Base - abs_corr / 100)
mu_min = 1e-6
```

Volterra：

- 线性项使用完整 `M + K` 非因果向量；
- 二阶项只取最近 `M2` 个 tap 的上三角组合；
- `mu2 = alpha_u * mu1`。

RFF-LMS：

- 固定随机特征；
- `rff_seed` 由 trial 参数、mode_key、repeat_idx 和 random_state 稳定哈希生成；
- `rff_seed` 会写入 trial params、trial_history、JSON 和 cache key。

## 搜索空间

公共搜索项：

```text
Fs_Target: [25, 50, 100]
TW: [6, 8, 10]
Kstop: [0.2, 0.3, 0.5]
max_order: [8, 12, 16, 20]
M_base: [1, 2]
C_scale: [0.6, 0.9, 1.2, 1.5]
K_max: [8, 12, 16, 20, 30]
Spec_Penalty_Width: [0.1, 0.2, 0.3]
Spec_Penalty_Weight: [0.1, 0.2, 0.4]
smooth_win_len: [3, 5, 7, 9]
hr_range_hz: [15/60, 20/60, 25/60, 30/60, 35/60, 40/60]
slew_limit_bpm: [8, 9, 10, 11, 12, 13, 14, 15]
slew_step_bpm: [5, 7, 9]
```

LMS：

```text
LMS_Mu_Base: [0.008, 0.01, 0.012]
```

Volterra：

```text
LMS_Mu_Base: [0.008, 0.01, 0.012]
alpha_u: [0.01, 0.05, 0.1, 0.2]
M2: [2, 3, 4, 5]
```

RFF-LMS：

```text
LMS_Mu_Base: [0.006, 0.008, 0.01]
rff_D: [50, 100, 200, 300]
rff_sigma: [0.1, 0.5, 1.0, 2.0, 5.0]
```

## 优化目标

`OPTIMIZATION_OBJECTIVE` 可选：

```text
aae
accuracy
```

`aae`：最小化 adaptive AAE。

`accuracy`：最小化 `100 - adaptive_acc_pct`。

accuracy 定义为目标段内 `abs_err <= 5.0 bpm` 的窗口比例 × 100。

训练进度会同时打印当前 AAE、accuracy、objective_mode 和 objective_value。

## 参考信号来源

支持 7 类 CascadeScheme：

```text
ACC3
HF2
CF2
HF2_CF2
CF2_HF2
ACC3_HF2
HF2_ACC3
```

Notebook 的 `ACTIVE_CASCADE_SCHEMES` 支持多选；`CASCADE_TRAIN_BUDGETS` 支持为每个参考源独立设置 `n_trials` 和 `n_repeats`。

## 按运动类型组织训练

训练按 `motion_type` 分组进行。这里的“train”不是传统模型拟合；每个 trial 只是评估一组超参数，自适应滤波器权重在窗口内在线更新，不跨文件保存。

### split 模式

`DATA_SPLIT_MODE = "split"`：

- 同一 motion_type 内固定随机种子划分；
- test 集默认 1 个文件；
- validation 集默认 1 个文件；
- train 集为剩余文件；
- 三者不重叠；
- 若 `good_count - m - k < 1`，训练单元格会停止并打印明确错误。

每个 trial 参数会在 train/val/test 上完整运行 HR 估计；objective 只使用 validation 集指定训练段总指标。best params 确定后记录 test 效果。

### all_train 模式

`DATA_SPLIT_MODE = "all_train"`：

- 同一 motion_type 所有好样本合并作为 train；
- 同时也作为 test；
- objective 使用 train 集指定训练段总指标。

## 输出目录

每个训练单元格输出到：

```text
outputs/<run_name>/
```

`run_name` 自动由以下字段组成：

```text
target scopes
cascade schemes
adaptive filters
objective mode
data split mode
```

示例：

```text
outputs/motion_only-motion_recovery__ACC3-HF2-CF2-HF2_CF2-HF2_ACC3__lms-volterra-rff_lms__accuracy__split/
```

主流程输出：

```text
qc/
qc/motion_type_samples.csv
signal_figures/
motion_types/<motion_type>/split_files.csv
motion_types/<motion_type>/mode_summary_aae.csv
motion_types/<motion_type>/mode_summary_accuracy.csv
motion_types/<motion_type>/per_group_aae.csv
motion_types/<motion_type>/per_group_accuracy.csv
motion_types/<motion_type>/best_params_lms.csv
motion_types/<motion_type>/best_params_volterra.csv
motion_types/<motion_type>/best_params_rff_lms.csv
motion_types/<motion_type>/best_params_all.json
motion_types/<motion_type>/bayes_curve_data.csv
motion_types/<motion_type>/bayes_curve.png
final_summary/
batch_summary.csv
```

主训练流程不输出 HR 曲线图。HR 曲线只在 Notebook 末尾“手动重画最佳参数”单元格输出。

所有运动类型训练完成后，`final_summary/` 固定输出 6 个 CSV：

```text
motion_only_test_aae.csv
motion_only_test_accuracy.csv
motion_recovery_test_aae.csv
motion_recovery_test_accuracy.csv
motion_post10_test_aae.csv
motion_post10_test_accuracy.csv
```

若某个 TargetScope 未激活，对应 CSV 仍会创建为空表并带表头。

## 贝叶斯训练曲线

每个 motion_type 输出一张 `bayes_curve.png` 和一个 `bayes_curve_data.csv`。

子图数量动态等于：

```text
激活 target scopes × 激活 cascade schemes × 激活 adaptive filters
```

`bayes_curve_data.csv` 包含：

```text
motion_type, target_scope, cascade_scheme, adaptive_filter,
repeat_idx, trial_idx, objective_value, aae_bpm, accuracy_pct,
best_so_far, success, reason
```

## 手动重画最佳参数

Notebook 末尾手动输入：

```text
SENSOR_CSV_PATH
REF_CSV_PATH
TARGET_SCOPE
CASCADE_SCHEME
ADAPTIVE_FILTER
BEST_PARAM_CSV_PATH
OUTPUT_DIR
```

功能：

- 读取单个原始运动 CSV 和参考 HR CSV；
- 按 `BEST_PARAM_CSV_PATH` 中的单一模式参数重新运行预处理、分段、对齐、Fmove、baseline HR 和 adaptive HR；
- 输出训练段 HR 曲线；
- 输出全局 HR 曲线，其中非训练段使用 Hamming + FFT 在 0.5-2 Hz 主频提取 HR。

训练段图：

- 真实 HR：黑色实线；
- 未去伪影 PPG baseline：灰色虚线；
- adaptive HR：蓝色实线；
- legend 标注 AAE 与 accuracy。

全局图：

- 全局真实 HR：黑色实线；
- 全局未去伪影 PPG baseline：灰色虚线；
- 训练段 adaptive HR + 非训练段 Hamming FFT HR：蓝色实线；
- legend 标注全局 AAE 与 accuracy。

## 性能与缓存

主流程继续复用只依赖 `sample_id / Fs_Target / TW` 的缓存：

- 重采样；
- 分段；
- 对齐；
- Fmove。

trial 级缓存键包含：

```text
sample_id
Fs_Target
TW
target_scope
cascade_scheme
adaptive_filter
filter-specific params
rff_seed
```

默认 `n_jobs=1`，避免 Windows/Jupyter 下过度并行导致内存压力。大数组不会写入 JSON，trial 结束后只保留必要 CSV/JSON/PNG。

## 2026-05 有限修复后的协议与工程说明

### LOGO 输出语义

`data_split_mode="leave_one_group_out"` 时，每个 `motion_type` 内按 group 逐一留一：

- 每个 fold 的 `test` 是当前 held-out group；
- 每个 fold 的 `train` 是同一 `motion_type` 下除 held-out group 外的所有 group；
- 每个 fold 独立运行 Optuna 并得到自己的 `best_params`；
- LOGO aggregate 指标不是 fold AAE/accuracy 的简单平均，而是拼接所有 held-out test windows 的 `adaptive_hr_bpm` 与 `ref_hr_bpm` 后重新计算。

输出中新增以下字段用于区分语义：

- `result_level`: `fold`、`aggregate` 或 `single_split`；
- `aggregation`: LOGO aggregate 行为 `logo_window_concat`；
- `params_semantics`: fold 行为 `fold_best_params`，普通 split 行为 `best_params_for_this_split`，LOGO aggregate 行为 `representative_fold_best_params_not_global`；
- `representative_fold_id` / `representative_heldout_group_id`: LOGO aggregate 行中用于说明 `best_params` 仅来自哪个代表 fold，兼容旧输出，不表示 motion_type 级统一全局最优参数。

### test Tdelay 协议

当前 test 阶段仍允许使用该样本的参考 HR 估计 global Tdelay。该步骤被视为每个样本的 alignment/label 对齐预处理。本轮修改没有把 test Tdelay 改成训练集统计值，也不声明为部署式无参考泛化。

### 缓存生命周期

缓存分为两层：

- global alignment/global Tdelay cache：挂在 dataset 的 `_global_tdelay_cache`，key 包含 sample、`Fs_Target`、`Alignment_TW`、`Alignment_Step` 与静息 HR 提取参数，不包含 adaptive `TW`；
- TrialBase/heavy cache：挂在 `_trial_base_cache` 以及 TrialBase 内部的 `norm_window_cache`、`delay_estimate_cache`、`spectral_cache`，依赖 adaptive `TW`、窗口和 trial 过程。

LOGO 中每个 fold 后调用 `clear_trial_heavy_caches()`，只清重缓存并保留 `_global_tdelay_cache`，避免同一 group 在多个 fold 中重复估计 global Tdelay。每个 motion_type 结束后调用 `clear_all_caches()` 清理全部 dataset 级缓存，避免跨 motion_type 长期保留内存。旧的 `clear_trial_caches()` 保留为兼容入口，语义等同 full clear。

### delay search

窗口内 envelope/direct delay search 仍是逐 lag 的有效重叠区 Pearson correlation，不是 FFT cross-correlation。默认优先使用 numba JIT 加速；numba 不可用或 JIT 调用失败时自动 fallback 到 Python reference。delay sign convention 与原实现保持一致。

### 并行

`n_jobs > 1` 且 `num_repeats > 1` 时，从 Optuna repeat 级并行：每个 repeat 独立创建 sampler/study 和本地 bounded trial cache，完成后由主进程合并 history 与 best params。不做 window 级并行，因为窗口 HR tracking 依赖 `previous_hr` 顺序状态。并行进度以 repeat 粒度报告，字段包含 `parallel_level="repeat"`、`repeat_done`、`repeat_total`、`n_jobs` 和当前 best。

### stage JSON

正式 batch full eval 默认 `save_stage_json=False`，因此 `adaptive_stages_json` / `lms_stages_json` 默认保留为空字符串列，不生成每窗口每级详细 JSON。light eval 永不生成 stage JSON。需要 debug、redraw 或抽查时，可显式传 `save_stage_json=True`；`debug_mode=True` 也会开启。

### spectral_utils

Hamming window、rFFT 频率轴、FFT magnitude spectrum、dominant frequency 和参考式 peak candidates 已集中到 `src/ppg_hr/experimental/spectral_utils.py`。alignment 与 cascade 模块复用该工具，以减少重复实现并统一缓存 key。
