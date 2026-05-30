
# change5 批量自适应滤波协议说明

本文档说明 `python_notebook_base` 当前 `change5` 分支中，`notebooks/run_batch_adaptive_protocol.ipynb` 与 `src/ppg_hr/` 相关代码的使用方式、数据格式、处理流程、贝叶斯训练参数、输出结果和重绘/重输出方法。

本文面向第一次使用该工程的人，重点说明“应该改哪里、运行后会发生什么、输出在哪里看”。

---

## 0. change5 改造速览

当前 notebook 已同步三阶段改造，核心变化如下：

```text
阶段 1:
    收缩 Optuna 搜索空间，减少低价值组合。
    RFF-LMS 默认改为特征空间 NLMS；新训练搜索 `rff_sigma_scale`，并按当前 tap 窗口 robust 距离生成 `sigma_eff`。旧结果只有 `rff_sigma` 时仍按 fixed sigma 回放。KLMS 默认使用归一化距离、字典上限和冻结新中心策略。

阶段 2:
    新增 PPG 输入策略 ppg_input_transform。
    新增 global_objective_strategy，其中 deployment_global 用于模拟部署时 rest/motion/recovery 的 final HR 选择。
    新增 cascade_guard_policy，rms_guard 可在级联某级输出 RMS 异常时回退本级输出。
    global alignment cache key 已包含 PPG 输入变换相关字段。

阶段 3:
    训练输出统一增加 param_* 分列。
    replay 和 diagnostics 统一通过 protocol_params_from_record 读取参数，优先 param_*，再兼容 best_params_json / params JSON。HR 后处理默认 `postprocess_method="fft"`；可切换为 SSR 峰值提取分支，SSR 失败会按配置回退 FFT。
    窗口诊断图增加权重/诊断图，并在启用 guard 时区分完整级联输出和 guard 后实际使用输出。
```

对新手最重要的结论：

```text
想改实验策略:
    优先改 notebook 第 0 块的固定参数。

想启动训练:
    运行第 0/1/5/7 块。

想复现某个单文件曲线:
    运行第 9/10 块，参数会从 Stage-6 记录恢复。

想看某个窗口为什么选出这个 HR:
    运行第 11 块，看 waveform、spectrum、weights 和 stage 摘要。
```

## 1. 工程结构与运行说明

### 1.1 推荐工程根目录

Notebook 中建议把项目根目录设置为当前仓库根目录，例如：

```python
PROJECT_ROOT = Path(r"D:\python_notebook_base")
SRC_DIR = PROJECT_ROOT / "src"
TESTDATA_DIR = PROJECT_ROOT / "testdata"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
```

运行前需要保证 `src` 被加入 Python 路径：

```python
import sys
from pathlib import Path

PROJECT_ROOT = Path(r"D:\python_notebook_base")  # 请按本机实际仓库路径修改
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
```

`PROJECT_ROOT` 是运行 notebook 时的本机路径，只应作为本地配置使用；提交 notebook 时不要把临时实验路径写进版本库。

核心 Notebook：

```text
notebooks/run_batch_adaptive_protocol.ipynb
```

核心代码目录：

```text
src/ppg_hr/
```

主要子模块含义：

```text
src/ppg_hr/params.py
    协议枚举、默认参数、搜索空间定义。

src/ppg_hr/preprocess/data_loader.py
    兼容旧入口的数据读取工具，定义原始传感器 CSV 需要的字段名。

src/ppg_hr/experimental/batch_pairing.py
    扫描 input_dir，识别 multi_<运动类型><编号>.csv 与对应 _ref.csv。

src/ppg_hr/experimental/qc.py
    输入样本质量控制，只根据前 10 秒 Ut1/Ut2 判断样本是否进入训练。

src/ppg_hr/experimental/preprocess_protocol.py
    协议主预处理：时间轴、缺失值、PPG 毛刺、CF 计算、带通、重采样。

src/ppg_hr/experimental/segmentation.py
    基于三轴 ACC 的运动段检测。

src/ppg_hr/experimental/alignment.py
    静息段 PPG-HR 与参考 HR 的全局 Tdelay 搜索和对齐。

src/ppg_hr/experimental/cascade_solver.py
    核心 HR 估计、自适应滤波、窗口级评估与缓存。

src/ppg_hr/experimental/protocol_search_space.py
    Optuna trial 参数解码，生成 ProtocolTrialParams。

src/ppg_hr/experimental/run_batch_protocol.py
    Notebook 调用的总调度器，负责 QC、预处理、分组、训练、汇总、重画和重输出。

src/ppg_hr/experimental/protocol_outputs.py
    QC 表格、信号图、汇总表、诊断图等输出函数。
```

### 1.2 主入口函数

批量训练主入口是：

```python
from ppg_hr.experimental.run_batch_protocol import run_batch_adaptive_protocol
```

典型调用形式：

```python
result = run_batch_adaptive_protocol(
    input_dir=TESTDATA_DIR,
    output_root=RUN_OUTPUT_DIR,
    max_iterations=MAX_ITERATIONS,
    num_repeats=NUM_REPEATS,
    random_state=RANDOM_STATE,
    num_seed_points=NUM_SEED_POINTS,
    fs_origin=FS_ORIGIN,
    n_jobs=N_JOBS,
    target_scopes=ACTIVE_TARGET_SCOPES,
    cascade_schemes=ACTIVE_CASCADE_SCHEMES,
    adaptive_filters=ACTIVE_ADAPTIVE_FILTERS,
    objective_mode=OPTIMIZATION_OBJECTIVE,
    data_split_mode=DATA_SPLIT_MODE,
    cascade_train_budgets=CASCADE_TRAIN_BUDGETS,
    trial_param_overrides=TRIAL_PARAM_OVERRIDES,
    project_root=PROJECT_ROOT,
    clean_outputs=CLEAN_OUTPUTS,
)
```

### 1.3 输出目录保护

主流程支持 `clean_outputs=True`，但代码只允许清空：

```text
PROJECT_ROOT / outputs / <本次 run_name>
```

不会允许清空以下目录：

```text
PROJECT_ROOT
PROJECT_ROOT/testdata
PROJECT_ROOT/src
PROJECT_ROOT/notebooks
PROJECT_ROOT/outputs
```

因此，如果需要重跑同一个配置，可以清理本次 run 目录；不要手动删除 `src`、`notebooks` 或 `testdata`。

---

## 2. 数据读入格式说明

### 2.1 文件命名规则

传感器文件和参考 HR 文件必须成对出现。

传感器文件：

```text
multi_<运动类型拼音><编号>.csv
```

参考 HR 文件：

```text
multi_<运动类型拼音><编号>_ref.csv
```

示例：

```text
multi_kaihe1.csv
multi_kaihe1_ref.csv
```

该样本会被解析为：

```text
motion_type = "kaihe"
motion_index = 1
motion_id / group_id = "kaihe1"
```

### 2.2 当前合法运动类型

当前代码只接受以下 5 类运动：

```text
tiaosheng
wanju
fuwo
kaihe
bobi
```

如果文件名不是这些运动类型，或没有对应的 `_ref.csv`，不会进入 QC、预处理或训练，而是写入：

```text
outputs/<run_name>/qc/unpaired_samples.csv
```

### 2.3 传感器 CSV 必需字段

传感器 CSV 至少需要包含以下字段：

```text
Uc1(mV)
Uc2(mV)
Ut1(mV)
Ut2(mV)
PPG_Green
PPG_Red
PPG_IR
AccX(g)
AccY(g)
AccZ(g)
GyroX(dps)
GyroY(dps)
GyroZ(dps)
```

可选时间字段：

```text
Time(s)
```

如果存在 `Time(s)` 且有效值数量足够，代码会优先使用它；否则按采样率从 0 重建时间轴：

```python
time_s = np.arange(n) / fs_origin
```

可选 QC 字段：

```text
SampleIndex
Seq
ValidFlag
InterpFlag
GapLen
MissingBefore
```

如果旧 CSV 没有这些字段，代码会自动补默认值：

```text
ValidFlag = 1
InterpFlag = 0
GapLen = 0
MissingBefore = 0
SampleIndex = 0, 1, 2, ...
Seq = NaN
```

### 2.4 参考 HR CSV 格式

参考 HR CSV 支持常见两类格式：

第一类：带列名，例如：

```text
time_s,hr_bpm
0,72
1,73
2,74
```

第二类：类似 Polar 导出的格式，前几行是说明，后面某两列分别是时间和 HR。代码会尝试跳过前三行后读取。

要求：

```text
参考时间列能解析为秒或 HH:MM:SS
参考 HR 列能解析为 bpm 数值
至少有 2 个有效点
```

---

## 3. 数据处理流程说明

主流程不是简单地“读文件后训练”，而是按下面顺序执行。

### 3.1 样本配对

扫描 `input_dir` 下所有 CSV：

1. 找到合法传感器文件 `multi_<motion_type><index>.csv`。
2. 查找对应的 `multi_<motion_type><index>_ref.csv`。
3. 成功配对后生成 `SamplePair`。
4. 不合法或缺少配对的文件写入 `unpaired_samples.csv`。

### 3.2 QC：样本质量控制

QC 只检查传感器 CSV 的前 10 秒 `Ut1(mV)` 和 `Ut2(mV)`。

默认采样率：

```python
fs_origin = 100
```

因此前 10 秒对应：

```python
1000 rows
```

QC 规则：

1. 对 `Ut1`、`Ut2` 分别拟合 4 阶多项式慢变基线。
2. 原始信号减去基线，得到高频残差。
3. 如果任一路残差 STD 大于 `2.5 mV`，判坏。
4. 如果两路 STD 比值大于 `3`，判坏。
5. 统计大于 `3 * STD` 的离群点比例。
6. 如果两路离群点比例差异超过 3 倍，且不是两路都低于 3%，判坏。

输出：

```text
outputs/<run_name>/qc/good_samples.csv
outputs/<run_name>/qc/bad_samples.csv
outputs/<run_name>/qc/qc_summary.csv
outputs/<run_name>/qc/unpaired_samples.csv
outputs/<run_name>/qc/motion_type_samples.csv
```

`bad_samples.csv` 主要字段：

```text
group_id
motion_type
data_file
ref_file
file_name
status
reason
std_ut1
std_ut2
outlier_count_ut1
outlier_count_ut2
outlier_ratio_ut1
outlier_ratio_ut2
is_good
```

### 3.3 协议预处理

通过 `load_and_preprocess_protocol()` 处理每个好样本。

输出统一为 13 路协议信号：

```text
ppg_green
ppg_red
ppg_ir
hf1
hf2
cf1
cf2
accx
accy
accz
gyrox
gyroy
gyroz
```

其中：

```python
hf1 = Ut1
hf2 = Ut2
cf1 = Uc1 / (Ut1 - Uc1)
cf2 = Uc2 / (Ut2 - Uc2)
```

如果 CF 分母过小或结果非有限值，会用插值和近邻补齐，最后仍非法的位置置为 0。

### 3.4 缺失值与毛刺处理

处理逻辑：

1. 所有数值列先转为 numeric。
2. 非有限值变成 NaN。
3. 缺失值先线性插值，再近邻补边。
4. PPG 使用滑动中位数相关方法处理短毛刺；如果失败，回退为前值/均值式处理。
5. QC 元数据会随重采样映射到新时间轴。

### 3.5 分类型带通滤波

预处理后按通道类型做零相位 Butterworth 带通滤波：

```text
PPG: 0.5 - 5 Hz
HF:  0.1 - 5 Hz
CF:  0.1 - 5 Hz
ACC: 0.5 - 10 Hz
Gyro: 0.5 - 10 Hz
```

使用：

```python
scipy.signal.butter(order=4)
scipy.signal.filtfilt()
```

如果信号太短或滤波参数非法，会回退为去均值信号，避免流程崩溃。

### 3.6 重采样

训练 trial 中会根据 `Fs_Target` 把所有传感器通道重采样到目标采样率：

```python
scipy.signal.resample_poly()
```

当前搜索空间：

```text
Fs_Target: 25, 50, 100
```

参考 HR 本身仍按参考 CSV 时间轴使用，不把参考 HR 变成高频传感器通道。

### 3.7 初始信号图输出

主流程会为每个好样本输出 13 路信号图：

```text
outputs/<run_name>/signal_figures/
```

主要图：

```text
raw_clean_13ch_<group_id>_<motion_type>.png
motion_bandpass_13ch_<group_id>_<motion_type>.png
```

含义：

```text
raw_clean_13ch:
    原始信号与清洗后信号对比。

motion_bandpass_13ch:
    运动段附近的带通信号，用于检查滤波和分段是否合理。
```

### 3.8 运动分段

运动分段基于三轴 ACC 合模长：

```python
acc_mag = sqrt(accx^2 + accy^2 + accz^2)
```

核心逻辑：

1. 前 30 秒作为静息/校准期。
2. 计算前 30 秒 ACC 合模长 STD。
3. 阈值为：

```python
threshold = 3 * rest_std
```

4. 用 `TW` 秒窗口、1 秒步长滑动。
5. 从静息窗到连续运动窗，定位运动开始。
6. 从连续运动窗到连续静息窗，定位运动结束。

如果分段失败，样本/模式不会直接让整个批处理崩溃，而是在输出 summary 中记录失败原因。

### 3.9 TargetScope：训练作用范围

当前支持：

```text
motion_only
motion_recovery
motion_post10
global
```

含义：

```text
motion_only:
    只在运动段使用 adaptive HR。

motion_recovery:
    在运动段 + 恢复段使用 adaptive HR。

motion_post10:
    在运动段 + 运动结束后 10 秒使用 adaptive HR；不足 10 秒则到文件末尾。

global:
    全段作为目标范围。也接受别名 all、global_all。
```

### 3.10 静息段全局 Tdelay 对齐

代码把“全局 Tdelay 搜索窗口”和“训练窗口”解耦。

重要参数：

```text
Alignment_TW:
    只用于静息段 PPG-HR 提取和全局 Tdelay 搜索，默认 8 秒。

TW:
    只用于后续自适应滤波训练/验证/测试切窗。
```

因此：

```text
同一个 Fs_Target
同一个 Alignment_TW
不同 TW = 6 / 8 / 10
```

会复用同一个全局 Tdelay，但训练窗口数量和窗口中心仍会随着 `TW` 改变。

delay 符号约定：

```text
delay_s > 0:
    认为传感器信号滞后参考 HR。
    对传感器通道左移，裁掉前 delay_s 对应样本。

delay_s < 0:
    认为传感器信号提前参考 HR。
    对传感器通道右移，前端补该通道首样本值 arr[0]，尾部截断。
    不补 0。
```

参考 HR：

```text
参考 HR 通常是 1 Hz。
不做 0.1 秒级高频平移。
仍保留 TW / 2 半窗补偿。
```

### 3.11 delay search 模式

训练中可选：

```text
delay_estimation_mode = "envelope"
delay_estimation_mode = "direct"
```

当前窗口内 delay search 仍然是逐 lag 有效重叠区 Pearson correlation，不是 FFT cross-correlation。

默认优先使用 numba JIT 加速；如果 numba 不可用或 JIT 失败，则回退到 Python reference 实现。

### 3.12 自适应滤波器

当前允许的 adaptive filter：

```text
lms
volterra
rff_lms
klms
```

#### LMS

LMS 步长：

```python
mu = max(LMS_Mu_Min, LMS_Mu_Base - abs_corr / 100)
```

默认：

```text
LMS_Mu_Min = 1e-6
```

#### Volterra

Volterra 使用：

```text
线性项：完整 M + K 非因果向量
二阶项：最近 M2 个 tap 的上三角组合
mu2 = alpha_u * mu1
```

#### RFF-LMS

RFF-LMS 使用固定随机特征。

`rff_seed` 由 trial 参数、mode_key、repeat_idx 和 random_state 稳定哈希生成，并写入：

```text
trial params
trial_history
JSON
cache key
```

change3 阶段 1 后，RFF-LMS 默认使用特征空间 NLMS：

```text
rff_update_mode = "nlms"
rff_nlms_eps = 1e-6
rff_leakage = 0.0
rff_err_clip = None
rff_theta_norm_guard = None
```

这些是固定工程策略，不进入 Optuna 搜索空间，但会进入 `ProtocolTrialParams`、trial cache key 和 `param_*` 记录。旧的 LMS 更新仍可通过 `rff_update_mode="lms"` 兼容测试。

#### KLMS

KLMS 默认使用归一化距离和归一化更新，避免 tap 维度变化后 `klms_epsilon` 的实际尺度漂移：

```text
klms_distance_mode = "normalized"
klms_normalized_update = True
klms_nlms_eps = 1e-6
klms_max_dictionary_size = 300
klms_center_prune_policy = "freeze_new_centers"
```

`klms_epsilon` 字段名保留兼容旧记录；在默认 `normalized` 模式下，它的语义是 `squared_distance / tap_dim` 的阈值。若需要复现旧逻辑，可使用 `klms_distance_mode="absolute_squared"`。

---

## 4. 贝叶斯训练模式参数说明

### 4.1 训练不是传统模型拟合

本工程里的 “train / val / test” 不是训练一个跨文件保存的机器学习模型。

每个 Optuna trial 只是评估一组信号处理超参数：

```text
Fs_Target
TW
Kstop
max_order
M_base
C_scale
K_max
...
```

自适应滤波器权重只在当前样本窗口内在线更新，不跨文件保存。

### 4.2 运动类型内分组训练

主流程按 `motion_type` 分组训练。

例如：

```text
kaihe1
kaihe2
kaihe3
```

属于同一个：

```text
motion_type = "kaihe"
```

会在 `kaihe` 组内做 split / all_train / leave_one_group_out。

### 4.3 DATA_SPLIT_MODE

支持三种：

```text
split
all_train
leave_one_group_out
```

#### split

```python
DATA_SPLIT_MODE = "split"
```

同一 motion_type 内随机划分：

```text
train: 剩余样本
val:   默认 1 个样本
test:  默认 1 个样本
```

对应参数：

```python
val_groups_per_type = 1
test_groups_per_type = 1
```

约束：

```text
good_count - val_groups_per_type - test_groups_per_type >= 1
```

如果某个运动类型好样本不足，会报明确错误。

#### all_train

```python
DATA_SPLIT_MODE = "all_train"
```

含义：

```text
同一 motion_type 的所有好样本都作为 train。
同时也作为 test。
objective 使用 train 指标。
```

适合样本很少时先跑通流程，但不能作为严格泛化评估。

#### leave_one_group_out

```python
DATA_SPLIT_MODE = "leave_one_group_out"
```

简称 LOGO。

含义：

```text
每个 fold 留出一个 group 作为 test。
其余 group 作为 train。
每个 fold 独立运行 Optuna。
```

如果某个 motion_type 少于 2 个好样本，无法执行 LOGO。

LOGO 的 aggregate 指标不是简单平均 fold AAE / accuracy，而是拼接所有 held-out test windows 后重新计算。

LOGO 输出里要特别注意：

```text
result_level:
    fold / aggregate / single_split

aggregation:
    LOGO aggregate 行为是 logo_window_concat

params_semantics:
    fold_best_params
    best_params_for_this_split
    representative_fold_best_params_not_global
```

LOGO aggregate 行中的 best params 只是代表 fold 参数，用于兼容旧输出，不表示该 motion_type 有一个统一全局最优参数。

### 4.4 TargetScope

Notebook 中常用：

```python
ACTIVE_TARGET_SCOPES = [
    "motion_only",
    "motion_recovery",
    "motion_post10",
]
```

也可以启用：

```python
"global"
```

如果写：

```python
"all"
"global_all"
```

会被映射为：

```python
"global"
```

### 4.5 CascadeScheme

支持 7 类参考源 / 级联方案：

```text
ACC3
HF2
CF2
HF2_CF2
CF2_HF2
ACC3_HF2
HF2_ACC3
```

含义简述：

```text
ACC3:
    使用三轴加速度作为运动参考。

HF2:
    使用两路热膜高频信号。

CF2:
    使用两路冷膜比值信号。

HF2_CF2:
    先用 HF2，再用 CF2。

CF2_HF2:
    先用 CF2，再用 HF2。

ACC3_HF2:
    先用 ACC3，再用 HF2。

HF2_ACC3:
    先用 HF2，再用 ACC3。
```

Notebook 可通过：

```python
ACTIVE_CASCADE_SCHEMES = [
    "ACC3",
    "HF2",
    "CF2",
    "HF2_CF2",
    "HF2_ACC3",
]
```

控制本次跑哪些方案。

### 4.6 AdaptiveFilter

可选：

```python
ACTIVE_ADAPTIVE_FILTERS = [
    "lms",
    "volterra",
    "rff_lms",
    "klms",
]
```

建议初次测试只开：

```python
ACTIVE_ADAPTIVE_FILTERS = ["lms"]
```

等流程确认后再打开 Volterra 和 RFF-LMS。

### 4.7 优化目标

```python
OPTIMIZATION_OBJECTIVE = "aae"
```

或：

```python
OPTIMIZATION_OBJECTIVE = "accuracy"
```

或：

```python
OPTIMIZATION_OBJECTIVE = "posthoc_aae"
```

含义：

```text
aae:
    最小化 adaptive AAE（对齐前，基于原始参考 HR）。

accuracy:
    最小化 100 - adaptive_acc_pct（对齐前）。

posthoc_aae:
    最小化后对齐 AAE（基于 oracle 最佳时移后的参考 HR）。
    注意：后对齐会用参考 HR 搜索最优时间偏移，再计算误差，
    属于 oracle 指标，可能引入过拟合风险，建议仅在诊断或
    特定场景下使用。
```

accuracy 定义：

```text
abs_err <= 5 bpm 的窗口比例 × 100
```

### 4.8 公共搜索空间

公共搜索项：

```text
Fs_Target: [25, 50]
TW: [6, 8, 10]
Kstop: [0.2, 0.3, 0.5]
max_order: [8, 12, 16, 20]
M_base: [1, 2]
C_scale: [0.6, 0.9, 1.2, 1.5]
K_max: [8, 12, 16, 20, 30]
Spec_Penalty_Width: [0.1, 0.2, 0.3]
hr_range_hz: [20/60, 25/60, 30/60, 35/60, 40/60]
slew_limit_bpm: [8, 10, 12, 14]
slew_step_bpm: [5, 7, 9]
```

阶段 1 收缩搜索空间后，`smooth_win_len`、`Rest_HR_Track_Band_BPM`、`Rest_HR_Slew_Limit_BPM`、`Rest_HR_Slew_Step_BPM` 不再作为公共搜索项采样。它们仍保留在参数 dataclass 中，可通过 `trial_param_overrides` 固定传入，方便兼容旧记录和少量手动实验。

LMS 专属搜索项：

```text
LMS_Mu_Base: [0.004, 0.006, 0.008]
```

Volterra 专属搜索项：

```text
LMS_Mu_Base: [0.004, 0.006, 0.008]
alpha_u: [0.005, 0.01, 0.03, 0.05, 0.1]
M2: [2, 3]
```

RFF-LMS 专属搜索项：

```text
RFF_LMS_Mu_Base: [0.001, 0.002, 0.004, 0.006]
rff_D: [50, 100, 200]
    rff_sigma_scale: [0.5, 1.0, 2.0, 4.0]
```

注意：RFF-LMS 采样的是 `RFF_LMS_Mu_Base`，但解码后仍写回统一字段：

```text
LMS_Mu_Base
```

`rff_sigma` 现在是 deprecated 兼容字段。新训练不再搜索绝对 sigma，而是搜索
`rff_sigma_scale`。每个 RFF 窗口会先根据 tap matrix 中相邻 tap 向量或有限
pairwise L2 距离的 median 估计 `sigma_base`，再计算：

```text
sigma_eff = max(min_sigma, rff_sigma_scale * sigma_base)
```

Stage JSON / CSV 中会记录 `rff_sigma_scale` 与实际 `sigma_eff`。旧 CSV 或旧
JSON 只有 `rff_sigma`、没有 `rff_sigma_scale` 时，重绘会自动进入 fixed sigma
fallback，不会因为缺新字段而中断。

这样下游求解器可以统一读取。

KLMS 专属搜索项：

```text
klms_step_size: [0.005, 0.01, 0.02, 0.05]
klms_sigma: [0.5, 1.0, 2.0, 5.0]
klms_epsilon: [0.005, 0.01, 0.02, 0.05, 0.1]
```

### 4.9 固定参数：不进入贝叶斯搜索空间

以下参数不是 Optuna 搜索项，而是固定实验配置，通常通过 `trial_param_overrides` 传入：

```python
TRIAL_PARAM_OVERRIDES = {
    "TW_F": 0.0,
    "normalization_mode": "minmax",
    "qc_policy": "fallback_baseline",
    "delay_estimation_mode": "envelope",
    "Alignment_TW": 8.0,
    "Alignment_Step": 1.0,
    "rff_update_mode": "nlms",
    "klms_distance_mode": "normalized",
    "ppg_input_transform": "raw_bandpass",
    "global_objective_strategy": "current_global_adaptive",
    "cascade_guard_policy": "none",
}
```

这些字段的来源通常是 notebook 第 0 块的 `REST_HR_TRIAL_OVERRIDES`。第 5 块 `run_training_cell()` 会复制这份字典，并把调用处临时传入的覆盖项叠加进去。

#### TW_F

```text
TW_F:
    自适应滤波前置收敛上下文长度。
    0 表示沿用旧的单 TW 窗口行为。
    不进入 Optuna 搜索。
```

输出目录 run_name 会包含：

```text
TW_F0s
TW_F2p5s
```

#### normalization_mode

支持：

```text
minmax
zscore
none
```

默认：

```text
minmax
```

#### qc_policy

默认：

```text
fallback_baseline
```

含义：

```text
如果窗口级 QC 判断该窗口不适合 adaptive，
则回退使用 baseline PPG HR。
```

其他策略需要结合代码确认后再使用，不建议随意改。

#### delay_estimation_mode

默认：

```text
envelope
```

可选：

```text
direct
```

#### Alignment_TW / Alignment_Step

```text
Alignment_TW:
    静息段全局 Tdelay 搜索窗口长度，默认 8 秒。

Alignment_Step:
    静息段全局 Tdelay 搜索步长，默认 1 秒。
```

#### RFF/KLMS 稳定性固定参数

这些参数只影响对应滤波器，不参与 Optuna 采样：

```text
rff_update_mode:
    默认 "nlms"，表示 RFF-LMS 在随机特征空间做归一化 LMS 更新。

rff_nlms_eps / rff_leakage / rff_err_clip / rff_theta_norm_guard:
    RFF-LMS 的数值保护参数，默认尽量保持轻量，只启用 NLMS 分母保护。

klms_distance_mode:
    默认 "normalized"，表示中心距离使用 squared_distance / tap_dim。

klms_max_dictionary_size:
    KLMS 字典中心上限，默认 300。

klms_center_prune_policy:
    默认 "freeze_new_centers"，到达字典上限后不再新增中心。
```

#### PPG 输入、global objective 与 cascade guard

阶段 2 新增三类固定策略：

```text
ppg_input_transform:
    "raw_bandpass" 表示沿用旧流程，清洗后 PPG 直接带通。
    "log_absorbance" 表示先用慢变 I0(t) 计算 -log(I/I0)，再进入带通和后续 HR 求解。

global_objective_strategy:
    "current_global_adaptive" 保留旧 global 逻辑。
    "deployment_global" 更贴近部署场景：rest 优先 baseline，motion 优先 adaptive，recovery 由融合/回退逻辑决定。

cascade_guard_policy:
    "none" 表示使用完整级联输出。
    "rms_guard" 会检查每一级输出 RMS ratio，异常时回退本级输出，避免某一级发散污染后续级联。
```

修改 `ppg_input_transform` 后，全局 Tdelay cache key 会随之变化；不要把不同输入策略的缓存结果混在一起解读。

### 4.10 Rest HR 与 Tdelay 相关固定参数

以下为静息段后处理的固定配置，不进入贝叶斯搜索空间。
静息段谱峰追踪参数（`Rest_HR_Track_Band_BPM` / `Rest_HR_Slew_Limit_BPM` /
`Rest_HR_Slew_Step_BPM`）在阶段 1 后也改为固定参数，通常从 notebook 第 0 块传入。

常用默认值：

```text
Rest_HR_Band_BPM: (40, 180)
Rest_HR_Smooth_Method: "median"
Rest_HR_Smooth_Win: 3
Rest_HR_Peak_Percent: 0.3
Rest_HR_Spec_Penalty_Enable: True
Rest_HR_Spec_Penalty_Weight: 0.2
Rest_HR_Spec_Penalty_Width_Hz: 0.2
Rest_Alignment_Score_Mode: "aae"
```

`Rest_Alignment_Score_Mode` 可选：

```text
aae
mae
std
```

其中：

```text
aae / mae:
    更贴近最终 HR 误差。

std:
    保留旧逻辑用于复现实验。
```

### 4.11 自适应滤波后 time bias 诊断

相关参数：

```text
Enable_Time_Bias_After: True
Time_Bias_After_Range_S: (-5, 5)
Time_Bias_After_Step_S: 1
Time_Bias_After_Mode: "posthoc_oracle_alignment"
```

该部分只影响 post-hoc 诊断指标和手动重画，不进入 Optuna 搜索空间。

### 4.12 Recovery fallback 参数

相关参数：

```text
Recovery_Grace_S: 10
Recovery_Diff_Bpm: 20
Recovery_Cross_Diff_Bpm: 8
```

用于恢复段 final HR 融合或 fallback 判断。

### 4.13 每个 cascade 独立训练预算

主流程参数：

```python
max_iterations
num_repeats
```

是默认预算。

可以用：

```python
CASCADE_TRAIN_BUDGETS = {
    "ACC3": {"n_trials": 50, "n_repeats": 1},
    "HF2": {"n_trials": 50, "n_repeats": 1},
    "CF2": {"n_trials": 50, "n_repeats": 1},
}
```

覆盖某些 cascade 的预算。

实际使用逻辑：

```text
如果 CASCADE_TRAIN_BUDGETS 中有该 cascade:
    使用对应 n_trials / n_repeats

否则:
    使用 max_iterations / num_repeats
```

### 4.14 并行

参数：

```python
n_jobs = 1
```

默认建议保持 1，尤其在 Windows + Jupyter 下更稳。

如果：

```text
n_jobs > 1
num_repeats > 1
```

会在 Optuna repeat 级并行。

不做 window 级并行，因为窗口 HR tracking 依赖上一窗口 HR，是顺序状态。

### 4.15 stage JSON

窗口诊断时，stage JSON 会记录每一级级联滤波的短摘要和必要波形字段：

```text
channel
M / K / mu
reference_channel_ranking
output_signal
cascade_full_output_signal
cascade_guarded_output_signal
diagnostics
```

含义：

```text
output_signal:
    本级最终交给后续流程使用的输出。

cascade_full_output_signal:
    不考虑 guard 回退时的完整级联输出。

cascade_guarded_output_signal:
    启用 rms_guard 后实际使用的输出。
```

普通训练不会把所有长诊断数组无限扩展写入汇总 CSV；这些字段主要用于 Stage-8 单窗诊断图。

正式 batch 默认：

```python
save_stage_json = False
```

因此以下列默认是空字符串：

```text
adaptive_stages_json
lms_stages_json
```

如果需要调试每个窗口每一级的详细信息，可以显式开启：

```python
save_stage_json=True
```

或者：

```python
debug_mode=True
```

---

## 5. 输出格式与输出参数说明

### 5.1 run_name 规则

如果不手动指定 `output_root`，代码会根据本次配置生成：

```text
outputs/<run_name>/
```

`run_name` 包含：

```text
target scopes
cascade schemes
adaptive filters
objective mode
data split mode
TW_F 标签
```

示例：

```text
motion_only-motion_recovery__ACC3-HF2-CF2__lms__accuracy__split__TW_F0s
```

### 5.2 总体输出结构

典型输出：

```text
outputs/<run_name>/
    qc/
        good_samples.csv
        bad_samples.csv
        unpaired_samples.csv
        qc_summary.csv
        motion_type_samples.csv

    signal_figures/
        raw_clean_13ch_<group_id>_<motion_type>.png
        motion_bandpass_13ch_<group_id>_<motion_type>.png

    motion_types/
        <motion_type>/
            _checkpoint.json
            _modes/
                <target_scope>__<cascade_scheme>__<adaptive_filter>/
                    mode_result.json
                    history.csv
                    per_group.csv
                    fold_<id>_history.csv
                    fold_<id>_per_group.csv
            split_files.csv
            fold_split_files.csv
            mode_summary_aae.csv
            mode_summary_accuracy.csv
            per_group_aae.csv
            per_group_accuracy.csv
            best_params_lms.csv
            best_params_volterra.csv
            best_params_rff_lms.csv
            best_params_klms.csv
            best_params_all.json
            bayes_curve_data.csv
            bayes_curve.png

            best_params_and_alignment.csv
            best_metrics.csv
            motion_frequency_and_params.csv
            full_report.json

    final_summary/
        motion_only_test_aae.csv
        motion_only_test_accuracy.csv
        motion_recovery_test_aae.csv
        motion_recovery_test_accuracy.csv
        motion_post10_test_aae.csv
        motion_post10_test_accuracy.csv
        global_test_aae.csv
        global_test_accuracy.csv

    batch_summary.csv
```

某些表在没有启用对应 TargetScope 时仍可能生成空表头，便于后处理脚本统一读取。

### 5.3 qc/ 输出

#### good_samples.csv

通过 QC 的样本。

#### bad_samples.csv

未通过 QC 的样本，重点看：

```text
reason
std_ut1
std_ut2
outlier_count_ut1
outlier_count_ut2
outlier_ratio_ut1
outlier_ratio_ut2
```

#### unpaired_samples.csv

未配对或命名不合法的 CSV。

#### qc_summary.csv

样本数量汇总：

```text
paired_samples
good_samples
bad_samples
unpaired_samples
```

#### motion_type_samples.csv

记录每个配对样本的：

```text
group_id
motion_type
motion_index
data_file
ref_file
```

### 5.4 signal_figures/ 输出

每个好样本两类图：

```text
raw_clean_13ch_<group_id>_<motion_type>.png
motion_bandpass_13ch_<group_id>_<motion_type>.png
```

用于运行训练前检查：

```text
原始信号是否明显异常
清洗后信号是否合理
运动段检测是否大致正确
带通后 ACC/HF/CF/PPG 是否仍有可用信息
```

### 5.5 split_files.csv / fold_split_files.csv

`split_files.csv` 说明当前 motion_type 下每个 group 被放入哪个 split：

```text
train
val
test
```

如果是 LOGO，还会输出：

```text
fold_split_files.csv
```

字段包括：

```text
fold_id
heldout_group_id
split
group_id
motion_type
data_file
ref_file
reason
```

### 5.6 mode_summary_aae.csv / mode_summary_accuracy.csv

每个 motion_type 下，不同模式组合的汇总结果。

模式组合由以下字段决定：

```text
target_scope
cascade_scheme
adaptive_filter
objective_mode
data_split_mode
```

常看字段：

```text
success
reason
train_aae_bpm
val_aae_bpm
test_aae_bpm
train_accuracy_pct
val_accuracy_pct
test_accuracy_pct
best_repeat_idx
best_trial_idx
n_trials
n_repeats
result_level
aggregation
params_semantics
```

### 5.7 per_group_aae.csv / per_group_accuracy.csv

按 group 展开后的效果表。

用于看：

```text
同一个 motion_type 下，哪个具体样本效果差
是否某个 group 拖累整体结果
LOGO 每个 held-out group 表现如何
```

### 5.8 best_params_*.csv

按 adaptive filter 分类保存最佳参数：

```text
best_params_lms.csv
best_params_volterra.csv
best_params_rff_lms.csv
best_params_klms.csv
```

常用于手动重画最佳参数 HR 曲线。

change3 后，这些表和 `best_params_and_alignment.csv` 都会尽量写出 `param_*` 分列，例如：

```text
param_Fs_Target
param_TW
param_LMS_Mu_Base
param_rff_update_mode
param_klms_distance_mode
param_ppg_input_transform
param_global_objective_strategy
param_cascade_guard_policy
```

`param_*` 是新 replay / diagnostics 的首选参数来源。它比 JSON 字符串更适合人工筛选，也能避免不同字段同名时混淆。

### 5.9 best_params_all.json

保存每个 motion_type 下所有模式的最佳参数和指标。

适合程序读取，不适合人工快速浏览。

当前实现只在某个 `motion_type` 的全部模式都完成后重建一次该文件。
模式执行过程中会优先写 `_modes/<mode_key>/mode_result.json` 和 `_checkpoint.json`，
用于模式级 resume；这些内部工件不改变 `best_params_all.json` 的最终格式。

Notebook 使用提示：
如果你在 notebook 中重新运行第 7/8 训练单元，并且输出目录保持不变，那么只有
`_checkpoint.json` 里已经标记为 `done` 的 mode 会被直接跳过；没有 `done`
标记的 mode 会整模式重跑，而不是从某个 trial / fold 的中间状态继续。

### 5.10 bayes_curve_data.csv / bayes_curve.png

每个 motion_type 输出一组贝叶斯优化曲线数据和图。

`bayes_curve_data.csv` 字段：

```text
motion_type
target_scope
cascade_scheme
adaptive_filter
repeat_idx
trial_idx
objective_value
aae_bpm
accuracy_pct
best_so_far
success
reason
param_*
```

`bayes_curve_data.csv` 中的 `param_*` 表示每个 trial 实际使用的完整参数快照。它适合排查“某次 trial 为什么好/差”，也适合和 Stage-6 最优记录做字段对照。

模式执行完成后会先把该模式的 history 落到 `_modes/<mode_key>/history.csv`，然后刷新
motion_type 级 `bayes_curve_data.csv`。这允许在清空内存中的 `result.history` 之后，
仍然基于落盘 history 重建 `bayes_curve.png`。

`bayes_curve.png` 的子图数量由下面三者相乘决定：

```text
激活 target scopes × 激活 cascade schemes × 激活 adaptive filters
```

Resume 只支持模式级：只有 `_checkpoint.json` 中标记为 `done` 的 mode 会在重启后跳过；
如果某个 mode 只有部分中间文件、但没有 `done` 标记，则会整模式重跑，而不是从 trial
或 fold 中途续跑。

Notebook 使用提示：
第 9 块输出检查主要看 motion_type 根目录下的 Stage-6 文件是否齐全；`_modes/` 和
`_checkpoint.json` 是内部恢复工件，通常只在排查中断恢复或确认某个 mode 是否已经完成时查看。

### 5.11 best_params_and_alignment.csv

记录最佳参数和对齐信息。

常见字段：

```text
motion_type
split
mode
target_scope
cascade_scheme
adaptive_filter
adaptive_data_type
TW_F
normalization_mode
best_tdelay_s
best_params_json
result_level
params_semantics
param_*
```

这是后续 replay / 重输出的重要输入表。

参数读取优先级：

```text
1. param_* 分列
2. best_params_json 或 params JSON
3. 同行裸字段
4. ProtocolTrialParams 默认值
```

因此旧训练输出仍可重绘，但正式分析建议使用包含 `param_*` 的新输出。

### 5.12 best_metrics.csv

记录最佳模式的核心指标。

常见字段：

```text
baseline_aae_bpm
adaptive_aae_bpm
final_aae_bpm
baseline_acc_pct
adaptive_acc_pct
final_acc_pct
posthoc_baseline_aae_bpm
posthoc_adaptive_aae_bpm
posthoc_final_aae_bpm
posthoc_baseline_acc_pct
posthoc_adaptive_acc_pct
posthoc_final_acc_pct
time_bias_after_s
n_recovery_fallback
num_windows
```

### 5.13 motion_frequency_and_params.csv

记录运动频率和参考通道相关信息。

常见字段：

```text
motion_frequency_hz
penalty_ref_channel
reference_channel_ranking_summary
```

### 5.14 full_report.json

更完整的报告，包含：

```text
motion_type
mode results
params
metrics
fusion_source_distribution
git_commit_hash
```

### 5.15 final_summary/

最终跨运动类型汇总目录。

常见输出：

```text
motion_only_test_aae.csv
motion_only_test_accuracy.csv
motion_recovery_test_aae.csv
motion_recovery_test_accuracy.csv
motion_post10_test_aae.csv
motion_post10_test_accuracy.csv
global_test_aae.csv
global_test_accuracy.csv
```

这些表适合最后横向比较：

```text
不同运动类型
不同 cascade
不同 adaptive filter
不同 target scope
```

### 5.16 batch_summary.csv

批处理级别总表。

常见字段：

```text
motion_type
sample
status
reason
split
result_csv
report_json
```

用于快速判断：

```text
哪些 motion_type 成功
哪些失败
失败原因是什么
对应结果文件在哪里
```

---

## 6. 重绘与重输出表格说明

主训练流程 `run_batch_adaptive_protocol()` 主要负责批量训练、参数搜索、汇总表和基础信号图输出。

训练完成后，Notebook 后半部分提供三类“重绘 / 重输出”功能：

```text
1. 对单个文件使用 best_params 重绘 HR 曲线；
2. 对单个文件重绘窗口级波形与频谱诊断图；
3. 对指定滤波器与指定滤波数据类型重绘跨运动类型汇总表。
```

需要特别注意：Notebook 中存在两种参数使用模式。

```text
代码段 10 / 11：
    使用 best_params 的数据类型 A；
    实际滤波也使用同一个数据类型 A。

代码段 13 / 14：
    CROSS_RUN_PARAM_SCHEME 提供 best_params 的来源数据类型 A；
    CROSS_RUN_APPLIED_SCHEME 指定实际滤波时使用的数据类型 B；
    因此可以出现“用 A 的最优参数，实际应用到 B 上滤波”的跨方案重放。
```

---

### 6.1 单个文件 best_params HR 曲线重绘

#### 6.1.1 功能目的

该功能用于抽查某一个原始文件在某组最佳参数下的 HR 曲线表现。

它会重新执行：

```text
读取单个 sensor CSV / ref CSV
-> 预处理
-> 分段
-> 对齐
-> 使用 best_params 重新运行 HR 解算
-> 输出 HR 曲线图和 HR 曲线 CSV
```

适合回答以下问题：

```text
某个样本的最终 HR 曲线是否合理？
adaptive HR 是否比 baseline HR 更接近参考 HR？
final HR 在 rest / motion / recovery 三段分别来自哪里？
recovery fallback 是否过多？
```

#### 6.1.2 代码段 10 / 11：参数来源方案 = 实际滤波方案

代码段 10 / 11 的模式是最直观的重绘模式：

```text
best_params 来自某个 cascade_scheme；
实际重跑滤波也使用同一个 cascade_scheme。
```

例如：

```text
best_params 来源：ACC3
实际滤波数据：ACC3
```

或者：

```text
best_params 来源：HF2_CF2
实际滤波数据：HF2_CF2
```

这种模式用于验证“某个训练结果本身”是否合理。

典型手动设置项：

```python
SENSOR_CSV_PATH = Path(r"...\multi_kaihe1.csv")
REF_CSV_PATH = Path(r"...\multi_kaihe1_ref.csv")

TARGET_SCOPE = "motion_only"
CASCADE_SCHEME = "ACC3"
ADAPTIVE_FILTER = "lms"

BEST_PARAM_CSV_PATH = Path(r"...\motion_types\kaihe\best_params_lms.csv")
OUTPUT_DIR = Path(r"...\redraw_single_file")
```

其中：

```text
CASCADE_SCHEME:
    同时表示 best_params 的来源数据类型；
    也表示实际滤波时使用的数据类型。
```

输出通常包括：

```text
HR 曲线 PNG
HR 曲线 CSV
```

HR 曲线中通常包含：

```text
Reference HR
Baseline PPG HR
Adaptive HR
Final HR
```

HR 曲线 CSV 常见字段：

```text
time_s
segment_label
reference_hr_bpm
baseline_hr_bpm
adaptive_hr_bpm
final_hr_bpm
final_source
fusion_reason
```

---

### 6.2 单个文件窗口级波形与频谱诊断图重绘

change3 后，窗口级诊断会输出三类图：

```text
window_waveform_*.png:
    当前窗口的原始 PPG、级联 stage 中间输出、完整级联输出和 guard 后实际使用输出。

window_spectrum_*.png:
    FFT window 内的原始频谱、自适应滤波后频谱、运动惩罚后频谱，以及 motion/候选 HR 标记。

window_weights_*.png:
    自适应滤波器权重、KLMS 字典或短诊断摘要。
```

如果 `cascade_guard_policy="rms_guard"`，波形图中的“完整级联”和“guard 后实际使用”可能不同。这通常表示某一级输出 RMS 异常，被 guard 回退保护，不代表 replay 失败。

#### 6.2.1 功能目的

窗口级诊断图不是最终汇总表，而是用于排查某个文件、某个窗口、某个滤波结果为什么好或坏。

它重点检查：

```text
滤波前 PPG 波形
补偿参考信号波形
自适应滤波后波形
窗口内频谱
谱峰候选
最终选中的 HR 峰
baseline / adaptive / final HR 的差异
```

适合排查：

```text
是否锁错峰；
是否倍频 / 半频误判；
滤波后波形是否发散；
滤波后主频是否仍可识别；
运动伪影频率是否压过 PPG 主频；
某个窗口为什么 fallback；
某个窗口为什么 adaptive HR 偏差很大。
```

#### 6.2.2 诊断图与 HR 曲线重绘的区别

HR 曲线重绘关注：

```text
窗口级 HR 数值随时间的变化。
```

窗口级波形与频谱诊断关注：

```text
每个窗口内部到底发生了什么。
```

因此它通常会比 HR 曲线更细，适合 debug，而不是最终展示。

#### 6.2.3 典型手动设置项

通常需要指定：

```python
SENSOR_CSV_PATH = Path(r"...\multi_kaihe1.csv")
REF_CSV_PATH = Path(r"...\multi_kaihe1_ref.csv")

TARGET_SCOPE = "motion_only"
CASCADE_SCHEME = "ACC3"
ADAPTIVE_FILTER = "lms"

BEST_PARAM_CSV_PATH = Path(r"...\motion_types\kaihe\best_params_lms.csv")
OUTPUT_DIR = Path(r"...\window_diagnostics")
```

还可能需要指定：

```python
WINDOW_INDEX_LIST = [0, 1, 2, 3]
```

或通过时间范围筛选：

```python
TIME_RANGE_S = (60, 100)
```

具体变量名以 Notebook 当前单元为准。

#### 6.2.4 建议重点查看的字段

窗口诊断 CSV 或图中建议关注：

```text
time_s
segment_label
ref_hr_bpm
baseline_ppg_hr_bpm
adaptive_hr_bpm
final_hr_bpm
final_source
qc_status
missing_ratio
ValidFlag_ratio
best_tdelay_s
motion_frequency_hz
selected_peak_hz
selected_peak_bpm
```

如果在图中看到：

```text
滤波后波形幅度逐渐变大；
主峰从真实 HR 跳到运动频率；
连续多个窗口 selected peak 靠近上一错误 HR；
adaptive HR 与 baseline HR 长时间分离；
```

说明问题可能不在最终 fusion，而在滤波、delay、频谱选峰或状态转移阶段。

---

### 6.3 跨运动类型汇总表重绘

#### 6.3.1 功能目的

跨运动类型汇总表用于把不同 `motion_type` 的结果重新整理成横向对比表。

它不是重新训练，而是读取已有输出目录中的记录，例如：

```text
motion_types/<motion_type>/best_params_and_alignment.csv
motion_types/<motion_type>/best_metrics.csv
motion_types/<motion_type>/motion_frequency_and_params.csv
motion_types/<motion_type>/full_report.json
```

然后重建指定条件下的汇总表。

适合回答：

```text
某个 adaptive_filter 在所有运动类型上效果如何？
某个 cascade / applied scheme 是否稳定？
不同运动类型的 AAE / accuracy 谁更差？
同一套 best_params 换到另一类滤波数据后是否仍有效？
```

---

### 6.4 代码段 13 / 14：跨方案参数重放模式

#### 6.4.1 两个 scheme 的含义

代码段 13 / 14 新增了两个关键变量：

```python
CROSS_RUN_PARAM_SCHEME = "ACC3"
CROSS_RUN_APPLIED_SCHEME = "HF2"
```

它们含义不同：

```text
CROSS_RUN_PARAM_SCHEME:
    提供 best_params 的数据类型 / cascade 方案。
    也就是参数是从哪个方案训练出来的。

CROSS_RUN_APPLIED_SCHEME:
    实际重跑滤波时使用的数据类型 / cascade 方案。
    也就是这些参数被应用到哪个方案上。
```

因此存在两种情况。

第一种：同方案重放。

```python
CROSS_RUN_PARAM_SCHEME = "ACC3"
CROSS_RUN_APPLIED_SCHEME = "ACC3"
```

含义：

```text
用 ACC3 训练得到的最优参数；
仍然应用到 ACC3 上滤波。
```

这与代码段 10 / 11 的逻辑基本一致。

第二种：跨方案重放。

```python
CROSS_RUN_PARAM_SCHEME = "ACC3"
CROSS_RUN_APPLIED_SCHEME = "HF2"
```

含义：

```text
读取 ACC3 训练得到的最优参数；
但实际滤波时使用 HF2 数据。
```

这可以用于评估：

```text
某一类参数是否能迁移到另一类参考数据；
参数好是因为参数本身稳健，还是因为对应参考数据类型更合适；
不同参考源之间是否存在共享的最优参数结构。
```

#### 6.4.2 为什么需要区分 param scheme 和 applied scheme

因为 `best_params` 里包含的参数主要是：

```text
Fs_Target
TW
Kstop
max_order
M_base
C_scale
K_max
Spec_Penalty_Width
smooth_win_len
hr_range_hz
slew_limit_bpm
slew_step_bpm
Rest_HR_Track_Band_BPM
Rest_HR_Slew_Limit_BPM
Rest_HR_Slew_Step_BPM
LMS_Mu_Base
Volterra / RFF / KLMS 专属参数
```

这些参数本身不一定绑定某个数据源。

但是实际滤波时的参考通道由 `cascade_scheme` 决定，例如：

```text
ACC3:
    使用三轴加速度作为补偿参考。

HF2:
    使用两路热膜高频信号作为补偿参考。

CF2:
    使用两路冷膜比值信号作为补偿参考。

HF2_CF2:
    先用 HF2，再用 CF2。

ACC3_HF2:
    先用 ACC3，再用 HF2。
```

所以可以出现：

```text
参数来自 ACC3；
实际滤波用 HF2；
参数来自 HF2_CF2；
实际滤波用 ACC3_HF2。
```

这就是代码段 13 / 14 与代码段 10 / 11 的核心差异。

---

### 6.5 跨运动类型汇总表的推荐命名方式

为了避免混淆，建议重绘输出表名中同时包含：

```text
param_scheme
applied_scheme
adaptive_filter
target_scope
split / mode
TW_F
```

例如：

```text
cross_motion_lms_param-ACC3_applied-HF2_motion_only_test_TW_F0s.csv
```

这样可以一眼看出：

```text
参数来自 ACC3；
实际滤波使用 HF2；
滤波器是 lms；
目标段是 motion_only；
评估 split 是 test；
TW_F 是 0s。
```

---

### 6.6 使用跨方案重放时的注意事项

跨方案重放不是标准训练结果本身，而是一个诊断 / 迁移实验。

解释结果时要注意：

```text
如果 param_scheme = applied_scheme：
    结果表示该方案自身 best_params 的复现实验。

如果 param_scheme != applied_scheme：
    结果表示“参数迁移到另一个参考源后”的效果；
    不能简单说 applied_scheme 自己的最优效果就是这个结果。
```

例如：

```text
CROSS_RUN_PARAM_SCHEME = "ACC3"
CROSS_RUN_APPLIED_SCHEME = "HF2"
```

如果效果差，可能说明：

```text
ACC3 的最优参数不适合 HF2；
HF2 本身不适合该运动类型；
当前参数空间对跨源迁移不稳；
delay / M / K 映射依赖参考源特性。
```

如果效果好，可能说明：

```text
该组参数具有跨参考源稳健性；
或者 ACC3 与 HF2 在该运动类型中捕捉到相似的伪影结构。
```

---

### 6.7 建议的重绘 / 重输出顺序

建议不要一开始就做跨方案汇总。

推荐顺序：

```text
1. 先用代码段 10 / 11，对单文件做同方案 HR 曲线重绘；
2. 如果 HR 曲线异常，再做窗口级波形与频谱诊断图；
3. 确认单文件重绘逻辑正常后，再做跨运动类型汇总；
4. 最后再使用代码段 13 / 14 做 param_scheme != applied_scheme 的跨方案迁移实验。
```

---

## 7. 手动设置项汇总

本节按照实际使用流程汇总需要手动设置的路径和超参数。

---

### 7.1 数据读取 / 初始绘图测试

这一部分用于确认：

```text
文件能否配对；
QC 是否通过；
预处理是否能完成；
初始 13 路信号图是否合理；
主流程是否能跑通。
```

#### 必改路径

```python
PROJECT_ROOT = Path(r"D:\python_notebook_base3")
SRC_DIR = PROJECT_ROOT / "src"
TESTDATA_DIR = PROJECT_ROOT / "testdata"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
```

如果数据不在 `testdata`，需要改：

```python
INPUT_DIR = Path(r"D:\your_data_dir")
```

#### 原始采样率

```python
FS_ORIGIN = 100
```

如果传感器原始 CSV 是 100 Hz，保持默认。

#### 输出目录与是否清理

```python
RUN_NAME = "your_run_name"
RUN_OUTPUT_DIR = OUTPUT_ROOT / RUN_NAME

CLEAN_OUTPUTS = False
```

第一次测试建议：

```python
CLEAN_OUTPUTS = False
```

确认输出目录没问题后，再改成：

```python
CLEAN_OUTPUTS = True
```

#### 快速跑通配置

```python
ACTIVE_TARGET_SCOPES = ["motion_only"]
ACTIVE_CASCADE_SCHEMES = ["ACC3"]
ACTIVE_ADAPTIVE_FILTERS = ["lms"]

OPTIMIZATION_OBJECTIVE = "aae"
DATA_SPLIT_MODE = "all_train"

MAX_ITERATIONS = 1
NUM_REPEATS = 1
NUM_SEED_POINTS = 1
N_JOBS = 1
```

检查输出：

```text
qc/good_samples.csv
qc/bad_samples.csv
qc/unpaired_samples.csv
qc/qc_summary.csv
signal_figures/
batch_summary.csv
```

---

### 7.2 贝叶斯优化训练

这一部分决定正式训练范围、训练预算和评价目标。

#### TargetScope

```python
ACTIVE_TARGET_SCOPES = [
    "motion_only",
    "motion_recovery",
    "motion_post10",
]
```

如需全段目标，可加入：

```python
"global"
```

其中：

```text
motion_only:
    只评估运动段。

motion_recovery:
    评估运动段 + 恢复段。

motion_post10:
    评估运动段 + 运动结束后 10 秒。

global:
    评估全段。
```

#### CascadeScheme

快速测试：

```python
ACTIVE_CASCADE_SCHEMES = ["ACC3"]
```

正式比较：

```python
ACTIVE_CASCADE_SCHEMES = [
    "ACC3",
    "HF2",
    "CF2",
    "HF2_CF2",
    "CF2_HF2",
    "ACC3_HF2",
    "HF2_ACC3",
]
```

#### AdaptiveFilter

快速测试：

```python
ACTIVE_ADAPTIVE_FILTERS = ["lms"]
```

正式比较：

```python
ACTIVE_ADAPTIVE_FILTERS = [
    "lms",
    "volterra",
    "rff_lms",
    "klms",
]
```

#### 优化目标

以误差为目标：

```python
OPTIMIZATION_OBJECTIVE = "aae"
```

以 5 bpm 内准确率为目标：

```python
OPTIMIZATION_OBJECTIVE = "accuracy"
```

#### 数据划分模式

样本少、先跑通：

```python
DATA_SPLIT_MODE = "all_train"
```

常规 train / val / test：

```python
DATA_SPLIT_MODE = "split"
VAL_GROUPS_PER_TYPE = 1
TEST_GROUPS_PER_TYPE = 1
```

逐样本留一：

```python
DATA_SPLIT_MODE = "leave_one_group_out"
```

#### 训练预算

快速 smoke test：

```python
MAX_ITERATIONS = 1
NUM_REPEATS = 1
NUM_SEED_POINTS = 1
N_JOBS = 1
```

小预算测试：

```python
MAX_ITERATIONS = 5
NUM_REPEATS = 1
NUM_SEED_POINTS = 2
N_JOBS = 1
```

正式训练示例：

```python
MAX_ITERATIONS = 350
NUM_REPEATS = 3
NUM_SEED_POINTS = 10
RANDOM_STATE = 42
N_JOBS = 1
```

#### 每个 cascade 独立训练预算

```python
CASCADE_TRAIN_BUDGETS = {
    "ACC3": {"n_trials": 50, "n_repeats": 1},
    "HF2": {"n_trials": 50, "n_repeats": 1},
    "CF2": {"n_trials": 50, "n_repeats": 1},
}
```

实际逻辑：

```text
如果某个 cascade 在 CASCADE_TRAIN_BUDGETS 中：
    使用对应 n_trials / n_repeats。

否则：
    使用 MAX_ITERATIONS / NUM_REPEATS。
```

#### 固定 trial 参数

```python
TRIAL_PARAM_OVERRIDES = {
    "TW_F": 0.0,
    "normalization_mode": "minmax",
    "qc_policy": "fallback_baseline",
    "delay_estimation_mode": "envelope",
    "Alignment_TW": 8.0,
    "Alignment_Step": 1.0,
    "Rest_Alignment_Score_Mode": "aae",
    "Enable_Time_Bias_After": True,
    "Time_Bias_After_Range_S": (-5.0, 5.0),
    "Time_Bias_After_Step_S": 1.0,
    "rff_update_mode": "nlms",
    "klms_distance_mode": "normalized",
    "ppg_input_transform": "raw_bandpass",
    "global_objective_strategy": "current_global_adaptive",
    "cascade_guard_policy": "none",
}
```

常见修改项：

```text
TW_F:
    自适应滤波前置上下文长度。
    0 表示旧行为。

normalization_mode:
    minmax / zscore / none。

qc_policy:
    默认 fallback_baseline。

delay_estimation_mode:
    envelope / direct。

Alignment_TW:
    静息段全局 Tdelay 搜索窗口。

rff_update_mode / klms_distance_mode:
    RFF-LMS 与 KLMS 的稳定性策略。

ppg_input_transform:
    PPG 输入变换策略，默认 raw_bandpass；可切换 log_absorbance 做对照。

global_objective_strategy:
    global 目标和 final HR 选择策略。

cascade_guard_policy:
    级联保护策略，默认 none；rms_guard 会在某级输出异常时回退。
```

---

### 7.3 单文件同方案 HR 曲线重绘：代码段 10 / 11

适用场景：

```text
检查某个文件在自己 best_params 下的 HR 曲线。
```

需要设置：

```python
SENSOR_CSV_PATH = Path(r"...\multi_kaihe1.csv")
REF_CSV_PATH = Path(r"...\multi_kaihe1_ref.csv")

RESULTS_ROOT = OUTPUT_ROOT / "<run_name>"
MOTION_TYPE = "kaihe"

TARGET_SCOPE = "motion_only"
CASCADE_SCHEME = "ACC3"
ADAPTIVE_FILTER = "lms"

TW_F = 0.0
OUTPUT_DIR = Path(r"...\redraw_single_file")
```

这里：

```text
CASCADE_SCHEME 同时表示：
    best_params 来源数据类型；
    实际滤波数据类型。
```

即：

```text
param_scheme = applied_scheme = CASCADE_SCHEME
```

检查输出：

```text
HR 曲线 PNG
HR 曲线 CSV
```

重点查看：

```text
reference_hr_bpm
baseline_hr_bpm
adaptive_hr_bpm
final_hr_bpm
final_source
fusion_reason
segment_label
```

---

### 7.4 单文件窗口级波形与频谱诊断图重绘

适用场景：

```text
HR 曲线异常；
需要看具体窗口内的滤波波形和频谱选峰。
```

需要设置：

```python
SENSOR_CSV_PATH = Path(r"...\multi_kaihe1.csv")
REF_CSV_PATH = Path(r"...\multi_kaihe1_ref.csv")

RESULTS_ROOT = OUTPUT_ROOT / "<run_name>"
MOTION_TYPE = "kaihe"

TARGET_SCOPE = "motion_only"
CASCADE_SCHEME = "ACC3"
ADAPTIVE_FILTER = "lms"

TW_F = 0.0
OUTPUT_DIR = Path(r"...\window_diagnostics")
```

如果 Notebook 提供窗口筛选项，则可以设置：

```python
WINDOW_INDEX_LIST = [0, 1, 2, 3]
```

或：

```python
TIME_RANGE_S = (60, 100)
```

具体变量名以当前 Notebook 单元为准。

重点检查：

```text
滤波前 PPG
补偿参考通道
滤波后 residual / adaptive waveform
频谱主峰
top-k 峰候选
selected HR
previous HR
baseline HR
adaptive HR
reference HR
```

如果排查自适应滤波发散，建议额外关注：

```text
滤波后波形幅值是否随时间变大；
滤波后频谱是否被低频漂移或运动频率主导；
selected peak 是否连续锁在错误频率；
adaptive HR 是否长期偏离 baseline 和 reference。
```

---

### 7.5 跨运动类型汇总表重绘：同方案模式

适用场景：

```text
重新生成某个 filter / cascade / target_scope 在所有 motion_type 上的汇总表。
```

同方案模式：

```python
CROSS_RUN_PARAM_SCHEME = "ACC3"
CROSS_RUN_APPLIED_SCHEME = "ACC3"
```

含义：

```text
使用 ACC3 的 best_params；
实际也用 ACC3 滤波。
```

常用设置：

```python
RESULTS_ROOT = OUTPUT_ROOT / "<run_name>"

CROSS_RUN_TARGET_SCOPE = "motion_only"
CROSS_RUN_ADAPTIVE_FILTER = "lms"

CROSS_RUN_PARAM_SCHEME = "ACC3"
CROSS_RUN_APPLIED_SCHEME = "ACC3"

CROSS_RUN_TW_F = 0.0
CROSS_RUN_OUTPUT_DIR = RESULTS_ROOT / "cross_motion_redraw"
```

建议输出表名包含：

```text
param-ACC3_applied-ACC3_lms_motion_only_TW_F0s
```

---

### 7.6 跨运动类型汇总表重绘：跨方案模式

适用场景：

```text
想验证某个数据类型训练出来的 best_params，
应用到另一个数据类型滤波时效果如何。
```

例如：

```python
CROSS_RUN_PARAM_SCHEME = "ACC3"
CROSS_RUN_APPLIED_SCHEME = "HF2"
```

含义：

```text
读取 ACC3 方案下训练得到的 best_params；
实际重放时使用 HF2 作为滤波数据类型。
```

常用设置：

```python
RESULTS_ROOT = OUTPUT_ROOT / "<run_name>"

CROSS_RUN_TARGET_SCOPE = "motion_only"
CROSS_RUN_ADAPTIVE_FILTER = "lms"

CROSS_RUN_PARAM_SCHEME = "ACC3"
CROSS_RUN_APPLIED_SCHEME = "HF2"

CROSS_RUN_TW_F = 0.0
CROSS_RUN_OUTPUT_DIR = RESULTS_ROOT / "cross_motion_redraw"
```

建议输出表名包含：

```text
param-ACC3_applied-HF2_lms_motion_only_TW_F0s
```

解释结果时必须注明：

```text
这不是 HF2 自己训练得到的最优结果；
这是 ACC3 参数迁移到 HF2 数据类型上的结果。
```

---

### 7.7 重绘 / 重输出时最容易混淆的变量

#### TARGET_SCOPE / CROSS_RUN_TARGET_SCOPE

控制评估哪一段：

```text
motion_only
motion_recovery
motion_post10
global
```

#### ADAPTIVE_FILTER / CROSS_RUN_ADAPTIVE_FILTER

控制滤波器类型：

```text
lms
volterra
rff_lms
klms
```

#### CASCADE_SCHEME

在代码段 10 / 11 中同时表示：

```text
best_params 来源 scheme；
实际滤波 applied scheme。
```

#### CROSS_RUN_PARAM_SCHEME

只表示：

```text
best_params 从哪个 scheme 的训练结果中读取。
```

#### CROSS_RUN_APPLIED_SCHEME

只表示：

```text
重放时实际使用哪个 scheme 的参考数据进行滤波。
```

#### TW_F / CROSS_RUN_TW_F

用于筛选或固定：

```text
自适应滤波前置上下文长度。
```

如果训练时用了：

```text
TW_F = 0.0
```

重绘时也应保持：

```text
TW_F = 0.0
```

否则可能找不到对应记录，或者重放结果与原训练记录不一致。

---

### 7.8 推荐完整使用顺序

#### 第一步：主流程 smoke test

```python
ACTIVE_TARGET_SCOPES = ["motion_only"]
ACTIVE_CASCADE_SCHEMES = ["ACC3"]
ACTIVE_ADAPTIVE_FILTERS = ["lms"]
DATA_SPLIT_MODE = "all_train"
MAX_ITERATIONS = 1
NUM_REPEATS = 1
```

检查：

```text
qc/
signal_figures/
batch_summary.csv
```

#### 第二步：小预算训练

```python
MAX_ITERATIONS = 5
NUM_REPEATS = 1
NUM_SEED_POINTS = 2
```

检查：

```text
motion_types/<motion_type>/mode_summary_aae.csv
motion_types/<motion_type>/best_params_and_alignment.csv
motion_types/<motion_type>/best_metrics.csv
```

#### 第三步：单文件同方案 HR 曲线重绘

使用代码段 10 / 11：

```text
param_scheme = applied_scheme
```

检查：

```text
HR 曲线是否合理；
final_source 是否符合预期；
motion 段 adaptive 是否明显改善；
recovery 是否频繁 fallback。
```

#### 第四步：窗口级波形与频谱诊断图

如果 HR 曲线异常，再检查：

```text
滤波前后波形；
频谱主峰；
top-k 峰候选；
selected peak；
是否锁错峰；
是否滤波发散。
```

#### 第五步：正式训练

扩大：

```python
ACTIVE_TARGET_SCOPES
ACTIVE_CASCADE_SCHEMES
ACTIVE_ADAPTIVE_FILTERS
MAX_ITERATIONS
NUM_REPEATS
```

建议仍保持：

```python
N_JOBS = 1
save_stage_json = False
```

#### 第六步：跨运动类型同方案汇总

例如：

```python
CROSS_RUN_PARAM_SCHEME = "ACC3"
CROSS_RUN_APPLIED_SCHEME = "ACC3"
```

用于复核某个方案自己的整体效果。

#### 第七步：跨运动类型跨方案汇总

例如：

```python
CROSS_RUN_PARAM_SCHEME = "ACC3"
CROSS_RUN_APPLIED_SCHEME = "HF2"
```

用于评估参数迁移性。

解释时必须区分：

```text
param scheme:
    参数来自哪里。

applied scheme:
    实际滤波用哪里。
```

不能把跨方案重放结果直接当成 `CROSS_RUN_APPLIED_SCHEME` 自身训练得到的最优结果。

---

## 8. Notebook 代码块快速上手指南

本章按代码块的顺序逐一说明每个代码块的功能，并按照参数大类汇总修改位置。只覆盖代码块（不含 markdown 说明块），且不涉及跨运动类型汇总与跨滤波架构回放（代码块 12/13/14）。

### 8.1 代码块总览

Notebook 共 15 个代码块，按流水线组织为"环境初始化 — 诊断预览 — 训练 — 结果检查与回放"四个阶段：

```
代码块 0  (Cell  2): 环境初始化与全局参数
代码块 1  (Cell  4): 样本发现与 QC
代码块 2  (Cell  6): 单样本预处理预览 (smoke test)
代码块 3  (Cell  8): 静息段 Tdelay 对齐诊断图
代码块 4  (Cell 10): 全段未对齐 PPG-HR 图
代码块 5  (Cell 12): 训练函数封装 (只定义，不跑训练)
代码块 6  (Cell 14): 静息段原始 PPG 与 PPG-HR 双 y 轴图
代码块 7  (Cell 16): 训练单元 1 — all_train 拟合上限
代码块 8  (Cell 18): 训练单元 2 — 84 模式连通性检查 (默认关闭)
代码块 9  (Cell 20): Stage-6 输出检查
代码块 10 (Cell 22): Stage-7 单文件 HR 曲线回放
代码块 11 (Cell 24): Stage-8 窗口级波形与频谱诊断
代码块 12 (Cell 26): Stage-9 跨运动类型汇总表 (本章不展开)
代码块 13 (Cell 28): 跨滤波架构 Stage-7 replay (本章不展开)
代码块 14 (Cell 30): 跨滤波架构 Stage-8 窗口诊断 (本章不展开)
```

### 8.2 各代码块功能说明

#### 代码块 0 (Cell 2) — 环境初始化与全局参数

**必须第一个运行。** 一次性设置整个 notebook 的工作环境：

- 定义项目路径 `PROJECT_ROOT`、源码路径 `SRC_DIR`、测试数据目录 `TESTDATA_DIR`、输出根目录 `OUTPUT_ROOT`
- 将 `src/` 加入 `sys.path`，导入 `ppg_hr` 所有依赖
- 定义**全部固定参数**（TW_F、normalization_mode、qc_policy、Alignment_TW、静息段 HR 后处理参数、恢复段容差等）
- 定义 change3 新增固定策略（RFF/KLMS 稳定性、PPG 输入变换、global objective、cascade guard、param_* 读取优先级）
- 定义默认训练组合（target scope / cascade scheme / adaptive filter / 预算）
- 调用 `make_unique_output_dir()` 创建本次运行的输出子目录 `RUN_OUTPUT_DIR`

运行后打印项目路径和当前参数汇总，确认环境正确。

#### 代码块 1 (Cell 4) — 样本发现与 QC

**依赖代码块 0。** 自动扫描 `testdata/` 目录：

- 通过 `discover_sample_pairs_with_unpaired` 识别 `multi_<运动类型><编号>.csv` 及其 `_ref.csv` 配对
- 逐样本执行 `quality_filter_sample`，分出 `good_pairs`（通过 QC）和 `bad_pairs`（未通过）
- 未配对文件写入 `unpaired_df`
- 额外对新格式 CSV（优先 `multi_tiaosheng1.csv`）做字段验证，检查 QC 元数据列是否齐全

输出 `good_pairs` 是后续所有绘图和训练的输入来源。

#### 代码块 2 (Cell 6) — 单样本预处理预览

**依赖代码块 1。** 取 `good_pairs[0]` 做快速体检（smoke test）：

- 加载传感器 CSV 和参考 HR CSV
- 重采样到 100 Hz
- 运行 ACC 运动分段检测
- 用固定 `ALIGNMENT_TW` 执行全局 Tdelay 对齐
- 输出摘要表格（运动起止时间、静息/运动/恢复窗口数量、best_tdelay_s）

如果此处失败，后续训练大概率也会失败。

#### 代码块 3 (Cell 8) — 静息段 Tdelay 对齐诊断图

**依赖代码块 1 + 代码块 0 的 REST_HR_* 参数。** 作用：

- 按运动类型逐个加载数据集
- 调用 `plot_rest_alignment_diagnostics_by_motion_type` 生成对齐诊断 PNG
- 每张图叠加静息段 PPG-HR 曲线与参考 HR 曲线，用于人工判断全局 Tdelay 是否合理

如果静息段 HR 偏离严重，优先回代码块 0 调整 `REST_HR_*` 参数。

#### 代码块 4 (Cell 10) — 全段未对齐 PPG-HR 图

**依赖代码块 1 + 代码块 3 的数据集（缺失时自动补加载）。** 作用：

- 不做 Tdelay 对齐，在完整 `PPG_Green` 上直接分窗提取 HR
- 生成每个运动类型的全段未对齐 PPG-HR PNG
- 用于判断异常是来自原始 PPG 主频问题、分段位置还是后处理参数

输出目录为 `OUTPUT_ROOT / "allfield"`。

#### 代码块 5 (Cell 12) — 训练函数封装

**不跑训练，只定义函数。** 依赖代码块 0 的全部固定参数。作用：

- 定义 `run_training_cell()` 函数，封装 `run_batch_adaptive_protocol` 的完整调用
- 定义 `notebook_progress` 进度回调，每 10 个 trial 打印一次关键指标
- 定义 `make_budgets()` 快捷生成每个 cascade scheme 的训练预算
- 合并 `trial_param_overrides`，把第 0 块固定策略传入 `ProtocolTrialParams`

后续代码块 7/8 只需填参数组合 + 调用 `run_training_cell()` 即可启动训练。

#### 代码块 6 (Cell 14) — 静息段原始 PPG 与 PPG-HR 双 y 轴图

**依赖代码块 1 + 已有数据集。** 作用：

- 对每个运动类型画一张双 y 轴图：左轴是原始 PPG 波形，右轴是从 PPG 估计出的 HR
- 是代码块 3（Tdelay 诊断）的辅助视角，侧重观察静息段信号质量（漂移、断裂、周期不清等）

#### 代码块 7 (Cell 16) — 训练单元 1：all_train 拟合上限

**这是核心训练代码块。** 依赖代码块 1 的 `good_pairs` + 代码块 5 的 `run_training_cell`。作用：

- 用所有可用样本训练并在同批样本上测试，观察流程在训练集上的拟合上限
- 按本块设定的 `ACTIVE_TARGET_SCOPES`、`ACTIVE_CASCADE_SCHEMES`、`ACTIVE_ADAPTIVE_FILTERS` 启动 Optuna 搜索
- 每个运动类型独立训练，结果写入 `RUN_OUTPUT_DIR` 下的 Stage-6 文件

小规模检查保持 `n_trials=1, n_repeats=1`；正式实验可增大预算或扩展组合。

#### 代码块 8 (Cell 18) — 训练单元 2：84 模式连通性检查

**默认关闭 (`RUN_FULL_84_MODE_TEST = False`)。** 依赖代码块 5。作用：

- 可选地跑 3 种 target scope × 7 种 cascade scheme × 4 种 adaptive filter = 84 个组合，其中 adaptive filter 包含 `lms` / `volterra` / `rff_lms` / `klms`
- 用于在正式实验前确认全部模式是否能连通、无崩溃

开启时输出 `result_full_84`，后续代码块 9/10/11 可切换使用该结果。

#### 代码块 9 (Cell 20) — Stage-6 输出检查

**依赖代码块 7 或代码块 8 的训练结果。** 作用：

- 自动选择最近一次训练结果（优先级：`result_full_84` > `result_full_63` > `result_all_train`）
- 检查每个运动类型目录下是否存在 4 个 Stage-6 文件：
  - `best_params_and_alignment.csv`
  - `best_metrics.csv`
  - `motion_frequency_and_params.csv`
  - `full_report.json`
- 展示关键列和 fusion 分布
- 确认新记录是否含有 `param_*` 分列，供后续 replay/diagnostics 优先读取

如果代码块 10 找不到 `best_params_and_alignment.csv`，先回本块确认路径正确。

#### 代码块 10 (Cell 22) — Stage-7 单文件 HR 曲线回放

**依赖代码块 9 能找到的 Stage-6 记录。** 默认开启 (`RUN_STAGE7_REPLAY = True`)。作用：

- 从 `best_params_and_alignment.csv` 自动匹配并恢复最优参数
- 对**单个指定的 CSV 文件**（`REPLAY_SIGNAL_CSV` + `REPLAY_REF_CSV`）重新跑完整 HR 提取
- 输出 HR 曲线 PNG（含 Reference/Baseline/Adaptive/Final 四条曲线）和窗口级 CSV
- 参数恢复统一使用 `protocol_params_from_record`，优先 `param_*`，再兼容旧 JSON

需要指定：`REPLAY_SIGNAL_CSV`、`REPLAY_REF_CSV`、`REPLAY_MOTION_TYPE`、`REPLAY_TARGET_SCOPE`、`REPLAY_ADAPTIVE_FILTER`、`REPLAY_CASCADE_SCHEME`。

#### 代码块 11 (Cell 24) — Stage-8 窗口级波形与频谱诊断

**依赖代码块 10 的 replay 结果。** 默认开启 (`RUN_WINDOW_DIAGNOSTICS = True`)。作用：

- 选取一个 FFT 窗口（由 `MANUAL_ALIGNED_FFT_START_S` 指定秒数起点），深度诊断
- 生成三类图：
  - **波形图**：双 y 轴，左轴画原始 PPG 和各 stage 中间波形，右轴画最终自适应滤波后波形
  - **频谱图**：仅用 FFT window 内数据，画三条归一化幅频谱（原始 PPG / 自适应滤波后 / 加运动惩罚后），标注 motion 峰和候选 HR
- 额外生成 **权重/诊断图**：查看 `window_weights_*.png` 中的滤波器权重、KLMS 字典或短诊断摘要
- 输出 stage 摘要表（每级的 channel、M、K、mu、reference_channel_ranking 等）

注：stage 1/2/... 是级联滤波中每个自适应步骤的中间输出，不是最终 HR，也不是新的搜索参数表。

### 8.3 参数分类与修改位置

#### 第一类：路径配置

| 参数 | 修改位置 | 说明 |
|------|----------|------|
| `PROJECT_ROOT` | 代码块 0 | 项目根目录，仓库移动时只改这里 |
| `SRC_DIR` | 代码块 0 | 源码目录，由 PROJECT_ROOT 派生 |
| `TESTDATA_DIR` | 代码块 0 | 测试数据目录，由 PROJECT_ROOT 派生 |
| `OUTPUT_ROOT` | 代码块 0 | 输出根目录，由 PROJECT_ROOT 派生 |
| `CLEAN_OUTPUTS` | 代码块 0 | `True` 时每次运行覆盖之前的输出子目录 |
| `FS_ORIGIN` | 代码块 0 | 原始采样率，默认 100 |
| `RANDOM_STATE` | 代码块 0 | 随机种子，默认 42 |

#### 第二类：核心算法固定参数

| 参数 | 修改位置 | 说明 |
|------|----------|------|
| `TW_F` | 代码块 0；可在代码块 7/8 覆盖 | 未来窗口前置长度(s)，不进搜索空间。这是最核心的固定参数 |
| `NORMALIZATION_MODE` | 代码块 0 | 归一化方式：`"minmax"` / `"zscore"` / `"none"` |
| `QC_POLICY` | 代码块 0 | QC 策略，默认 `"fallback_baseline"` |
| `DELAY_ESTIMATION_MODE` | 代码块 0 | 延迟估计模式：`"envelope"` / `"direct"` |

这些参数通过 `REST_HR_TRIAL_OVERRIDES` 字典统一传入训练流程，不进入 Optuna 搜索空间。

#### 第三类：对齐与时间参数

| 参数 | 修改位置 | 说明 |
|------|----------|------|
| `ALIGNMENT_TW` | 代码块 0 | 静息段 FFT 对齐窗口(s)，**不等于训练窗口 TW**，默认 8.0 |
| `ALIGNMENT_STEP_S` | 代码块 0 | 对齐搜索步长(s)，默认 1.0 |
| `ENABLE_TIME_BIAS_AFTER` | 代码块 0 | 是否启用对齐后时移修正 |
| `TIME_BIAS_AFTER_RANGE_S` | 代码块 0 | 时移搜索范围(s)，默认 (-5, 5) |
| `TIME_BIAS_AFTER_STEP_S` | 代码块 0 | 时移搜索步长(s)，默认 1.0 |
| `TIME_BIAS_AFTER_MODE` | 代码块 0 | 时移模式，默认 `"posthoc_oracle_alignment"` |
| `REST_ALIGNMENT_SCORE_MODE` | 代码块 0 | 对齐评分模式：`"aae"` / `"mae"` / `"std"` |

#### 第四类：静息段 HR 后处理参数

以下为固定后处理参数，不进入 Optuna 搜索空间。
静息段谱峰追踪参数（`REST_HR_TRACK_BAND_BPM` / `REST_HR_SLEW_LIMIT_BPM` /
`REST_HR_SLEW_STEP_BPM`）已进入公共搜索空间，由贝叶斯优化自动选取。

| 参数 | 修改位置 | 说明 |
|------|----------|------|
| `REST_HR_BAND_BPM` | 代码块 0 | HR 带通范围(bpm)，默认 (40, 180) |
| `REST_HR_SMOOTH_METHOD` | 代码块 0 | 平滑方法，默认 `"median"` |
| `REST_HR_SMOOTH_WIN` | 代码块 0 | 平滑窗口大小(帧数)，默认 7 |
| `REST_HR_PEAK_PERCENT` | 代码块 0 | 峰值检测阈值比例，默认 0.3 |
| `REST_HR_SPEC_PENALTY_ENABLE` | 代码块 0 | 是否启用频谱惩罚 |
| `REST_HR_SPEC_PENALTY_WEIGHT` | 代码块 0 | 频谱惩罚权重，默认 0.2 |
| `REST_HR_SPEC_PENALTY_WIDTH_HZ` | 代码块 0 | 频谱惩罚宽度(Hz)，默认 0.2 |

> 如果静息段 HR 曲线明显偏离参考（代码块 3/4/6 诊断图异常），优先调整此类参数。

#### 第五类：恢复段容差参数

| 参数 | 修改位置 | 说明 |
|------|----------|------|
| `RECOVERY_GRACE_S` | 代码块 0 | 恢复段宽限期(s)，默认 15.0 |
| `RECOVERY_DIFF_BPM` | 代码块 0 | 恢复段允许的 HR 偏差(bpm)，默认 150.0 |
| `RECOVERY_CROSS_DIFF_BPM` | 代码块 0 | 跨源恢复偏差阈值(bpm)，默认 8.0 |

#### 第六类：训练组合与搜索预算

| 参数 | 修改位置 | 说明 |
|------|----------|------|
| `ACTIVE_TARGET_SCOPES` | 代码块 0（默认值）；代码块 7（正式训练）；代码块 8（84 模式） | 评估窗口范围：`"global"` / `"motion_only"` / `"motion_recovery"` / `"motion_post10"` |
| `ACTIVE_CASCADE_SCHEMES` | 同上 | 级联滤波方案：`"ACC3"` / `"HF2"` / `"CF2"` / `"HF2_CF2"` / `"CF2_HF2"` / `"ACC3_HF2"` / `"HF2_ACC3"` |
| `ACTIVE_ADAPTIVE_FILTERS` | 同上 | 自适应滤波器类型：`"lms"` / `"volterra"` / `"rff_lms"` / `"klms"` |
| `CASCADE_TRAIN_BUDGETS` | 同上 | 每个 cascade scheme 的 `n_trials` 和 `n_repeats` |
| `OPTIMIZATION_OBJECTIVE` | 同上 | 优化目标：`"aae"` / `"accuracy"` / `"posthoc_aae"` |
| `DATA_SPLIT_MODE` | 同上 | 数据划分：`"all_train"` / `"split"` / `"leave_one_group_out"` |
| `VAL_GROUPS_PER_TYPE` | 同上 | 每运动类型验证组数，默认 1 |
| `TEST_GROUPS_PER_TYPE` | 同上 | 每运动类型测试组数，默认 1 |

> 快速测试建议：代码块 7 中保持 `n_trials=1, n_repeats=1`，只开 `["ACC3"]` + `["lms"]`。确认跑通后再扩大组合和预算。代码块 0 中的默认值仅用于初始化输出目录，实际训练参数以代码块 7 为准。

#### 第七类：诊断开关与单文件选择

| 参数 | 修改位置 | 说明 |
|------|----------|------|
| `RUN_FULL_84_MODE_TEST` | 代码块 8 | `True` 时跑 84 模式连通性检查，默认 `False` |
| `RUN_STAGE7_REPLAY` | 代码块 10 | `True` 时启用 Stage-7 单文件回放 |
| `REPLAY_SIGNAL_CSV` | 代码块 10 | 要回放的传感器 CSV 路径 |
| `REPLAY_REF_CSV` | 代码块 10 | 要回放的参考心率 CSV 路径 |
| `REPLAY_MOTION_TYPE` | 代码块 10 | 回放样本的运动类型（如 `"tiaosheng"`） |
| `REPLAY_TARGET_SCOPE` | 代码块 10 | 回放使用的 target scope |
| `REPLAY_ADAPTIVE_FILTER` | 代码块 10 | 回放使用的滤波器类型 |
| `REPLAY_CASCADE_SCHEME` | 代码块 10 | 回放使用的级联方案 |
| `REPLAY_TW_F` | 代码块 10 | 回放使用的 TW_F，默认沿用代码块 0 的值 |
| `RUN_WINDOW_DIAGNOSTICS` | 代码块 11 | `True` 时启用 Stage-8 窗口诊断 |
| `MANUAL_ALIGNED_FFT_START_S` | 代码块 11 | 手动指定诊断窗口起点(秒)，设为 `None` 则自动取 replay 第一窗 |

#### 第八类：QC 元数据列

| 参数 | 修改位置 | 说明 |
|------|----------|------|
| `QC_METADATA_COLUMNS` | 代码块 1 | 新格式 CSV 中期望出现的 QC 元数据列名列表 |

### 8.4 典型工作流

**首次使用（诊断阶段）：**

```
代码块 0 → 代码块 1 → 代码块 2 → 代码块 3 → 代码块 4 → 代码块 6
(环境)    (QC)       (预览)     (对齐图)    (未对齐图)   (双y轴图)
```
检查所有诊断图，确认参数合理后再进入训练。

**小规模训练：**

```
代码块 5 → 代码块 7 → 代码块 9
(封装)    (all_train) (检查Stage-6输出)
```

**结果诊断：**

```
代码块 10 → 代码块 11
(replay HR曲线) (窗口波形/频谱)
```

**参数调优循环：**
如果训练效果不理想，按以下顺序排查：

1. 先看代码块 3 的对齐诊断图 --- 静息段对齐是否正常？不正常则调整第四类参数（静息段 HR 后处理）
2. 再看代码块 4 的未对齐 PPG-HR 图 --- 原始 PPG 主频是否可信？不可信则可能是数据本身问题
3. 看代码块 10 的 replay HR 曲线 --- Adaptive HR 是否优于 Baseline？Final HR 是否合理？
4. 看代码块 11 的窗口频谱 --- 是否锁错峰？滤波后波形是否发散？如果是，调整第六类参数（训练组合与搜索预算）
5. 全局参数调整回代码块 0，训练参数调整回代码块 7，重复训练

## 8. change5 新增：FFT/SSR、ACC3 对比与批参考源对比

### 8.1 HR 后处理配置

Notebook 第 0 块新增：

```python
TRAIN_HR_POSTPROCESS_METHOD = "fft"
REDRAW_HR_POSTPROCESS_METHOD = "fft"
SSR_NUM_ATOMS = 5
SSR_LAMBDA = 0.15
SSR_HARMONIC_TOL_BPM = 5.0
SSR_FALLBACK_TO_FFT = True
SSR_GRID_RESOLUTION_BPM = 1.0
```

默认训练仍使用 FFT，保证旧逻辑一致。把 `TRAIN_HR_POSTPROCESS_METHOD` 改为
`"ssr"` 后，SSR 只作为 HR peak extraction/postprocess 分支：它只使用 PPG、
运动参考和上一窗预测 HR，不使用 `ref_hr_bpm` 参与选峰，避免标签泄漏。SSR
候选为空或失败时，若 `SSR_FALLBACK_TO_FFT=True`，会记录
`postprocess_fallback="fft"` 并安全回退 FFT。

### 8.2 ACC3 参考源诊断对比

每个 motion_type 的每个 mode 在 best_params 确定后，会额外用同一 best_params
把实际 `cascade_scheme` 替换为 `ACC3` 做 test split 复评估。该对比只用于诊断，
不重新优化，不改变原 mode 的 best_params，也不参与 objective。

新增输出列：

```text
acc3_compare_aae_bpm
acc3_compare_accuracy_pct
acc3_compare_num_windows
acc3_compare_status
acc3_compare_reason
```

这些列会出现在 `mode_summary_*.csv`、`best_params_*.csv`、
`best_params_all.json` 和 `best_params_and_alignment.csv` 中。若原方案本来就是
`ACC3`，状态为 `same_as_original`，直接复用原 test 指标。

### 8.3 批参考源对比

Notebook 第 15 块调用：

```python
run_batch_reference_compare(
    motion_type=MOTION_TYPE,
    best_param_source_cascade_scheme=BEST_PARAM_SOURCE_CASCADE_SCHEME,
    actual_reference_cascade_scheme=ACTUAL_REFERENCE_CASCADE_SCHEME,
    target_scope=TARGET_SCOPE,
    adaptive_filter=ADAPTIVE_FILTER,
    best_param_csv_path=BEST_PARAM_CSV_PATH,
    output_dir=OUTPUT_DIR,
)
```

该函数读取 `best_params_and_alignment.csv` 和同目录的 `split_files.csv`，默认使用
test split，对“参数来源参考源”和“实际参考源”分别用同一 best_params 复评估，
输出：

```text
batch_reference_compare_<motion_type>.csv
```

CSV 使用 `utf-8-sig`，包含 source/actual 两套 `AAE`、`accuracy`、
`num_windows`、`success` 和 `reason` 字段。

### 8.4 旧结果兼容

旧 CSV 缺少 `rff_sigma_scale`、`postprocess_method`、SSR 参数或
`acc3_compare_*` 字段时，读取逻辑会回退到 `ProtocolTrialParams` 默认值；旧 RFF
结果只有 `rff_sigma` 时使用 fixed sigma fallback。正式复现实验仍建议优先使用
包含 `param_*` 分列的新 Stage-6 输出。
