# python_notebook_base 批量自适应滤波 Notebook 说明

本文档对应：

```text
D:\HeartDecode\outline-PPGtoHR-main\python_notebook_base\notebooks\run_batch_adaptive_protocol.ipynb
```

该 Notebook 用于 PPG 心率估计中的批量数据配对、原始质量过滤、信号预处理、运动/静息/恢复分段、PPG 与参考 HR 对齐、Fmove 估计、2 个优化目标 x 7 个级联自适应滤波方案的 Optuna 贝叶斯优化、CSV/JSON/PNG 结果输出。

## 1. 当前路径

工程根目录：

```text
D:\HeartDecode\outline-PPGtoHR-main
```

Notebook 代码目录：

```text
D:\HeartDecode\outline-PPGtoHR-main\python_notebook_base
```

输入数据目录：

```text
D:\HeartDecode\outline-PPGtoHR-main\python_notebook_base\testdata
```

输出目录：

```text
D:\HeartDecode\outline-PPGtoHR-main\python_notebook_base\outputs\batch_adaptive_protocol
```

## 2. 数据命名规则

多通道运动数据：

```text
multi_<运动拼音数字>.csv
```

参考心率数据：

```text
multi_<运动拼音数字>_ref.csv
```

例如 `multi_kaihe1.csv` 与 `multi_kaihe1_ref.csv` 会配成一组，`group_id` 为 `kaihe1`。没有成对出现的 CSV 会写入 `unpaired_samples.csv`，不会进入 QC 或优化。

## 3. 运行方式

推荐在 Jupyter 中从上到下运行 Notebook。第 0 节会把：

```python
PROJECT_ROOT = Path(r"D:\HeartDecode\outline-PPGtoHR-main")
PYTHON_DIR = PROJECT_ROOT / "python_notebook_base"
TESTDATA_DIR = PYTHON_DIR / "testdata"
SRC_DIR = PYTHON_DIR / "src"
```

加入运行环境。运行前请确认：

- `TESTDATA_DIR` 中存在配对的 `multi_*.csv` 和 `multi_*_ref.csv`；
- 当前 kernel 安装了 `numpy/scipy/pandas/matplotlib/scikit-learn/optuna`；
- `python_notebook_base/outputs` 可写；
- `CLEAN_OUTPUTS` 默认为 `False`，需要重新生成全部输出时手动改为 `True`。

也可以用命令行做烟测：

```bat
conda run -n PPG_sensor_env python python_notebook_base\run_debug_protocol.py
```

## 4. Debug 与正式参数

Notebook 默认是调试模式：

```python
DEBUG_MODE = True
QUICK_TEST_N_TRIALS = 3
MAX_ITERATIONS = QUICK_TEST_N_TRIALS
NUM_REPEATS = 3
N_JOBS = 1
```

调试模式会完整跑 14 个模式，并保留每个模式 3 个 repeat，但每个 repeat 只跑 3 个 trial，便于先验证路径、输出和 repeat 逻辑。

正式训练切换为：

```python
DEBUG_MODE = False
FORMAL_N_TRIALS = 250
MAX_ITERATIONS = FORMAL_N_TRIALS
NUM_REPEATS = 3
N_JOBS = None
```

每个样本正式配置为 2 个优化目标 x 7 个滤波方案 x 3 repeats x 250 trials。JSON 中会记录 `DEBUG_MODE`、`n_trials`、`n_repeats` 和 `best_repeat_idx`。

## 5. 进度显示

Notebook 第 8 节的 `notebook_progress(info)` 会显示批量训练进度，包括：

- QC 正在检查第几个配对文件；
- 当前训练到第几个好样本；
- 当前样本名；
- 当前 14 个模式中的第几个模式；
- 当前目标段和级联方案；
- 当前 repeat 与 trial；
- 当前 trial AAE 与历史最优 AAE；
- 样本完成后输出 CSV 和 JSON 路径。

调试模式默认每个 trial 都打印。正式模式默认每 10 个 trial 打印一次，同时总会打印每个 repeat 的第 1 个和最后 1 个 trial：

```python
PROGRESS_EVERY_N_TRIALS = 1   # debug
PROGRESS_EVERY_N_TRIALS = 10  # formal
```

## 6. GPU 与加速说明

当前代码不会使用 GPU。主要计算路径是：

- NumPy / SciPy 的滤波、FFT、重采样、相关性计算；
- Optuna 的 TPE 采样；
- scikit-learn `RandomForestRegressor` 参数重要性；
- Python 循环中的逐窗非因果 LMS。

这些实现默认运行在 CPU 上。直接加 GPU 不会自动变快，因为 SciPy `filtfilt/resample_poly`、scikit-learn 随机森林和当前 LMS 循环都不是 GPU 版本。若要真正利用 GPU，需要把核心数组计算迁移到 CuPy / PyTorch / Numba CUDA，并重写或替换滤波、FFT、LMS 与随机森林重要性部分，改动较大，且 Windows/Jupyter 环境和数据传输开销也需要验证。

短期更现实的加速方向是：

- 先用 `DEBUG_MODE=True` 跑通；
- 正式训练时减少不必要的输出显示；
- 对不同样本或不同模式做进程级并行；
- 对 `_run_windows` / `noncausal_lms_filter` 做 Numba CPU JIT 或缓存中间结果。

## 7. QC 规则

QC 只检查每个多通道运动文件前 10 秒的 `Ut1(mV)` 与 `Ut2(mV)`：

- 原始采样率按 100 Hz，前 10 秒共 1000 点；
- 对 Ut1/Ut2 分别做 4 阶多项式基线拟合并扣除；
- 计算去基线后的 STD；
- 统计绝对值超过 `3 * STD` 的离群点数量和比例；
- 任一路 STD > 2.5 mV，判坏；
- 任一路 STD 是另一者 3 倍以上，判坏；
- 若离群点比例一者大于另一者 3 倍以上，且两者不同时小于 1%，判坏；
- 若两路离群比例都小于 1%，即使比例倍数超过 3，也不因该规则判坏。

QC 表格输出包含 `group_id`、`data_file`、`ref_file`、`is_good`、`reason`、两路 STD、两路离群点数量和两路离群点比例。

## 8. 自适应滤波与动态 LMS 步长

每个时间窗内会先将 13 路信号分别归一化到 0-1。HF、CF、ACC 相对 PPG Green 的包络时延由 `Kstop * Fmove` 低通包络和 Pearson 相关估计得到。

LMS 参数按当前选中的通道独立计算：

```text
LMS_Mu_Base = 0.01
LMS_Mu_Min = 1e-5
curr_corr = abs(best_corr)
mu = max(LMS_Mu_Min, LMS_Mu_Base - curr_corr / 100)
```

ACC、HF、CF 三类传感器各自使用自己的 `curr_corr` 与 `mu`。混合级联时，每一级都会记录 `sensor_type`、`channel`、`D_opt_samples`、`R_max`、`curr_corr`、`M`、`K`、`mu` 和 `mode`，并写入结果 CSV 的 `lms_stages_json` 与 JSON 的 `lms_stage_summary`。

## 9. 优化目标与级联方案

优化目标：

- `motion_only`：只以运动段 adaptive HR 的 AAE 最小为目标；
- `motion_recovery`：以运动段 + 恢复段 adaptive HR 的 AAE 最小为目标。

7 种级联方案：

- `ACC3`
- `HF2`
- `CF2`
- `HF2_CF2`
- `CF2_HF2`
- `ACC3_HF2`
- `HF2_ACC3`

每个模式运行 3 个独立 repeat，随机种子为 `random_state + repeat_idx`，最终取 AAE 最小的 repeat 参数作为该模式输出。贝叶斯训练曲线只绘制最终最优 repeat 的 trial history。

`accuracy` 当前定义为目标段内绝对误差 `< 5 bpm` 的窗口比例，单位为百分比。

## 10. 输出文件

批处理级 CSV：

```text
csv/good_samples.csv
csv/bad_samples.csv
csv/unpaired_samples.csv
csv/qc_summary.csv
csv/batch_summary.csv
```

每个成功样本：

```text
csv/adaptive_results_<group_id>.csv
report/Best_Params_Result_<group_id>.json
filtered_motion_signals/filtered_motion_13ch_<group_id>.png
hr_compare/hr_compare_motion_only_<group_id>.png
hr_compare/hr_compare_motion_recovery_<group_id>.png
bayes_training_curves/bayes_curve_motion_only_<group_id>.png
bayes_training_curves/bayes_curve_motion_recovery_<group_id>.png
```

跨样本指标矩阵会输出到新的结果文件夹：

```text
metric_matrix_tables/filtered_motion_aae_bpm.csv
metric_matrix_tables/filtered_motion_accuracy_pct.csv
metric_matrix_tables/filtered_motion_recovery_aae_bpm.csv
metric_matrix_tables/filtered_motion_recovery_accuracy_pct.csv
```

这 4 张表均为 `7 行 x n 列`：每行对应 1 种级联滤波方案，每列对应 1 个好采样运动 `group_id`。其中 AAE 与 accuracy 都只按实际滤波时间段计算：`motion_only` 只算运动段，`motion_recovery` 只算运动段 + 运动恢复段。

图片说明：

- 13 路带通滤波运动段信号图：5 个子图，分别为 HF、CF、PPG、ACC、Gyro；
- HR 对比图：每个目标 1 张图，7 个子图，对比真实 HR、未去伪影 PPG baseline、adaptive HR；
- 贝叶斯训练曲线图：每个目标 1 张图，7 个子图，只绘制最优 repeat 的 trial AAE。

## 11. 常见问题

中文乱码：代码会优先使用 `Microsoft YaHei/SimHei/Noto Sans CJK SC`，如果系统没有中文字体，Matplotlib 可能仍会警告或显示方块。安装中文字体后重跑绘图 cell。

Optuna 未安装：Notebook 导入会失败。安装 `optuna` 后重启 kernel；核心优化模块本身有随机搜索 fallback，但 Notebook 推荐安装 Optuna。

路径错误：第 0 节确认 `PROJECT_ROOT` 是 `D:\HeartDecode\outline-PPGtoHR-main`，不是旧路径，也不是 `python_notebook_base` 本身。

数据长度不足：QC、分段或对齐会记录失败原因并跳过该组；查看 `batch_summary.csv` 和 `bad_samples.csv`。

找不到运动段：ACC 合模长没有出现“10 个静息窗到 10 个运动窗再回到静息窗”的明确转移时会跳过，并在结果中记录 reason。

trial 数过大导致运行慢：先保持 `DEBUG_MODE=True` 跑通；确认输出正确后再改为 `DEBUG_MODE=False`。

清理输出：只改 `CLEAN_OUTPUTS=True`，清理函数限制在协议输出目录内，并禁止清理 `testdata`、工程根目录、源码目录。
