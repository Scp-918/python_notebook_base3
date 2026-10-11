# PPG 心率批量实验

本仓库提供以 Notebook 为入口的 PPG 心率求解实验闭环：读取传感器和参考心率数据，完成标定、QC、预处理、活动分段和时间对齐，再比较不同参考通道级联与自适应滤波器，优化参数并输出心率误差、图表和可回放记录。当前 change6 使用单受试者目录，不提供独立 CLI 或 GUI。

## 代码结构

| 路径 | 职责 |
| --- | --- |
| `src/ppg_hr/params.py` | 协议枚举、默认参数及搜索配置 |
| `src/ppg_hr/preprocess/` | 传感器/HR CSV 读取、序号修复、插值和 `ck.mat` 标定 |
| `src/ppg_hr/core/` | 基础峰值查找工具 |
| `src/ppg_hr/experimental/batch_pairing.py`、`qc.py`、`preprocess_protocol.py` | 严格配对、质量标记与协议数据构建 |
| `experimental/segmentation.py`、`alignment.py`、`envelope_delay.py` | 活动分段、静息 HR/Tdelay 与 post-hoc 对齐 |
| `experimental/cascade_solver.py`、`noncausal_lms.py`、`volterra.py`、`rff_lms.py`、`klms.py` | 通道级联与四类自适应滤波 |
| `experimental/spectral_utils.py`、`enhanced_tracking.py`、`post_motion_guard.py`、`fusion.py` | 频谱心率提取、增强追踪、运动后保护与融合 |
| `experimental/protocol_search_space.py`、`protocol_optimizer.py` | 搜索空间解码及优化工具 |
| `experimental/run_batch_protocol.py`、`protocol_outputs.py` | 批量调度、Optuna 优化、模式断点、记录、绘图和回放 |
| `notebooks/run_batch_adaptive_protocol.ipynb` | 主实验入口 |
| `scripts/update_change6_notebook.py` | Notebook 生成/更新辅助脚本 |
| `tests/`、`testdata/` | pytest 回归测试与测试数据 |
| `subject_data/`、`outputs/` | 受试者数据与本机实验结果；结果目录不提交 |

数据流为：文件配对 → 标定与 QC → 预处理 → 分段/对齐 → 按运动与实验组合优化 → 最优参数复算/融合 → 结果与断点 → 回放和诊断。

## 运行与开发

使用 conda 环境 `PPG_sensor_env`。本仓库未配置 Python 安装包，本地测试需要设置 `PYTHONPATH=src`。完整命令和编码、数据、提交及 worktree 守则以 [AGENTS.md](AGENTS.md) 为准。

Notebook 输入格式、配置和运行顺序见 [notebooks/readme.md](notebooks/readme.md)。传感器与参考心率文件按 `{subject}_{motion}_{index}_sensor.csv` / `{subject}_{motion}_{index}_HRdata.csv` 配对，motion 为 `write/gripper/run/rope`，标定读取受试者目录的上一级 `ck.mat`。

后续 worktree 固定放在 `D:\python_notebook_base\.worktrees`，目录内容由 `.gitignore` 忽略；其中分支的源码提交可以正常推送。
