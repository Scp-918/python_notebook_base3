# Repository Guidelines

## 项目结构与模块组织

本仓库是 PPG 心率求解算法的 Python 实现。源码位于 `src/ppg_hr/`：`params.py` 集中维护协议枚举、默认参数和搜索空间；`preprocess/` 负责 CSV 读取、清洗和预处理；`core/` 放置基础求解工具；`experimental/` 包含批量自适应协议、对齐、分段、融合、输出和回放逻辑。

测试位于 `tests/`，测试数据位于 `testdata/`，主实验入口是 `notebooks/run_batch_adaptive_protocol.ipynb`。运行产生的图表、CSV、JSON 等结果应写入 `outputs/`，不要提交生成物。

## 构建、测试与开发命令

使用 conda 环境 `PPG_sensor_env`。仓库当前没有 `setup.py` 或 `pyproject.toml`，因此本地命令需要显式设置 `PYTHONPATH=src`。

```powershell
$env:PYTHONPATH="src"; conda run -n ppg-hr python -m pytest -q tests/
$env:PYTHONPATH="src"; conda run -n ppg-hr python -m pytest -q tests/test_segmentation.py
$env:PYTHONPATH="src"; conda run -n ppg-hr ruff check src/ tests/
```

提交前运行完整测试；定位单个问题时先运行相关测试文件。Ruff 可用时用于静态检查。

## 编码风格与命名约定

Python 代码使用 4 空格缩进。新增公共函数应保留类型标注和简短 docstring。求解参数只在 `src/ppg_hr/params.py` 增加，并提供默认值，避免在 Notebook 或流程代码中散落硬编码默认参数。

函数、变量、模块名使用 `snake_case`；类、dataclass、Enum 使用 `PascalCase`。保留既有协议命名，例如 `ACC3`、`HF2_CF2`、`motion_only`、`rff_lms`。

## 测试指南

测试框架为 `pytest`。新增测试文件使用 `test_*.py` 命名，并放在 `tests/` 中与相关功能相邻。优先使用确定性的合成数据；涉及协议输出时，同时断言记录字段和生成文件。较大数据、临时输出和缓存不要提交。

## 数据、Notebook 与输出约定

传感器 CSV 命名为 `multi_<motion_type><index>.csv`，参考心率 CSV 命名为 `*_ref.csv`；合法运动类型包括 `tiaosheng`、`wanju`、`fuwo`、`kaihe`、`bobi`。QC 策略是坏样本只标记，不阻断后续计算。所有求解路径统一存窗口中心时间，误差和绘图通过 `time_bias` 做时移对齐。

不要提交 `notebooks/run_batch_adaptive_protocol.ipynb` 中的本机 `PROJECT_ROOT` 路径。较大流程变更后，同步更新 `notebooks/README.md`。

## 提交与 PR 规范

日常提交信息使用中文，保持短小、明确、原子化，例如 `修复参考心率CSV解析`。不要使用 `--no-verify`。历史中也存在少量 `docs:` 前缀，可用于纯文档修改。

PR 面向 `upstream/change3`，说明改动目的、核心文件、已运行测试和 Notebook/输出影响。不确定某个本地改动是否应提交时，先确认再纳入提交。
