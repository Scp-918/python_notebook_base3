# Repository Guidelines

## 项目结构与模块组织

本仓库是以 Notebook 为入口的 PPG 心率求解实验实现，用于比较运动伪影抑制、心率追踪和参考心率误差。主要代码在 `notebooks/` 与 `src/ppg_hr/`：`params.py` 管理参数；`preprocess/` 负责读取、修复和标定；`core/` 提供基础工具；`experimental/` 负责协议、优化、输出和回放。按任务定位并阅读相关代码，非必要不预读大量固定文档；需要结构说明或运行步骤时才查 `README.md` 或 `notebooks/readme.md`。

测试位于 `tests/`，测试数据位于 `testdata/`，主实验入口是 `notebooks/run_batch_adaptive_protocol.ipynb`。运行产生的图表、CSV、JSON 等结果应写入 `outputs/`，不要提交生成物。

## Orchestration policy

- 本项目额外授权主代理按本节规则自动派遣子代理，无需逐次请求委派许可。简单任务、需要连续判断的步骤直接完成；边界清楚、可独立交付且需要多次工具调用的工作优先委派。
- `explorer`：代码定位、调用链、依赖关系和明确问题的只读探索。`luna_worker`：边界与验收标准明确的实现、bug fix 和针对性测试。`worker`：复杂实现、跨模块修改，或 `luna_worker` 出现实质性验收失败后的升级任务；环境或权限问题先由主代理诊断，不以同模型反复重试代替升级。
- `worker` 使用 `gpt-6.1-sol / medium`，用途保持不变。`researcher` 使用 `gpt-5.6-luna / medium`，只读查询官方文档、库或 API，返回版本适用性、直接来源链接和不确定项。
- `architect` 使用 `gpt-6-astra / high`，只读提供方案、取舍和验收建议；仅在大型计划制定前，或同类错误经复现与修复尝试仍多次重现时调用，日常小任务不调用。科研/架构最终判断仍由主代理负责。
- 高风险任务、重要算法修改和里程碑验收使用只读 `reviewer`；`default` 仅作 fallback，不优先使用。模型、推理强度和权限以当前加载的角色配置为准；配置未加载时明确报告，避免静默替换。
- 默认同一时刻只运行一个写入型子代理；主代理避免同时修改其负责的文件。独立只读任务可以并行。子代理不得继续派生子代理。
- 主代理始终负责拆解、科研/架构判断、集成和最终验收；委派结果是证据输入，不自动等于验收通过。
- 派遣说明必须包含目标、负责的文件/范围、约束、验收标准和必要上下文；告知写入型代理保留其他人的改动。默认使用新上下文（`fork_turns="none"`），只传必要信息；审查者获得需求、基准、差异和验证证据，独立形成结论。
- 子代理运行与改动相关的测试；bug fix 先复现，再验证修复。报告结果、文件位置、实际运行的命令与结果、未验证项和剩余风险；只读代理报告证据和可执行检查，不为测试申请写权限。
- 主代理检查实际差异及验收证据，整合后运行必要检查；提交前运行完整测试。失败时区分代码问题与环境问题，未通过或未执行的检查必须明确报告。

## 构建、测试与开发命令

使用 conda 环境 `PPG_sensor_env`。仓库当前没有 `setup.py` 或 `pyproject.toml`，因此本地命令需要显式设置 `PYTHONPATH=src`。

```powershell
$env:PYTHONPATH="src"; conda run -n PPG_sensor_env python -m pytest -q tests/
$env:PYTHONPATH="src"; conda run -n PPG_sensor_env python -m pytest -q tests/test_segmentation.py
$env:PYTHONPATH="src"; conda run -n PPG_sensor_env ruff check src/ tests/
```

提交前运行完整测试；定位单个问题时先运行相关测试文件。Ruff 可用时用于静态检查。

## 编码风格与命名约定

Python 代码使用 4 空格缩进。新增公共函数应保留类型标注和简短 docstring。求解参数只在 `src/ppg_hr/params.py` 增加，并提供默认值，避免在 Notebook 或流程代码中散落硬编码默认参数。

函数、变量、模块名使用 `snake_case`；类、dataclass、Enum 使用 `PascalCase`。保留既有协议命名，例如 `ACC3`、`HF2_CF2`、`motion_only`、`rff_lms`。

## 测试指南

测试框架为 `pytest`。新增测试文件使用 `test_*.py` 命名，并放在 `tests/` 中与相关功能相邻。优先使用确定性的合成数据；涉及协议输出时，同时断言记录字段和生成文件。较大数据、临时输出和缓存不要提交。

## 数据、Notebook 与输出约定

当前 change6 入口只接受一个受试者目录：传感器 CSV 命名为 `{subject}_{motion}_{index}_sensor.csv`，参考心率 CSV 命名为 `{subject}_{motion}_{index}_HRdata.csv`；合法运动类型为 `write`、`gripper`、`run`、`rope`。标定读取 `SUBJECT_DIR.parent / "ck.mat"`。未配对文件记录原因，不进入训练；QC 坏样本保留标记并按回退策略继续计算。所有求解路径统一存窗口中心时间，误差和绘图通过时间偏移做对齐。

Notebook 路径使用仓库相对路径或 `PPG_SUBJECT_DIR` 环境变量；提交前清空运行输出、执行计数和本机绝对路径，训练、写图、回放和诊断开关恢复为 `False`。结果写入 `SUBJECT_DIR.parent / "outputs" / "{subject}__{run_signature}"`，保持模式级断点以支持续跑。较大流程变更后，同步更新 `notebooks/readme.md`。

## Git worktree 约定

后续新增 worktree 统一放在 `D:\python_notebook_base\.worktrees\<任务名>` 下，即使从已有 worktree 发起操作，也使用此固定目录；分支默认使用 `codex/` 前缀。创建前检查 `git worktree list`，优先复用适合当前任务的工作树，并用 `git check-ignore .worktrees/<任务名>` 确认目录已忽略。

根目录 `.gitignore` 中的 `/.worktrees/` 及备注需要提交；该目录及其内部文件只保留在本机，不纳入 Git 同步。工作树中的源码变更仍通过各自分支正常提交和推送。清理时使用 `git worktree remove`，先确认没有需要保留的未提交工作。

根目录 `AGENTS.md` 是受 Git 跟踪的项目开发守则，必须包含在后续克隆和 worktree 检出中。创建前确认所选基准已包含最新守则；旧分支或旧基准需先同步守则再开发。未提交的守则修改不会自动进入新工作树，应先提交，或显式同步到新工作树并纳入其后续提交。

## 提交与 PR 规范

日常提交信息使用中文，保持短小、明确、原子化，例如 `修复参考心率CSV解析`。不要使用 `--no-verify`。历史中也存在少量 `docs:` 前缀，可用于纯文档修改。

本次及后续经用户授权的 `AGENTS.md` 修改属于正常提交范围，相关守则变更与对应工作一并提交、推送；纯守则修改可单独提交。保持文件受 Git 跟踪，不将其加入忽略规则。

PR 面向 `upstream/change3`，说明改动目的、核心文件、已运行测试和 Notebook/输出影响。不确定某个本地改动是否应提交时，先确认再纳入提交。
