# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目定位

PPG 心率求解算法 Python 实现，源码 `src/ppg_hr/`（含 `core/`、`preprocess/`、`experimental/` 三个子包），测试在 `tests/`，测试数据在 `testdata/`，主实验笔记本在 `notebooks/`。

## 语言与提交规范

- 始终使用中文交互，Git 提交信息使用中文，每次提交简要叙述修改内容。
- 高频原子化提交，严禁 `--no-verify`。

## 远程仓库与推送流程

- `origin`（个人 fork）：`git@github.com:wqwshn/PPGtoHR_notebook.git`（SSH）和 `https://github.com/wqwshn/PPGtoHR_notebook.git`（HTTPS 备用 push）
- `upstream`（原仓库）：`git@github.com:Scp-918/python_notebook_base3.git`（SSH）和 `https://github.com/Scp-918/python_notebook_base3.git`（HTTPS 备用 push）
- fetch 走 SSH，push 按 SSH -> HTTPS 顺序自动回退。
- HTTP/HTTPS 代理：`http://127.0.0.1:7890`，SSH 不受代理影响。
- **推送铁律**：本地开发完成后，先 push 到 `origin`，再向 `upstream` 发起 Pull Request。

## 环境与命令

- conda 环境 `ppg-hr`，所有命令通过 `conda run -n ppg-hr` 执行。
- 无 `setup.py`/`pyproject.toml`，通过 `sys.path.insert(0, src_dir)` 直接运行。

```bash
conda run -n ppg-hr python -m pytest -q tests/           # 完整测试
conda run -n ppg-hr python run_debug_protocol.py          # 冒烟测试
conda run -n ppg-hr ruff check src/                        # 静态检查
```

## 设计约定

- `src/ppg_hr/params.py` 是所有求解参数的唯一数据源，新增参数必须有默认值。
- `testdata/` 不提交到 git。
- **QC 策略**：坏样本只标记不阻断后续计算。
- **时间对齐**：所有求解路径统一存窗口中心时间，误差和绘图通过 `time_bias` 做时移对齐。
- **文件命名**：传感器 CSV 为 `multi_<motion_type><index>.csv`，参考 CSV 为 `*_ref.csv`。合法运动类型：tiaosheng, wanju, fuwo, kaihe, bobi。

## 文档维护

- 较大改动后需同步更新 `notebooks/readme.md`。
- 向本文件新增硬规则必须先获得用户同意。
