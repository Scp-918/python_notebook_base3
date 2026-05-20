# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目定位

PPG 心率求解算法 Python 实现，源码 `src/ppg_hr/`（含 `core/`、`preprocess/`、`experimental/` 三个子包），测试在 `tests/`，测试数据在 `testdata/`，主实验笔记本在 `notebooks/`。

## 语言与提交规范

- 始终使用中文交互，Git 提交信息使用中文，每次提交简要叙述修改内容。
- 高频原子化提交，严禁 `--no-verify`。

## 远程仓库

- `upstream`（原仓库）：`git@github.com:Scp-918/python_notebook_base3.git`（SSH fetch/push + HTTPS 备用 push）
- master 基于 `upstream/change3`，共享 git 历史，可直接发起 PR。

## PR 推送流程

日常开发在 master 上直接进行，无需新建分支：

```bash
# 1. 在 master 上正常开发和提交
git add <files>
git commit -m "描述改动"

# 2. 推送到自己的仓库
git push origin master

# 3. 向 upstream/change2 发起 PR
gh pr create --repo Scp-918/python_notebook_base3 --base change2 --head wqwshn:master \
    --title "PR 标题" --body "PR 描述"
```

注意：
- `notebooks/run_batch_adaptive_protocol.ipynb` 中的 `PROJECT_ROOT` 是本机路径，**严禁提交**。
- 不确定改动是否应提交时，先问再做。

## 环境与命令

- conda 环境 `PPG_sensor_env`，所有命令通过 `conda run -n PPG_sensor_env` 执行。
- 无 `setup.py`/`pyproject.toml`，通过 `sys.path.insert(0, src_dir)` 直接运行。

```bash
conda run -n PPG_sensor_env python -m pytest -q tests/           # 完整测试
conda run -n PPG_sensor_env python run_debug_protocol.py          # 冒烟测试
conda run -n PPG_sensor_env ruff check src/                        # 静态检查
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
