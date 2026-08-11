# change6 批处理 Notebook

主入口为 `run_batch_adaptive_protocol.ipynb`。它只接受一个受试者目录，不再兼容
`multi_*` 数据。目录内文件必须严格成对：

```text
{subject}_{motion}_{index}_sensor.csv
{subject}_{motion}_{index}_HRdata.csv
motion = write | gripper | run | rope
```

受试者名可以包含下划线；motion 和 index 从文件名右侧解析。标定始终读取
`SUBJECT_DIR.parent / "ck.mat"`。

## 默认实验配置

- 数据拆分：`all_train`，同一 motion 的所有 index 共同构成一次 objective。
- 范围与目标：`motion_post10`、对齐后的 `posthoc_aae`。
- PPG 输入：`log_absorbance`。
- 自适应滤波：只启用 LMS；Volterra、RFF-LMS、KLMS 仍可显式选择。
- 搜索预算：每个 motion/scope/scheme/filter 模式 200 trials、1 repeat。
- scheme：`ACC`、`HF2`、`UD2`、`ACC+HF2`、`HF2+CF2`、`ACC+UD2`。
- motion 顺序：`write`、`gripper`、`run`、`rope`。

Notebook 首个代码单元提供 `legacy/enhanced` 总开关和六项独立开关：方向性追踪、
动态惩罚、连续性保护、低锁定恢复、高锁定恢复、运动后动态保护。默认使用
`enhanced` 且全部开启。这些字段会进入参数 JSON/CSV、cache key、模式 checkpoint
指纹和 replay 参数恢复。

## 推荐运行顺序

1. 设置环境变量 `PPG_SUBJECT_DIR`，或在首个单元修改 `SUBJECT_DIR`。
2. 运行“只读发现、严格配对与标定检查”。
3. 运行单对静态预处理，检查 CF2/HF2/HF2comp/UD2、序号 QC 与标定元数据。
4. 明确需要正式优化后，把 `RUN_TRAINING` 改为 `True`。

默认 `RUN_TRAINING=False`，因此静态检查不会创建 Optuna study，也不会生成或保存
滤波权重。训练仍按模式级保存 history、manifest 和 checkpoint，不增加权重级存档。
