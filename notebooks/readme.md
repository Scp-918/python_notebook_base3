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

增强模式按来源和阶段使用参考方向参数：普通 raw/baseline 路径保持独立的静息
追踪；adaptive 在运动段使用 `35/15 bpm` 搜索范围，在恢复段使用 `20/25 bpm`；
运动结束后 reset-FFT 会清空自身历史并独立启动。恢复期最初继续输出 adaptive，随后
依次检查 `stable_crossover`、`gap_rescue` 和 `adaptive_rising_rescue`。切换到
reset-FFT 后不再回跳；若没有足够证据，则保持 adaptive 到记录结束。

`motion_post10` 只表示默认 objective 评价运动结束后 10 秒内的窗口，不作为运动后保护的固定退出时间。
`POST_MOTION_GUARD_SECONDS=None` 是新默认；只有显式填写数值时才恢复旧的 timeout
兼容行为。PPG 完整候选峰门限为 `0.15`，运动惩罚参考峰门限为 `0.30`。

## 输出与断点目录

Notebook 和 Python 后端统一使用以下确定性目录：

```text
SUBJECT_DIR.parent/
└── outputs/
    └── {subject}__{run_signature}/
        ├── motion_types/              # 四类运动的 Stage-6 记录和模式级断点
        ├── qc/
        ├── signal_figures/
        ├── final_summary/
        ├── replay/
        ├── window_diagnostics/
        ├── summary_tables/
        ├── cross_replay/
        ├── cross_window_diagnostics/
        ├── batch_reference_compare/
        └── batch_target_hr_curves/   # 单 scheme、全 motion/index 的目标段 HR 图与 manifest
```

`run_signature` 包含 scope、scheme、filter、objective、split、`TW_F` 和 HR 后处理方式。
重复使用相同配置时路径保持不变，因此可以恢复模式级 checkpoint。默认
`CLEAN_OUTPUTS=False`；不要在需要续跑时打开清理开关。

正式训练通过 Notebook 的精简回调在每个“motion × scope × scheme × filter”完成后
打印一次结果，包括 final 的 post-hoc AAE/±5 bpm 准确率、同一套最优参数改用 ACC
信号源后的对照指标，以及该模式优化与最终复算耗时。恢复断点时会明确标记为断点恢复，
不会伪装成本次重新训练。

## Notebook 章节

- 第 0 节：四层配置，包括核心路径、预处理/对齐、滤波/追踪和执行/输出。
- 第 1–2 节：严格配对、`ck.mat`、静态 QC、单样本预处理、分段和对齐预览。
- 第 3–4、6 节：Tdelay、全段 PPG-HR 和原始 PPG 双轴诊断图。
- 第 5 节：统一训练包装器。
- 第 7 节：正式 `all_train` 优化。
- 第 8 节：按当前组合动态计算模式数的连通性检查。
- 第 9 节：Stage-6、history 和 checkpoint 完整性检查。
- 第 10–12 节：单样本 replay、窗口诊断和跨运动汇总。
- 第 13–15 节：跨 scheme replay/窗口诊断和批量参考通道对比。
- 第 16 节：读取既有 Stage-6 最优参数，批量重绘当前受试者全部 motion/index 的
  post-hoc 对齐真实 HR—final HR 目标段曲线；输出目录会追加 scope、scheme、filter、
  split 和 `TW_F` 模式签名，不会启动 Optuna。

replay 和诊断通过 `motion_type + motion_index` 从当前 `SUBJECT_DIR` 的配对结果选择
样本，不再填写独立的传感器或 HR CSV 路径。

## 推荐运行顺序

1. 设置环境变量 `PPG_SUBJECT_DIR`，或在首个单元修改 `SUBJECT_DIR`。
2. 运行“只读发现、严格配对与标定检查”。
3. 运行单对静态预处理，检查 CF2/HF2/HF2comp/UD2、序号 QC 与标定元数据。
4. 如需训练，先核对 `RUN_OUTPUT_DIR`，再把 `RUN_ALL_TRAIN` 改为 `True`。
5. 训练完成后，按需打开第 9–16 节各自的 `RUN_*` 开关。

默认所有训练、写图、回放和诊断开关均为 `False`，因此顺序执行 Notebook 不会创建
Optuna study 或输出目录。训练仍只按模式保存 history、manifest 和 checkpoint，
不增加滤波权重级存档。
