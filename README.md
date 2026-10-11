# Physical4D 运动伪影补偿与心率复现

Notebook 入口为 `notebooks/run_batch_adaptive_protocol.ipynb`，生成脚本为 `scripts/update_change6_notebook.py`。训练使用统一 reference-parity 求解器、样本级冻结分段/Tdelay、确定性300点网格和 SQLite 逐cell账本。正式基线 ACC+LMS；复杂恢复机制默认关闭。参考实现锁定 `wqwshn/outline-PPGtoHR` 的 `02d7e091219aeff78e82d53ae0069acf7bedb467`，非线性更新核保留本项目已确认行为。

## 环境与输入

使用 conda `PPG_sensor_env`，命令显式设置 `PYTHONPATH=src`。输入为 `{subject}_{motion}_{index}_sensor.csv` 与对应 `_HRdata.csv`，运动为 write/gripper/run/rope；标定位于受试者父目录 `ck.mat`。未配对文件记录原因；QC异常保留回退。Notebook用仓库相对路径或 `PPG_SUBJECT_DIR`，全部运行开关初始 False。

```powershell
$env:PYTHONPATH='src'
conda run -n PPG_sensor_env python -m pytest -q tests/
conda run -n PPG_sensor_env ruff check src/ tests/
conda run -n PPG_sensor_env python scripts/update_change6_notebook.py
```

## 训练入口与固定配置

```python
from pathlib import Path
from ppg_hr.params import reference_parity_params
from ppg_hr.experimental.physical4d_run import run_physical4d

roots = run_physical4d(
    subjects=[Path('testdata/data/ts')], motions=['write', 'run'],
    references=[('ACC',), ('ACC', 'HF')],
    adaptive_filters=['lms', 'volterra'], scopes=['motion_post10'],
    fixed_params=reference_parity_params(), output_parent=Path('outputs'))
```

subjects、motions、reference组合、filter、scope均支持单选或多选。外层组合各有独立run_root，每个样本各执行300点。根目录仅含五项，例如 `S01-S02__wr__AH__L__mp10`。运动编码w/g/r/p，reference A/H/C/U保留组顺序，filter L/V/R/K，scope mo/mr/mp10/gl。默认输出到共同数据父目录的outputs；不同父目录必须指定output_parent。同名目录配置身份不同会拒绝续写；不同policy可用不同输出父目录。

首次预处理前写入并冻结 `config/fixed_hyperparameters.json`，完整保存预处理、detector、对齐、级联、全部滤波器、追踪/融合/恢复、评价及数值控制，包括关闭机制和未选滤波器。只剔除当前滤波器实际搜索的四维；网格与派生值保存在manifest及SQLite coordinates。`parameter_catalog.json`描述类型、单位、可选值、启用条件、参考/采用值与差异；`provenance.json`记录schema、实现身份、Git/参考提交和依赖版本。数值快照确定性hash进入manifest、cell、缓存、Excel和receipt。说明文字不进入数值hash。续跑与复现拒绝缺字段、未知字段、篡改、不支持schema和实现身份不符，不能用当前默认值填补。

## 算法约定

source-domain ACC/GYRO检测保持原阈值、bridge/min-run与bout选择。先计算一次分段，从首次检测运动前的片段生成完整8秒静息窗搜索Tdelay；无运动使用全程。默认[-5,5]秒、0.1秒步长，AAE、绝对时延、RMSE、候选顺序依次处理平局；少于2可比较窗回退0并记录原因。不同Fs复用同一分段和Tdelay。sensor/窗口/边界不移动，Tdelay后评价为 `HR_ref(t+Tdelay_s)`，posthoc后为 `HR_ref(t+Tdelay_s+posthoc_bias_s)`；reference绘图时间分别减去偏移。posthoc默认[-5,5]秒、1秒步长，是独立oracle诊断。

300点固定顺序为Fs→memory→step→penalty半宽，编号p4d_000…299，最后一维最快。Fs=25/50/100Hz，memory=40/80/120/160/200ms，半宽=3/6/12/18bpm；`max_order=round(Fs*memory/1000)`，Hz半宽=bpm/60。

| 滤波器 | 五点步长 |
|---|---|
| LMS | .006/.008/.010/.012/.016 |
| Volterra | .002/.004/.008/.012/.016 |
| RFF-LMS | .001/.002/.004/.006/.008 |
| KLMS | .005/.010/.020/.050/.100 |

每窗输入PPG对全部reference做direct correlation ±200ms lag搜索，按最大绝对相关全局排序，同分按组/通道配置顺序。最强reference决定shared delay和penalty reference：delay非负M=1，负值M=min(max(1,floor(abs(delay))),max_order)。所有通道共享M；ACC K=1，HF/CF/UD K=0；envelope为可选实验模式。固定smooth=5、penalty weight=.4、M_base=1、C_scale=1、K_max=16，观察窗8秒/hop1秒，prewarm0。

LMS默认步长 `max(1e-6,mu_base-abs(corr)/100)`；非线性默认关闭相关缩放。Volterra M2=3、alpha_u=.1，保留归一化更新；RFF D100、seed42、自适应robust带宽、sigma_scale1、NLMS eps1e-9，保留与参考固定sigma=1的差异；KLMS sigma1、epsilon.1、归一化距离/更新、字典300、freeze_new_centers、eps1e-6，网格步长不采用参考.2。差异进入参数目录。

`spectral_penalty_policy='soft_visible'` 默认保持候选可见并开启continuity protection；`suppressed_protected_continuous_visibility`实际接通候选排除和保护owner。owner匹配半宽3.5bpm、其后最多3次motion调用、边界保护1bpm；获取owner当窗仍选challenger，仅匹配owned target可越过排除，mismatch/band exit/expiry/recovery/EOF终止并记录。policy名称/版本/阈值进入固定身份，不参与网格，advanced visibility由policy派生，不开启其他rescue。

rise lineage、low/high-lock、post-motion recovery/guard、dual-reset均可在同一框架显式启用，默认关闭。每cell重置全部状态，在线逻辑仅依赖sensor/频谱/历史，不读取参考HR。Final优先dual-reset、guard、简化融合，随后仅一次final dynamics。恢复期保留reset-FFT与简化handoff。旧legacy入口为历史兼容，新训练不使用BO或旧冻结preset。

## 账本、输出与续跑

```text
<run_root>/
  config/{fixed_hyperparameters,parameter_catalog,provenance}.json
  manifest.json
  run.sqlite
  S01/
    align/<sample>__tdelay.png
    signal/<sample>__fft_signal.png
    grid/<sample>__p4d.xlsx
    summary/{sample_best,motion_summary}.xlsx
    replay/<sample>__<coordinate_id>/
  replay/batches/<selection_hash>/
  experiments/
  overall_summary.xlsx
```

SQLite含samples/coordinates/cell_metrics/attempt_events；cell主键(sample_id,coordinate_id)。每cell成功后即时事务写标量、完整参数审计和身份，技术失败仅记attempt/error，下次补缺；无有效窗记NULL/unscored。300点均已完成才sample_complete，重启优先跳完整样本。每次调用每个缺点尝试一次；最多保留当前样本当前Fs的准备缓存，跨Fs及样本清缓存/GC。

Tdelay图含搜索、选定点、静息HR对齐前后、有效窗与fallback；搜索/静息曲线也入SQLite。信号图含原始PPG、reference、源IMU及基础FFT并标识处理状态。P4D始终300行，包括缺失/失败，保存四维、派生值、两个偏移、两组AAE/accuracy、有效窗、状态、身份和参数审计。best仅完整样本有限posthoc final AAE，平局coordinate_id；运动与总体汇总保留样本数、完成/失败/可评分数、均值和样本标准差。同个体所有运动共享库与汇总，无运动级重复目录。

`repair_run_outputs(root)`补图与导出，不重跑网格；`rebuild_summaries(ledger)`可仅从只读SQLite重建汇总。

## 严格复现与独立实验

```python
from ppg_hr.experimental.physical4d_replay import resolve_replay_spec, replay_sample, replay_batch
spec = resolve_replay_spec(roots[0], 'ts_write_1', 'best_posthoc')
print(spec.preview())
receipt = replay_sample(spec)
report = replay_batch(roots[0], motions=['write'], selection_callback=print)
```

ReplaySpec合并保存固定快照与样本自身坐标，核对cell完整审计，恢复冻结分段/Tdelay/bias及输入内容身份。可映射数据目录，但sensor、HR与标定内容必须相同。复现从样本起点保持追踪历史，不运行detector、Tdelay、posthoc或网格搜索；保存bias无效保持unscored。指标绝对容差1e-6、有效窗数严格一致。单样本生成双面板HR、完整windows.csv、metrics.json、receipt.json，scope指标与绘图裁剪独立。指定窗口仅收集该窗完整stage并保留此前历史。

批量先冻结逐样本坐标清单，串行重算/立即保存receipt/释放内存；verified且身份及文件hash完整才cached_verified，失败与metric_mismatch继续后续样本。默认best跳过未完成或无有限best。汇总只统计verified/cached_verified，保留failed/skipped计数；训练数据库只读。

跨reference实验固定源样本自身坐标/分段/Tdelay，默认保存bias，显式重搜时另记实验bias。独立配置身份和experiments目录，不覆盖训练或精确复现。Notebook第10–16节依次为单样本、窗口诊断、汇总、reference实验/诊断、逐样本reference批对比、批量严格复现。详细操作见 `notebooks/readme.md`。

实现身份按算法及schema的AST计算，忽略参数默认值、参数目录说明、注释和格式；仅更改当前默认值不会阻止既有快照复现。字段/schema或实际计算代码改变仍会被一致性检查识别。参数类型依据声明校验，包含可空参数，恢复时不借用当前默认值。
