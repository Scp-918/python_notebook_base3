# PPG 心率批量实验

本仓库以 Notebook 为入口比较运动伪影补偿与心率误差。冻结核心 `core_enhanced_v1` 提供可解释的算法骨架，方便后续接入 Udeco interface-quality；当前不通过复杂 rescue 补指标。

## 冻结核心与模块

| 层 | 职责 | 核心文件 |
|---|---|---|
| 原始检测 / 协议 | 100Hz ACC+Gyro detector保留全部runs，single_bout/multi_bout独立选择 | `segmentation.py` |
| 1. Candidate spectrum | FFT或SSR产生未惩罚频谱/候选，记录实际estimator及fallback | `spectral_utils.py` |
| 2. Candidate tracker | 共享soft penalty、continuity protection、directional tracking与第一次slew | `penalty_policy.py`, `enhanced_tracking.py` |
| 3. Source selection | rest baseline、motion adaptive/QC回退、恢复期simple adaptive→reset-FFT handoff | `fusion.py` |
| 4. Final dynamics | 按rest/motion/recovery限制拼接后的Final；状态跨phase延续 | `fusion.py` |

上述文件位于 `src/ppg_hr/experimental/`；参数、reference registry和不可变preset快照集中在 `src/ppg_hr/params.py`。`cascade_solver.py` 组织完整求解，`run_batch_protocol.py` 管理batch、记录、断点、回放与图表。

显式开启：raw-source detector、8秒/1秒observation、adaptive filtering、共享soft penalty、continuity protection、directional candidate tracker、simple handoff、phase dynamics。显式关闭：low/high-lock、rise-lineage、gap/rising rescue、complex shadow、复杂post-motion rescue及interface-quality在线介入。实验机制模块仍可单独研究，冻结入口拒绝打开它们。

所有窗口记录start/center/end；phase、scope与handoff使用center。detector六轴四阶零相位带通（ACC .5–5Hz，Gyro .5–10Hz），分别计算magnitude；8/1秒STD(ddof=1)，前30秒baseline，阈值 `max(2.5*baseline_std, .05*max_window_score)`，任一路严格超过即motion。先桥接≤3窗内部false gap，再删除<5窗true run。

恢复handoff等待5秒，adaptive与reset-FFT差值≤6bpm连续2个有效窗切换，最迟20秒切到有效reset-FFT；motion_only在恢复立即使用reset-FFT，motion_post10最迟在adaptive范围结束时切换。Final的up/down limit/step（bpm）：rest 1.5/1.5与3/1.5，motion 5.5/3.5与2/1.5，recovery 1.5/1.5与3.5/3。

## 配置与搜索

```python
from ppg_hr.params import resolve_core_config, core_preset_metadata, ReferenceConfig
params = resolve_core_config()  # 固定验收：100Hz、160ms、LMS步长0.01
metadata = core_preset_metadata(params)
reference_groups_order = ("ACC", "HF")
reference = ReferenceConfig(reference_groups_order)
# reference_sequence: accx,accy,accz,hf1,hf2；reference_order_key: ACC+HF
```

只配置ordered groups；sequence和canonical key由registry派生。支持ACC/HF/CF/UD：`("UD",)`为单组，`("HF","ACC")`顺序与ACC+HF不同。路径、mode、CSV/JSON、cache、legend、summary和replay统一使用canonical key。CSV物理采集字段UD2仍是第二路电压推导通道，不是旧reference模式名称。

BO仅四维：Fs_Target `[25,50]`、memory_ms `[40,80,120,160,200]`、滤波器步长、Spec_Penalty_Width `[.1,.2,.3]`Hz。LMS/Volterra/RFF使用LMS_Mu_Base，KLMS使用klms_step_size；步长候选集中在params.py。Spec_Penalty_Weight=.2是保留比例，冻结；confidence共用、width为Hz半宽。HR窗8/1、tracker/handoff/dynamics及其他滤波器参数均冻结。memory只派生正延迟history上限，M/K仍由时延映射，K_max不因Fs自动物理化。prewarm_s=0是独立上下文，不控制HR窗。

## 数据血缘与评价

`ppg_acquired_raw → ppg_clean → ppg_algorithm_input`：采集值保留source网格，序号缺口NaN，毛刺不替换；clean承担missing/spike repair；algorithm input承担bandpass/log absorbance和目标域resample。raw Uh/Uc与标定生成独立UD calculation，一路形成adaptive reference的预处理副本，另一路与原始PPG、source时间、QC组成InterfaceQualityInput。quality算法尚未实现，没有adaptive output或reference HR依赖。

参考HR仅用于完成预测后的评价与绘图，不平移sensor、不裁剪预测窗，不参与tracker/quality/handoff/fusion。零相位、全记录阈值与前视滤波意味着本实验不声称严格实时因果。新旧AAE比较需注明detector、评价对齐和预测覆盖范围变化。

## 运行与验证

严格输入为受试者目录中的 `{subject}_{motion}_{index}_sensor.csv`（Pydisplay）和标准 `{subject}_{motion}_{index}_HRdata.csv`，motion为write/gripper/run/rope；标定在父目录ck.mat。没有旧loader/replay兼容层，旧结果需重新运行，新输出保存schema、preset/version、完整参数、flags及fingerprint。

主入口 [Notebook](notebooks/run_batch_adaptive_protocol.ipynb)，完整配置、固定验收、batch/resume/replay/summary/绘图步骤见 [notebooks/readme.md](notebooks/readme.md)。结果写入受试者父目录outputs，不提交生成物。

```powershell
$env:PYTHONPATH="src"
conda run -n PPG_sensor_env python -m pytest -q tests/
conda run -n PPG_sensor_env ruff check src/ tests/
conda run -n PPG_sensor_env python scripts/update_change6_notebook.py
```

若直接调用环境python.exe，PATH须包含该环境根目录、Scripts与Library/bin以加载NumPy/SciPy DLL；pytest可指定 `--basetemp=outputs/pytest-temp`。Notebook与生成脚本同步更新；提交前清空输出/执行计数，所有训练、写图、回放和诊断开关恢复False。开发守则见 [AGENTS.md](AGENTS.md)。
