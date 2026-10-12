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

## 完整固定超参数目录

下表列出 `reference_parity_params()` 的固定参数默认采用值、类型、单位、可选项、启用条件和参考差异，包含未选滤波器与默认关闭的机制。实际训练以每个 run_root 的 `config/fixed_hyperparameters.json` 为准；复现必须读取该快照，不以本表或当前代码默认值补缺。Physical4D 的四个搜索维度及派生值见下文“算法约定”。

LMS 使用 `max(1e-6, mu_base−abs(corr)/100)`；其他滤波器默认不做 correlation 缩放。Volterra 保留归一化线性/二阶更新；RFF 保留 robust 带宽与 NLMS（参考为固定 sigma=1）；KLMS 保留归一化距离、字典上限 300 及 freeze_new_centers。

| 固定参数 | 类型/单位 | 默认采用值 | 可选项 | 启用条件 | 说明/参考差异 |
|---|---|---|---|---|---|
| effective_mechanisms.advanced_visibility | bool/dimensionless | False | [False, True] | 固定记录；关闭机制或未选滤波器仍保留参数 | effective_mechanisms：advanced_visibility；实际采用值见adopted_value。 |
| effective_mechanisms.enable_continuity_protection | bool/dimensionless | True | [False, True] | 固定记录；关闭机制或未选滤波器仍保留参数 | effective_mechanisms：enable_continuity_protection；实际采用值见adopted_value。 |
| effective_mechanisms.enable_directional_tracking | bool/dimensionless | True | [False, True] | 固定记录；关闭机制或未选滤波器仍保留参数 | effective_mechanisms：enable_directional_tracking；实际采用值见adopted_value。 |
| effective_mechanisms.enable_dual_reset | bool/dimensionless | False | [False, True] | 固定记录；关闭机制或未选滤波器仍保留参数 | 启用纯PPG与Final弱先验两条独立reset轨迹及可信目标交接，默认关闭。 |
| effective_mechanisms.enable_dynamic_penalty | bool/dimensionless | True | [False, True] | 固定记录；关闭机制或未选滤波器仍保留参数 | effective_mechanisms：enable_dynamic_penalty；实际采用值见adopted_value。 |
| effective_mechanisms.enable_high_lock_recovery | bool/dimensionless | False | [False, True] | 固定记录；关闭机制或未选滤波器仍保留参数 | effective_mechanisms：enable_high_lock_recovery；实际采用值见adopted_value。 |
| effective_mechanisms.enable_low_lock_recovery | bool/dimensionless | False | [False, True] | 固定记录；关闭机制或未选滤波器仍保留参数 | effective_mechanisms：enable_low_lock_recovery；实际采用值见adopted_value。 |
| effective_mechanisms.enable_post_motion_guard | bool/dimensionless | False | [False, True] | 固定记录；关闭机制或未选滤波器仍保留参数 | 启用动态运动后adaptive/reset-FFT来源保护与交接，默认关闭。 |
| effective_mechanisms.enable_post_motion_protection | bool/dimensionless | False | [False, True] | 固定记录；关闭机制或未选滤波器仍保留参数 | effective_mechanisms：enable_post_motion_protection；实际采用值见adopted_value。 |
| effective_mechanisms.enable_post_motion_recovery | bool/dimensionless | False | [False, True] | 固定记录；关闭机制或未选滤波器仍保留参数 | 进入recovery时清空adaptive追踪历史；baseline仍延续，默认关闭。 |
| effective_mechanisms.enable_rise_lineage | bool/dimensionless | False | [False, True] | 固定记录；关闭机制或未选滤波器仍保留参数 | 启用有期限的上升候选谱峰lineage确认及重锚，默认关闭。 |
| evaluation.accuracy_threshold_bpm | float/bpm | 5.0 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | 绝对HR误差不超过该bpm阈值计为准确；评价scope由manifest固定。 |
| evaluation.alignment_fallback | str/dimensionless | zero_delay_continue | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | 比较窗口不足时使用Tdelay=0，记录原因后继续计算。 |
| evaluation.alignment_min_windows | int/windows | 2 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | Tdelay候选至少需要的有效静息比较窗口数。 |
| evaluation.alignment_tie_rule | str/dimensionless | aae_abs_delay_rmse_candidate_order | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | 固定AAE、绝对时延、RMSE、候选顺序依次处理平局。 |
| evaluation.replay_metric_atol | float/dimensionless | 1e-06 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | 保存指标与复算指标的绝对容差；有效窗口数仍须完全相同。 |
| evaluation.tdelay_range_s | list/s | [-5.0, 5.0] | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | 样本唯一固定系统时延的候选范围，秒。 |
| evaluation.tdelay_step_s | float/s | 0.1 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | 样本Tdelay搜索步长，秒。 |
| execution.reference_groups_order | list/dimensionless | ['ACC'] | ['ACC', 'HF', 'CF', 'UD'] | 固定记录；关闭机制或未选滤波器仍保留参数 | execution：reference_groups_order；实际采用值见adopted_value。 |
| execution.target_scope | str/dimensionless | motion_post10 | ['motion_only', 'motion_recovery', 'motion_post10', 'global'] | 固定记录；关闭机制或未选滤波器仍保留参数 | execution：target_scope；实际采用值见adopted_value。 |
| motion_detector.acc_band_hz | list/Hz | [0.5, 5.0] | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | motion_detector：acc_band_hz；实际采用值见adopted_value。 |
| motion_detector.baseline_multiplier | float/dimensionless | 2.5 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | motion_detector：baseline_multiplier；实际采用值见adopted_value。 |
| motion_detector.baseline_s | float/s | 30.0 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | motion_detector：baseline_s；实际采用值见adopted_value。 |
| motion_detector.bridge_gap_windows | int/windows | 3 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | motion_detector：bridge_gap_windows；实际采用值见adopted_value。 |
| motion_detector.filter_order | int/dimensionless | 4 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | motion_detector：filter_order；实际采用值见adopted_value。 |
| motion_detector.gyro_band_hz | list/Hz | [0.5, 10.0] | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | motion_detector：gyro_band_hz；实际采用值见adopted_value。 |
| motion_detector.minimum_run_windows | int/windows | 5 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | motion_detector：minimum_run_windows；实际采用值见adopted_value。 |
| motion_detector.peak_fraction | float/dimensionless | 0.05 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | motion_detector：peak_fraction；实际采用值见adopted_value。 |
| motion_detector.step_s | float/s | 1.0 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | motion_detector：step_s；实际采用值见adopted_value。 |
| motion_detector.window_s | float/s | 8.0 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | motion_detector：window_s；实际采用值见adopted_value。 |
| numerical.candidate_band_hz | list/Hz | [0.5, 4.0] | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | numerical：candidate_band_hz；实际采用值见adopted_value。 |
| numerical.dual_reset_raw_band_hz | list/Hz | [0.7, 4.0] | 按类型显式配置 | enable_dual_reset | dual-reset使用的原始FFT候选合法频带，默认0.7–4Hz。 |
| numerical.dual_reset_raw_persistence_windows | int/windows | 3 | 按类型显式配置 | enable_dual_reset | numerical：dual_reset_raw_persistence_windows；实际采用值见adopted_value。 |
| numerical.fft_min_length | int/dimensionless | 8192 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | FFT最短长度；实际取不小于当前信号长度的2次幂。 |
| numerical.high_lock_risk_near_motion_bpm | float/bpm | 8.0 | 按类型显式配置 | enable_high_lock_recovery | numerical：high_lock_risk_near_motion_bpm；实际采用值见adopted_value。 |
| numerical.input_zscore_std_eps | float/dimensionless | 1e-12 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | numerical：input_zscore_std_eps；实际采用值见adopted_value。 |
| numerical.klms_min_sigma | float/dimensionless | 1e-12 | 按类型显式配置 | adaptive_filter == klms | numerical：klms_min_sigma；实际采用值见adopted_value。 |
| numerical.lms_nlms_eps | float/dimensionless | 1e-09 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | numerical：lms_nlms_eps；实际采用值见adopted_value。 |
| numerical.penalty_challenger_min_amp_ratio | float/dimensionless | 0.45 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | numerical：penalty_challenger_min_amp_ratio；实际采用值见adopted_value。 |
| numerical.penalty_confidence_floor | float/dimensionless | 0.9 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | numerical：penalty_confidence_floor；实际采用值见adopted_value。 |
| numerical.penalty_harmonic_min_half_width_hz | float/Hz | 0.016666666666666666 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | numerical：penalty_harmonic_min_half_width_hz；实际采用值见adopted_value。 |
| numerical.penalty_reference_band_hz | list/Hz | [0.2, 5.0] | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | numerical：penalty_reference_band_hz；实际采用值见adopted_value。 |
| numerical.posthoc_min_valid_windows | int/windows | 2 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | oracle bias候选至少需要的有效比较窗口数，默认2。 |
| numerical.rff_max_distance_pairs | int/dimensionless | 2048 | 按类型显式配置 | adaptive_filter == rff_lms | numerical：rff_max_distance_pairs；实际采用值见adopted_value。 |
| numerical.rff_min_sigma | float/dimensionless | 1e-06 | 按类型显式配置 | adaptive_filter == rff_lms | numerical：rff_min_sigma；实际采用值见adopted_value。 |
| numerical.volterra_alpha_floor | float/dimensionless | 1e-06 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | numerical：volterra_alpha_floor；实际采用值见adopted_value。 |
| numerical.volterra_nlms_eps | float/dimensionless | 1e-09 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | numerical：volterra_nlms_eps；实际采用值见adopted_value。 |
| numerical.window_normalization_eps | float/dimensionless | 1e-12 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | numerical：window_normalization_eps；实际采用值见adopted_value。 |
| preprocessing.bandpass_pad_multiplier | int/dimensionless | 3 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | preprocessing：bandpass_pad_multiplier；实际采用值见adopted_value。 |
| preprocessing.cutoff_nyquist_fraction | float/dimensionless | 0.45 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | preprocessing：cutoff_nyquist_fraction；实际采用值见adopted_value。 |
| preprocessing.filter_order | int/dimensionless | 4 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | preprocessing：filter_order；实际采用值见adopted_value。 |
| preprocessing.fs_origin | int/Hz | 100 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | 原始采样率(Hz)；change6/Pydisplay 数据固定按 100 Hz 与序号校验。 |
| preprocessing.imu_band_hz | list/Hz | [0.5, 10.0] | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | preprocessing：imu_band_hz；实际采用值见adopted_value。 |
| preprocessing.minimum_bandpass_samples | int/dimensionless | 8 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | preprocessing：minimum_bandpass_samples；实际采用值见adopted_value。 |
| preprocessing.minimum_cutoff_hz | float/Hz | 0.001 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | preprocessing：minimum_cutoff_hz；实际采用值见adopted_value。 |
| preprocessing.missing_repair | str/dimensionless | linear_interpolation_edge_hold | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | preprocessing：missing_repair；实际采用值见adopted_value。 |
| preprocessing.outlier_mad_scale | float/dimensionless | 1.4826 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | MAD转标准差的一致性系数；参考正态分布1.4826。 |
| preprocessing.outlier_minimum_samples | int/dimensionless | 3 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | PPG移动中位数检测的最小样本数。 |
| preprocessing.outlier_threshold | float/dimensionless | 3.0 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | PPG毛刺检测阈值：局部偏差超过该倍数scaled MAD才修复。 |
| preprocessing.outlier_window_s | float/s | 1.0 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | preprocessing：outlier_window_s；实际采用值见adopted_value。 |
| preprocessing.ppg_band_hz | list/Hz | [0.5, 5.0] | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | preprocessing：ppg_band_hz；实际采用值见adopted_value。 |
| preprocessing.thermal_band_hz | list/Hz | [0.1, 5.0] | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | preprocessing：thermal_band_hz；实际采用值见adopted_value。 |
| preprocessing.thermal_voltage_scale | float/dimensionless | 1000.0 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | preprocessing：thermal_voltage_scale；实际采用值见adopted_value。 |
| preprocessing.ud_calculation_mode | str/dimensionless | smoothed | ['raw', 'smoothed'] | 固定记录；关闭机制或未选滤波器仍保留参数 | UD 公式输入模式；raw 直接使用原始 Uh/Uc，smoothed 先分别平滑四路电压再计算。 |
| preprocessing.ud_denominator_eps | float/dimensionless | 1e-09 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | preprocessing：ud_denominator_eps；实际采用值见adopted_value。 |
| preprocessing.ud_smoothing_window_s | float/s | 0.1 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | 模式 smoothed 的中心移动中位数窗口(秒)；100 Hz 下 0.1 秒约为 10 点。 |
| trial.C_scale | float/dimensionless | 1.0 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | 参考固定缩放1；reference-parity的M严格按floor(abs(shared delay))。 |
| trial.Enable_Time_Bias_After | bool/dimensionless | True | [False, True] | 固定记录；关闭机制或未选滤波器仍保留参数 | 是否计算滤波后 post-hoc 时间偏移诊断。 |
| trial.K_max | int/dimensionless | 16 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | 兼容参数固定16；reference-parity实际K为ACC=1，HF/CF/UD=0。 |
| trial.Kstop | float/dimensionless | 0.3 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | trial：Kstop；实际采用值见adopted_value。 |
| trial.LMS_Mu_Min | float/dimensionless | 1e-06 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | trial：LMS_Mu_Min；实际采用值见adopted_value。 |
| trial.M2 | int/dimensionless | 3 | 按类型显式配置 | adaptive_filter == volterra | trial：M2；实际采用值见adopted_value。；二阶长度一致；保留当前归一化线性/二阶更新。 |
| trial.M_base | int/dimensionless | 1 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | 参考规则基准历史tap数；主路径非负shared delay实际M=1。 |
| trial.Recovery_Cross_Diff_Bpm | float/bpm | 8.0 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | trial：Recovery_Cross_Diff_Bpm；实际采用值见adopted_value。 |
| trial.Recovery_Diff_Bpm | float/bpm | 20.0 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | trial：Recovery_Diff_Bpm；实际采用值见adopted_value。 |
| trial.Recovery_Grace_S | float/s | 10.0 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | trial：Recovery_Grace_S；实际采用值见adopted_value。 |
| trial.Rest_Alignment_Score_Mode | str/dimensionless | aae | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | Tdelay 评分指标；aae 为当前默认，std 用于兼容旧实验。 |
| trial.Rest_HR_Band_BPM | list/bpm | [40.0, 180.0] | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | 静息 HR 合法范围(bpm)；用于对齐阶段的频带限制。 |
| trial.Rest_HR_Peak_Percent | float/dimensionless | 0.3 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | 候选频谱峰相对主峰的最低幅值比例。 |
| trial.Rest_HR_Slew_Limit_BPM | float/bpm | 6.0 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | 静息 HR 允许直接接受的相邻变化阈值(bpm)。 |
| trial.Rest_HR_Slew_Step_BPM | float/bpm | 4.0 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | 超过静息变化阈值时每窗口最大修正步长(bpm)。 |
| trial.Rest_HR_Smooth_Method | str/dimensionless | median | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | 静息 HR 平滑方法；median 对孤立异常峰更稳健。 |
| trial.Rest_HR_Smooth_Win | int/windows | 3 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | 静息 HR 平滑窗口数；应使用正奇数以保持中心对齐。 |
| trial.Rest_HR_Spec_Penalty_Enable | bool/dimensionless | True | [False, True] | 固定记录；关闭机制或未选滤波器仍保留参数 | 是否在静息 HR 提取时启用运动频率惩罚。 |
| trial.Rest_HR_Spec_Penalty_Weight | float/dimensionless | 0.2 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | 静息HR提取的惩罚保留比例；保持基准值，独立于主路径0.4。 |
| trial.Rest_HR_Spec_Penalty_Width_Hz | float/Hz | 0.2 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | 静息运动峰惩罚半宽(Hz)；决定受抑制频带范围。 |
| trial.Rest_HR_Track_Band_BPM | float/bpm | 30.0 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | 只读冻结的诊断追踪带宽(bpm)，不参与Physical4D搜索。 |
| trial.SSR_Fallback_To_FFT | bool/dimensionless | True | [False, True] | 固定记录；关闭机制或未选滤波器仍保留参数 | trial：SSR_Fallback_To_FFT；实际采用值见adopted_value。 |
| trial.SSR_Grid_Resolution_BPM | float/bpm | 1.0 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | trial：SSR_Grid_Resolution_BPM；实际采用值见adopted_value。 |
| trial.SSR_Harmonic_Tol_BPM | float/bpm | 5.0 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | trial：SSR_Harmonic_Tol_BPM；实际采用值见adopted_value。 |
| trial.SSR_Lambda | float/dimensionless | 0.15 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | trial：SSR_Lambda；实际采用值见adopted_value。 |
| trial.SSR_Num_Atoms | int/dimensionless | 5 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | trial：SSR_Num_Atoms；实际采用值见adopted_value。 |
| trial.Spec_Penalty_Weight | float/dimensionless | 0.4 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | 运动频谱中心保留比例；固定0.4，越小抑制越强，不参与Physical4D。 |
| trial.Time_Bias_After_Mode | str/dimensionless | posthoc_oracle_alignment | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | post-hoc 对齐语义；oracle 只用于诊断指标，不参与部署选择。 |
| trial.Time_Bias_After_Range_S | list/s | [-5.0, 5.0] | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | post-hoc 时间偏移搜索区间(秒)，按(最小,最大)设置。 |
| trial.Time_Bias_After_Step_S | float/s | 1.0 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | post-hoc 时间偏移搜索步长(秒)。 |
| trial.adaptive_filter | str/dimensionless | lms | ['lms', 'volterra', 'rff_lms', 'klms'] | 固定记录；关闭机制或未选滤波器仍保留参数 | trial：adaptive_filter；实际采用值见adopted_value。 |
| trial.alpha_u | float/dimensionless | 0.1 | 按类型显式配置 | adaptive_filter == volterra | trial：alpha_u；实际采用值见adopted_value。 |
| trial.candidate_peak_threshold_ratio | float/dimensionless | 0.3 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | 运动惩罚参考峰相对主峰门限；参考默认 0.30。 |
| trial.cascade_guard_flat_std_eps | float/dimensionless | 1e-06 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | trial：cascade_guard_flat_std_eps；实际采用值见adopted_value。 |
| trial.cascade_guard_policy | str/dimensionless | none | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | trial：cascade_guard_policy；实际采用值见adopted_value。 |
| trial.cascade_guard_ratio_max | float/dimensionless | 5.0 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | trial：cascade_guard_ratio_max；实际采用值见adopted_value。 |
| trial.cascade_guard_ratio_min | float/dimensionless | 0.05 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | trial：cascade_guard_ratio_min；实际采用值见adopted_value。 |
| trial.cascade_guard_use_finite_zscore | bool/dimensionless | True | [False, True] | 固定记录；关闭机制或未选滤波器仍保留参数 | trial：cascade_guard_use_finite_zscore；实际采用值见adopted_value。 |
| trial.cascade_mode | str/dimensionless | reference_parity | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | Physical4D固定reference_parity；legacy仅兼容旧调用，不能进入新训练。 |
| trial.correlation_scale_lms | bool/dimensionless | True | [False, True] | 固定记录；关闭机制或未选滤波器仍保留参数 | LMS默认从基础步长减去abs(correlation)/100并限制到mu_min。 |
| trial.correlation_scale_nonlinear | bool/dimensionless | False | [False, True] | 固定记录；关闭机制或未选滤波器仍保留参数 | 非线性滤波器可选相同相关缩放，默认关闭以保留现有内核。 |
| trial.delay_estimation_mode | str/dimensionless | direct | ['direct', 'envelope'] | 固定记录；关闭机制或未选滤波器仍保留参数 | direct为主路径；envelope为可选Hilbert包络相关实验。 |
| trial.dual_reset_confirm_windows | int/windows | 3 | 按类型显式配置 | enable_dual_reset | trial：dual_reset_confirm_windows；实际采用值见adopted_value。 |
| trial.dual_reset_controlled_reanchor | bool/dimensionless | True | [False, True] | enable_dual_reset | trial：dual_reset_controlled_reanchor；实际采用值见adopted_value。 |
| trial.dual_reset_gap_bpm | float/bpm | 6.0 | 按类型显式配置 | enable_dual_reset | trial：dual_reset_gap_bpm；实际采用值见adopted_value。 |
| trial.dual_reset_max_held_previous | int/dimensionless | 0 | 按类型显式配置 | enable_dual_reset | trial：dual_reset_max_held_previous；实际采用值见adopted_value。 |
| trial.dual_reset_mechanism | str/dimensionless | trend_persistence | 按类型显式配置 | enable_dual_reset | trial：dual_reset_mechanism；实际采用值见adopted_value。 |
| trial.dual_reset_min_amp_ratio | float/dimensionless | 0.25 | 按类型显式配置 | enable_dual_reset | trial：dual_reset_min_amp_ratio；实际采用值见adopted_value。 |
| trial.dual_reset_prior_half_life_s | float/s | 10.0 | 按类型显式配置 | enable_dual_reset | trial：dual_reset_prior_half_life_s；实际采用值见adopted_value。 |
| trial.dual_reset_prior_invalidation_enable | bool/dimensionless | False | [False, True] | enable_dual_reset | trial：dual_reset_prior_invalidation_enable；实际采用值见adopted_value。 |
| trial.dual_reset_prior_invalidation_gap_bpm | float/bpm | 40.0 | 按类型显式配置 | enable_dual_reset | trial：dual_reset_prior_invalidation_gap_bpm；实际采用值见adopted_value。 |
| trial.dual_reset_prior_invalidation_hits | int/windows | 3 | 按类型显式配置 | enable_dual_reset | trial：dual_reset_prior_invalidation_hits；实际采用值见adopted_value。 |
| trial.dual_reset_prior_invalidation_prior_decline_bpm | float/bpm | 0.5 | 按类型显式配置 | enable_dual_reset | trial：dual_reset_prior_invalidation_prior_decline_bpm；实际采用值见adopted_value。 |
| trial.dual_reset_prior_invalidation_raw_decline_bpm | float/bpm | 0.5 | 按类型显式配置 | enable_dual_reset | trial：dual_reset_prior_invalidation_raw_decline_bpm；实际采用值见adopted_value。 |
| trial.dual_reset_qualification_windows | int/windows | 4 | 按类型显式配置 | enable_dual_reset | trial：dual_reset_qualification_windows；实际采用值见adopted_value。 |
| trial.dual_reset_readiness_hits | int/windows | 2 | 按类型显式配置 | enable_dual_reset | trial：dual_reset_readiness_hits；实际采用值见adopted_value。 |
| trial.dual_reset_reanchor_min_gap_bpm | float/bpm | 25.0 | 按类型显式配置 | enable_dual_reset | trial：dual_reset_reanchor_min_gap_bpm；实际采用值见adopted_value。 |
| trial.dual_reset_reanchor_prior_guard_bpm | float/bpm | 45.0 | 按类型显式配置 | enable_dual_reset | trial：dual_reset_reanchor_prior_guard_bpm；实际采用值见adopted_value。 |
| trial.dual_reset_trajectory_tolerance_bpm | float/bpm | 6.0 | 按类型显式配置 | enable_dual_reset | trial：dual_reset_trajectory_tolerance_bpm；实际采用值见adopted_value。 |
| trial.enable_continuity_protection | bool/dimensionless | True | [False, True] | 固定记录；关闭机制或未选滤波器仍保留参数 | trial：enable_continuity_protection；实际采用值见adopted_value。 |
| trial.enable_directional_tracking | bool/dimensionless | True | [False, True] | 固定记录；关闭机制或未选滤波器仍保留参数 | trial：enable_directional_tracking；实际采用值见adopted_value。 |
| trial.enable_dual_reset | bool/dimensionless | False | [False, True] | 固定记录；关闭机制或未选滤波器仍保留参数 | 启用纯PPG与Final弱先验两条独立reset轨迹及可信目标交接，默认关闭。 |
| trial.enable_dynamic_penalty | bool/dimensionless | True | [False, True] | 固定记录；关闭机制或未选滤波器仍保留参数 | trial：enable_dynamic_penalty；实际采用值见adopted_value。 |
| trial.enable_high_lock_recovery | bool/dimensionless | False | [False, True] | 固定记录；关闭机制或未选滤波器仍保留参数 | trial：enable_high_lock_recovery；实际采用值见adopted_value。 |
| trial.enable_low_lock_recovery | bool/dimensionless | False | [False, True] | 固定记录；关闭机制或未选滤波器仍保留参数 | trial：enable_low_lock_recovery；实际采用值见adopted_value。 |
| trial.enable_post_motion_guard | bool/dimensionless | False | [False, True] | 固定记录；关闭机制或未选滤波器仍保留参数 | 启用动态运动后adaptive/reset-FFT来源保护与交接，默认关闭。 |
| trial.enable_post_motion_protection | bool/dimensionless | False | [False, True] | 固定记录；关闭机制或未选滤波器仍保留参数 | trial：enable_post_motion_protection；实际采用值见adopted_value。 |
| trial.enable_post_motion_recovery | bool/dimensionless | False | [False, True] | 固定记录；关闭机制或未选滤波器仍保留参数 | 进入recovery时清空adaptive追踪历史；baseline仍延续，默认关闭。 |
| trial.enable_rise_lineage | bool/dimensionless | False | [False, True] | 固定记录；关闭机制或未选滤波器仍保留参数 | 启用有期限的上升候选谱峰lineage确认及重锚，默认关闭。 |
| trial.evaluation_step_s | float/s | 1.0 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | 评价诊断步长1秒，冻结。 |
| trial.evaluation_window_s | float/s | 8.0 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | 评价诊断窗8秒，冻结；不控制预测网格。 |
| trial.final_motion_down_limit_bpm | float/bpm | 2.0 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | 对应阶段的HR候选搜索范围或每窗变化阈值/步长；超过limit按step修正。 |
| trial.final_motion_down_step_bpm | float/bpm | 1.5 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | 对应阶段的HR候选搜索范围或每窗变化阈值/步长；超过limit按step修正。 |
| trial.final_motion_up_limit_bpm | float/bpm | 5.5 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | 对应阶段的HR候选搜索范围或每窗变化阈值/步长；超过limit按step修正。 |
| trial.final_motion_up_step_bpm | float/bpm | 3.5 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | 对应阶段的HR候选搜索范围或每窗变化阈值/步长；超过limit按step修正。 |
| trial.final_recovery_down_limit_bpm | float/bpm | 3.5 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | 对应阶段的HR候选搜索范围或每窗变化阈值/步长；超过limit按step修正。 |
| trial.final_recovery_down_step_bpm | float/bpm | 3.0 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | 对应阶段的HR候选搜索范围或每窗变化阈值/步长；超过limit按step修正。 |
| trial.final_recovery_up_limit_bpm | float/bpm | 1.5 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | 对应阶段的HR候选搜索范围或每窗变化阈值/步长；超过limit按step修正。 |
| trial.final_recovery_up_step_bpm | float/bpm | 1.5 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | 对应阶段的HR候选搜索范围或每窗变化阈值/步长；超过limit按step修正。 |
| trial.final_rest_down_limit_bpm | float/bpm | 3.0 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | 对应阶段的HR候选搜索范围或每窗变化阈值/步长；超过limit按step修正。 |
| trial.final_rest_down_step_bpm | float/bpm | 1.5 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | 对应阶段的HR候选搜索范围或每窗变化阈值/步长；超过limit按step修正。 |
| trial.final_rest_up_limit_bpm | float/bpm | 1.5 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | 对应阶段的HR候选搜索范围或每窗变化阈值/步长；超过limit按step修正。 |
| trial.final_rest_up_step_bpm | float/bpm | 1.5 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | 对应阶段的HR候选搜索范围或每窗变化阈值/步长；超过limit按step修正。 |
| trial.full_candidate_peak_threshold_ratio | float/dimensionless | 0.15 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | trial：full_candidate_peak_threshold_ratio；实际采用值见adopted_value。 |
| trial.global_objective_strategy | str/dimensionless | current_global_adaptive | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | trial：global_objective_strategy；实际采用值见adopted_value。 |
| trial.handoff_confirm_windows | int/windows | 2 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | trial：handoff_confirm_windows；实际采用值见adopted_value。 |
| trial.handoff_deadline_s | float/s | 20.0 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | trial：handoff_deadline_s；实际采用值见adopted_value。 |
| trial.handoff_difference_bpm | float/bpm | 6.0 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | trial：handoff_difference_bpm；实际采用值见adopted_value。 |
| trial.handoff_wait_s | float/s | 5.0 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | trial：handoff_wait_s；实际采用值见adopted_value。 |
| trial.high_lock_candidate_min_bpm | float/bpm | 85.0 | 按类型显式配置 | enable_high_lock_recovery | trial：high_lock_candidate_min_bpm；实际采用值见adopted_value。 |
| trial.high_lock_candidate_stable_bpm | float/bpm | 10.0 | 按类型显式配置 | enable_high_lock_recovery | trial：high_lock_candidate_stable_bpm；实际采用值见adopted_value。 |
| trial.high_lock_confirm_windows | int/windows | 3 | 按类型显式配置 | enable_high_lock_recovery | trial：high_lock_confirm_windows；实际采用值见adopted_value。 |
| trial.high_lock_cooldown_windows | int/windows | 4 | 按类型显式配置 | enable_high_lock_recovery | trial：high_lock_cooldown_windows；实际采用值见adopted_value。 |
| trial.high_lock_down_step_bpm | float/bpm | 20.0 | 按类型显式配置 | enable_high_lock_recovery | trial：high_lock_down_step_bpm；实际采用值见adopted_value。 |
| trial.high_lock_min_amp_ratio | float/dimensionless | 0.45 | 按类型显式配置 | enable_high_lock_recovery | trial：high_lock_min_amp_ratio；实际采用值见adopted_value。 |
| trial.high_lock_min_gap_bpm | float/bpm | 20.0 | 按类型显式配置 | enable_high_lock_recovery | trial：high_lock_min_gap_bpm；实际采用值见adopted_value。 |
| trial.high_lock_penalty_exclusion_bpm | float/bpm | 10.0 | 按类型显式配置 | enable_high_lock_recovery | trial：high_lock_penalty_exclusion_bpm；实际采用值见adopted_value。 |
| trial.high_lock_up_step_bpm | float/bpm | 3.0 | 按类型显式配置 | enable_high_lock_recovery | trial：high_lock_up_step_bpm；实际采用值见adopted_value。 |
| trial.hr_range_hz | float/Hz | 0.4166666666666667 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | trial：hr_range_hz；实际采用值见adopted_value。 |
| trial.hr_step_s | float/s | 1.0 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | HR窗口hop秒数；固定1秒。 |
| trial.hr_window_s | float/s | 8.0 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | HR观察窗秒数；固定8秒，不进入Physical4D。 |
| trial.klms_center_prune_policy | str/dimensionless | freeze_new_centers | ['freeze_new_centers', 'prune_oldest', 'prune_smallest_weight'] | adaptive_filter == klms | trial：klms_center_prune_policy；实际采用值见adopted_value。 |
| trial.klms_distance_mode | str/dimensionless | normalized | ['normalized', 'absolute_squared'] | adaptive_filter == klms | trial：klms_distance_mode；实际采用值见adopted_value。 |
| trial.klms_epsilon | float/dimensionless | 0.1 | 按类型显式配置 | adaptive_filter == klms | trial：klms_epsilon；实际采用值见adopted_value。；数值相同；保留当前归一化距离语义。 |
| trial.klms_max_dictionary_size | int/dimensionless | 300 | 按类型显式配置 | adaptive_filter == klms | trial：klms_max_dictionary_size；实际采用值见adopted_value。 |
| trial.klms_nlms_eps | float/dimensionless | 1e-06 | 按类型显式配置 | adaptive_filter == klms | trial：klms_nlms_eps；实际采用值见adopted_value。 |
| trial.klms_normalized_update | bool/dimensionless | True | [False, True] | adaptive_filter == klms | trial：klms_normalized_update；实际采用值见adopted_value。 |
| trial.klms_sigma | float/dimensionless | 1.0 | 按类型显式配置 | adaptive_filter == klms | trial：klms_sigma；实际采用值见adopted_value。 |
| trial.klms_step_size | float/dimensionless | 0.05 | 按类型显式配置 | adaptive_filter == klms | trial：klms_step_size；实际采用值见adopted_value。 |
| trial.log_absorbance_baseline_mode | str/dimensionless | rolling_median | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | log-absorbance 基线估计方法；rolling_median 抑制慢漂移。 |
| trial.log_absorbance_baseline_window_s | float/s | 5.0 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | 滚动基线窗口长度(秒)；越大越保留低频趋势。 |
| trial.log_absorbance_eps | float/dimensionless | 1e-06 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | log 比值的数值稳定下限，防止零值导致无穷大。 |
| trial.log_absorbance_ratio_clip | list/dimensionless | [0.001, 1000.0] | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | 进入 log 前的比值裁剪范围，防止极端异常值。 |
| trial.low_lock_candidate_stable_bpm | float/bpm | 10.0 | 按类型显式配置 | enable_low_lock_recovery | trial：low_lock_candidate_stable_bpm；实际采用值见adopted_value。 |
| trial.low_lock_confirm_windows | int/windows | 3 | 按类型显式配置 | enable_low_lock_recovery | trial：low_lock_confirm_windows；实际采用值见adopted_value。 |
| trial.low_lock_max_bpm | float/bpm | 80.0 | 按类型显式配置 | enable_low_lock_recovery | trial：low_lock_max_bpm；实际采用值见adopted_value。 |
| trial.low_lock_min_amp_ratio | float/dimensionless | 0.45 | 按类型显式配置 | enable_low_lock_recovery | trial：low_lock_min_amp_ratio；实际采用值见adopted_value。 |
| trial.low_lock_min_bpm | float/bpm | 50.0 | 按类型显式配置 | enable_low_lock_recovery | trial：low_lock_min_bpm；实际采用值见adopted_value。 |
| trial.low_lock_min_jump_bpm | float/bpm | 20.0 | 按类型显式配置 | enable_low_lock_recovery | trial：low_lock_min_jump_bpm；实际采用值见adopted_value。 |
| trial.low_lock_min_windows | int/windows | 4 | 按类型显式配置 | enable_low_lock_recovery | trial：low_lock_min_windows；实际采用值见adopted_value。 |
| trial.low_lock_step_bpm | float/bpm | 30.0 | 按类型显式配置 | enable_low_lock_recovery | trial：low_lock_step_bpm；实际采用值见adopted_value。 |
| trial.low_lock_target_min_bpm | float/bpm | 90.0 | 按类型显式配置 | enable_low_lock_recovery | trial：low_lock_target_min_bpm；实际采用值见adopted_value。 |
| trial.motion_policy | str/dimensionless | single_bout | ['single_bout', 'multi_bout'] | 固定记录；关闭机制或未选滤波器仍保留参数 | trial：motion_policy；实际采用值见adopted_value。 |
| trial.motion_tracking_range_down_bpm | float/bpm | 15.0 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | 对应阶段的HR候选搜索范围或每窗变化阈值/步长；超过limit按step修正。 |
| trial.motion_tracking_range_up_bpm | float/bpm | 35.0 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | 对应阶段的HR候选搜索范围或每窗变化阈值/步长；超过limit按step修正。 |
| trial.motion_tracking_slew_limit_down_bpm | float/bpm | 2.0 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | 对应阶段的HR候选搜索范围或每窗变化阈值/步长；超过limit按step修正。 |
| trial.motion_tracking_slew_limit_up_bpm | float/bpm | 5.5 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | 对应阶段的HR候选搜索范围或每窗变化阈值/步长；超过limit按step修正。 |
| trial.motion_tracking_slew_step_down_bpm | float/bpm | 1.5 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | 对应阶段的HR候选搜索范围或每窗变化阈值/步长；超过limit按step修正。 |
| trial.motion_tracking_slew_step_up_bpm | float/bpm | 3.5 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | 对应阶段的HR候选搜索范围或每窗变化阈值/步长；超过limit按step修正。 |
| trial.normalization_mode | str/dimensionless | minmax | ['minmax', 'zscore', 'none'] | 固定记录；关闭机制或未选滤波器仍保留参数 | 窗口内reference归一化：minmax、zscore或none。 |
| trial.objective_mode | str/dimensionless | posthoc_aae | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | trial：objective_mode；实际采用值见adopted_value。 |
| trial.post_motion_guard_crossover_gap_bpm | float/bpm | 2.0 | 按类型显式配置 | enable_post_motion_guard | trial：post_motion_guard_crossover_gap_bpm；实际采用值见adopted_value。 |
| trial.post_motion_guard_fft_floor_bpm | float/bpm | 55.0 | 按类型显式配置 | enable_post_motion_guard | trial：post_motion_guard_fft_floor_bpm；实际采用值见adopted_value。 |
| trial.post_motion_guard_fft_stable_bpm | float/bpm | 6.0 | 按类型显式配置 | enable_post_motion_guard | trial：post_motion_guard_fft_stable_bpm；实际采用值见adopted_value。 |
| trial.post_motion_guard_fft_stable_windows | int/windows | 3 | 按类型显式配置 | enable_post_motion_guard | trial：post_motion_guard_fft_stable_windows；实际采用值见adopted_value。 |
| trial.post_motion_guard_gap_rescue_enable | bool/dimensionless | False | [False, True] | enable_post_motion_guard | trial：post_motion_guard_gap_rescue_enable；实际采用值见adopted_value。 |
| trial.post_motion_guard_gap_rescue_min_hits | int/windows | 3 | 按类型显式配置 | enable_post_motion_guard | trial：post_motion_guard_gap_rescue_min_hits；实际采用值见adopted_value。 |
| trial.post_motion_guard_gap_rescue_windows | int/windows | 4 | 按类型显式配置 | enable_post_motion_guard | trial：post_motion_guard_gap_rescue_windows；实际采用值见adopted_value。 |
| trial.post_motion_guard_min_elapsed_s | float/s | 5.0 | 按类型显式配置 | enable_post_motion_guard | trial：post_motion_guard_min_elapsed_s；实际采用值见adopted_value。 |
| trial.post_motion_guard_recovery_step_down_bpm | float/bpm | 3.0 | 按类型显式配置 | enable_post_motion_guard | trial：post_motion_guard_recovery_step_down_bpm；实际采用值见adopted_value。 |
| trial.post_motion_guard_recovery_step_up_bpm | float/bpm | 1.5 | 按类型显式配置 | enable_post_motion_guard | trial：post_motion_guard_recovery_step_up_bpm；实际采用值见adopted_value。 |
| trial.post_motion_guard_rescue_gap_bpm | float/bpm | 20.0 | 按类型显式配置 | enable_post_motion_guard | trial：post_motion_guard_rescue_gap_bpm；实际采用值见adopted_value。 |
| trial.post_motion_guard_rising_slope_bpm_per_window | float/bpm | 1.5 | 按类型显式配置 | enable_post_motion_guard | trial：post_motion_guard_rising_slope_bpm_per_window；实际采用值见adopted_value。 |
| trial.post_motion_guard_rising_windows | int/windows | 3 | 按类型显式配置 | enable_post_motion_guard | trial：post_motion_guard_rising_windows；实际采用值见adopted_value。 |
| trial.post_motion_guard_seconds | NoneType/s | None | 按类型显式配置 | enable_post_motion_guard | trial：post_motion_guard_seconds；实际采用值见adopted_value。 |
| trial.post_motion_guard_stable_windows | int/windows | 3 | 按类型显式配置 | enable_post_motion_guard | trial：post_motion_guard_stable_windows；实际采用值见adopted_value。 |
| trial.post_motion_guard_upward_gap_bpm | float/bpm | 1.5 | 按类型显式配置 | enable_post_motion_guard | trial：post_motion_guard_upward_gap_bpm；实际采用值见adopted_value。 |
| trial.postprocess_method | str/dimensionless | fft | ['fft', 'ssr'] | 固定记录；关闭机制或未选滤波器仍保留参数 | trial：postprocess_method；实际采用值见adopted_value。 |
| trial.ppg_input_transform | str/dimensionless | log_absorbance | ['raw_bandpass', 'log_absorbance'] | 固定记录；关闭机制或未选滤波器仍保留参数 | PPG 输入变换；默认 log_absorbance，也可选择后端支持的既有变换。 |
| trial.prewarm_s | float/s | 0.0 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | 自适应前置上下文秒数；默认0，与8秒观察窗独立。 |
| trial.qc_policy | str/dimensionless | fallback_baseline | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | 坏窗口策略；fallback_baseline 表示保留计算并回退 baseline HR。 |
| trial.recovery_tracking_range_down_bpm | float/bpm | 25.0 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | 对应阶段的HR候选搜索范围或每窗变化阈值/步长；超过limit按step修正。 |
| trial.recovery_tracking_range_up_bpm | float/bpm | 20.0 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | 对应阶段的HR候选搜索范围或每窗变化阈值/步长；超过limit按step修正。 |
| trial.recovery_tracking_slew_limit_down_bpm | float/bpm | 3.5 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | 对应阶段的HR候选搜索范围或每窗变化阈值/步长；超过limit按step修正。 |
| trial.recovery_tracking_slew_limit_up_bpm | float/bpm | 1.5 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | 对应阶段的HR候选搜索范围或每窗变化阈值/步长；超过limit按step修正。 |
| trial.recovery_tracking_slew_step_down_bpm | float/bpm | 3.0 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | 对应阶段的HR候选搜索范围或每窗变化阈值/步长；超过limit按step修正。 |
| trial.recovery_tracking_slew_step_up_bpm | float/bpm | 1.5 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | 对应阶段的HR候选搜索范围或每窗变化阈值/步长；超过limit按step修正。 |
| trial.reference_lag_ms | float/ms | 200.0 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | 每窗reference correlation搜索半宽，毫秒。 |
| trial.rff_D | int/dimensionless | 100 | 按类型显式配置 | adaptive_filter == rff_lms | trial：rff_D；实际采用值见adopted_value。 |
| trial.rff_err_clip | NoneType/dimensionless | None | 按类型显式配置 | adaptive_filter == rff_lms | trial：rff_err_clip；实际采用值见adopted_value。 |
| trial.rff_leakage | float/dimensionless | 0.0 | 按类型显式配置 | adaptive_filter == rff_lms | trial：rff_leakage；实际采用值见adopted_value。 |
| trial.rff_nlms_eps | float/dimensionless | 1e-09 | 按类型显式配置 | adaptive_filter == rff_lms | trial：rff_nlms_eps；实际采用值见adopted_value。 |
| trial.rff_seed | int/dimensionless | 42 | 按类型显式配置 | adaptive_filter == rff_lms | 随机特征固定种子42；不同坐标不派生新seed。 |
| trial.rff_sigma_scale | float/dimensionless | 1.0 | 按类型显式配置 | adaptive_filter == rff_lms | trial：rff_sigma_scale；实际采用值见adopted_value。；保留robust tap距离自适应带宽；参考采用固定sigma=1。 |
| trial.rff_theta_norm_guard | NoneType/dimensionless | None | 按类型显式配置 | adaptive_filter == rff_lms | trial：rff_theta_norm_guard；实际采用值见adopted_value。 |
| trial.rff_update_mode | str/dimensionless | nlms | ['nlms', 'lms'] | adaptive_filter == rff_lms | trial：rff_update_mode；实际采用值见adopted_value。 |
| trial.rise_lineage_confirm_windows | int/windows | 3 | 按类型显式配置 | enable_rise_lineage | 连续跟踪候选达到该窗口数后才允许lineage reanchor。 |
| trial.rise_lineage_harmonic_multiple | float/dimensionless | 3.0 | 按类型显式配置 | enable_rise_lineage | 接近motion峰此谐波的lineage被阻止reanchor。 |
| trial.rise_lineage_max_age_windows | int/windows | 6 | 按类型显式配置 | enable_rise_lineage | lineage的最大追踪窗口寿命。 |
| trial.rise_lineage_max_downward_drift_bpm | float/bpm | 1.0 | 按类型显式配置 | enable_rise_lineage | lineage相邻窗允许的最大向下漂移bpm。 |
| trial.slew_limit_bpm | int/bpm | 10 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | trial：slew_limit_bpm；实际采用值见adopted_value。 |
| trial.slew_step_bpm | int/bpm | 7 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | trial：slew_step_bpm；实际采用值见adopted_value。 |
| trial.smooth_win_len | int/windows | 5 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | baseline/adaptive HR序列的移动中位数窗口数；主路径固定5。 |
| trial.spectral_penalty_policy | str/dimensionless | soft_visible | ['soft_visible', 'suppressed_protected_continuous_visibility'] | 固定记录；关闭机制或未选滤波器仍保留参数 | soft_visible默认保持候选可见并保护连续性；完整policy增加有期限的owner可见性。 |
| trial.spectral_penalty_policy_version | int/dimensionless | 1 | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | trial：spectral_penalty_policy_version；实际采用值见adopted_value。 |
| trial.tracker_mode | str/dimensionless | enhanced | 按类型显式配置 | 固定记录；关闭机制或未选滤波器仍保留参数 | trial：tracker_mode；实际采用值见adopted_value。 |
| trial.visibility_edge_guard_bpm | float/bpm | 1.0 | 按类型显式配置 | 完整spectral_penalty_policy | 候选排除频带的边缘保护宽度，默认1bpm。 |
| trial.visibility_match_half_width_bpm | float/bpm | 3.5 | 按类型显式配置 | 完整spectral_penalty_policy | owner与当前候选的匹配半宽，参考固定3.5bpm。 |
| trial.visibility_motion_invocations | int/dimensionless | 3 | 按类型显式配置 | 完整spectral_penalty_policy | owner取得后最多3次motion调用可越过排除；不按秒或所有窗口计数。 |

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
