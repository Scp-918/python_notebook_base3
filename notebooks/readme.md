# core_enhanced_v1 Notebook 操作说明

入口为 `run_batch_adaptive_protocol.ipynb`，由 `scripts/update_change6_notebook.py` 生成；修改注释和配置后重新生成，两者一并提交。核心定义与参数详见根README和params.py。

## 1. 环境、输入与配置

使用PPG_sensor_env，在仓库根目录或notebooks目录打开Notebook。设置PPG_SUBJECT_DIR指向一个受试者目录，或使用仓库相对路径；不写本机绝对路径。只支持严格Pydisplay sensor CSV、标准HRdata CSV，文件为subject_motion_index_sensor/HRdata.csv，motion为write/gripper/run/rope；父目录必须有ck.mat。缺少配对只报告原因，不进入训练；序号修复、missing/spike repair、UD分母QC和坏窗回退仍保留。

第0节分四个配置单元：路径/实验轴、预处理/评价、冻结preset/固定验收、输出/执行开关。参数集中于params.py；Notebook不复制可编辑的rescue或tracker参数。未知字段、旧字段及冻结机制覆盖明确报错。

- 固定验收默认Fs_Target=100Hz、memory_ms=160ms、LMS_Mu_Base=.01；HR observation固定8秒/1秒。
- memory_ms派生history_tap_limit=max(1,round(memory_ms*Fs_Target/1000))，只控制正时延历史上限；实际M/K按延迟估计，M_base=2、C_scale=1.2、K_max=12冻结。日志中的实际history/forward跨度用于核查，不宣称K的物理时长统一。
- prewarm_s=0是adaptive前置上下文，独立于observation与评价对齐8/1。
- 实验轴为filter、FFT/SSR、reference groups、single_bout/multi_bout、scope和raw_bandpass/log_absorbance。固定配置及实验轴由resolve_core_config解析。
- BO仅Fs_Target[25,50]、memory_ms[40,80,120,160,200]、滤波器步长、Spec_Penalty_Width[.1,.2,.3]Hz；RFF解码为LMS_Mu_Base，KLMS为klms_step_size。Weight=.2为保留比例，冻结，越小抑制越强。

## 2. 预览与固定验收

先执行第1节严格配对与标定。按需打开RUN_STATIC_QC，只做只读筛查。打开RUN_PREPROCESS_PREVIEW，在第2节指定PREVIEW_MOTION_TYPE/INDEX，检查source_metadata、QC和分段。

source IMU固定原始约100Hz，独立于adaptive和Fs_Target：六轴四阶零相位带通ACC .5–5Hz、Gyro .5–10Hz，分别magnitude，8/1秒STD(ddof=1)。前30秒baseline，阈值max(2.5*baseline_std,.05*max_window_score)，任一路严格超过即motion；先桥≤3窗false gap，再删<5窗true run。detector返回全部runs，独立policy默认选最长（等长最早）；multi_bout保留全部。metadata分别记录detected和selected结果。

打开RUN_FIXED_CORE执行第2节后的固定参数单元，不启动Optuna；输出fixed_core/windows.csv和preset.json。核查时间轴、phase、episode、恢复经过时间、候选来源和handoff。窗口统一start/center/end，状态以center判断，motion为首true center到下一false center半开区间。reference覆盖不足只使评价NaN，不裁剪预测。

四层pipeline：FFT/SSR spectrum → shared soft penalty与candidate tracker → source selection/handoff → phase-aware Final dynamics。tracker第一层slew保护候选连续性；Final第二层slew保护phase/source拼接，跨phase延续状态。rest up1.5/1.5、down3/1.5；motion up5.5/3.5、down2/1.5；recovery up1.5/1.5、down3.5/3（limit/step bpm）。low/high-lock、rise-lineage、gap/rising rescue、complex shadow及interface-quality在线介入全部显式关闭。

motion_only恢复立即reset-FFT；motion_post10控制真实adaptive范围，最迟10秒范围结束切换。一般handoff等待5秒，差值≤6bpm连续2有效窗切换，最迟20秒切到有效reset-FFT；每次motion结束重置tracker/handoff，episode内锁存。无效候选按QC有效源回退，不调用rescue，也不将范围外baseline冒充adaptive。

先看整体/rest/motion/recovery AAE、P95误差、最大单窗jump和起止/recovery曲线。性能异常先检查时间、reference顺序、M/K、state和penalty，勿通过重开rescue补指标。

## 3. Batch 与 resume

第7节只使用第0节的配置，不再隐式重置filter或重复跑Volterra。RUN_ALL_TRAIN=True启动按运动、scope、reference、filter划分的模式级搜索；各运动类型全部index共同形成objective。RUN_CONNECTIVITY_CHECK在第8节运行少量trial，仍会处理选中所有记录。先完成固定验收，再决定是否搜索。

输出为SUBJECT_DIR.parent/outputs/{subject}__{run_signature}；下含motion_types、模式checkpoint、history及Stage-6紧凑CSV/JSON。再次执行同一batch配置时自动读取模式断点；保持相同输入与完整有效配置继续断点；fingerprint包含schema、preset、所有配置与输入签名。改变配置不得误用旧checkpoint。CLEAN_OUTPUTS保持False，历史输出不自动迁移或修改。

reference唯一真源为ordered tuples：单组("ACC",)或("UD",)，双组("ACC","HF")展开accx/accy/accz/hf1/hf2，反向("HF","ACC")展开hf1/hf2/accx/accy/accz。通道不按延迟相关性再重排。canonical key ACC+HF/HF+ACC关联run/mode/path、CSV/JSON、cache、legend、summary和回放。物理采集字段UD2不是reference模式名。

## 4. 输出、回放、汇总与绘图

第9节RUN_OUTPUT_CHECK核查目录和记录，不启动训练。第10节设置RESULTS_ROOT、REPLAY_MOTION_TYPE/INDEX、filter、reference canonical key、scope及prewarm，打开RUN_STAGE7_REPLAY；恢复已保存配置，只返回当前core的plot和csv。第11节RUN_WINDOW_DIAGNOSTICS指定传感器时间域window_start_s，查看前置上下文、实际algorithm input、频谱/penalty权重、tracker和来源状态；参考HR只在完成预测后用于对照。

第12节RUN_CROSS_MOTION_SUMMARY只读已有记录，输出write/gripper/run/rope汇总。第13–15节可批量回放、诊断和reference比较；第16节可重画已有目标段。第3–6节独立开关生成静息评价对齐、完整曲线、原始信号图。写图均显式开启，结果进入outputs；检查实际estimator而非只看请求值（SSR无有效候选时记录FFT fallback）。

新结果含result_schema_version、preset_name/version、feature_flags、完整核心参数、参考实现SHA及config fingerprint。检查Stage-6 best_params.json/CSV、manifest和固定preset.json中的一致性；不要只看文件夹名字。旧schema、旧枚举、缺字段旧记录明确拒绝，没有兼容fallback。保留历史目录，使用新输出身份重新运行再回放。

## 5. PPG / UD 血缘及提交前检查

ppg_acquired_raw保留source采集网格，序号缺口NaN、毛刺原值；ppg_clean承担missing/spike repair；ppg_algorithm_input从clean做bandpass/log_absorbance及目标域resampling。改变Fs不改变原始source。

raw Uh/Uc加标定产生UD calculation，adaptive reference使用独立预处理副本；InterfaceQualityInput只含raw PPG、Uh/Uc、UD calculation、source时间和QC，不接受adaptive output或reference HR。quality暂未实现，无循环依赖或反向修改reference。参考HR仅评价，不参与在线求解；算法含零相位、全记录阈值和前视，不声称严格因果。

运行 `PYTHONPATH=src` 下的全量pytest与可用Ruff。提交前清空Notebook执行输出和计数，移除绝对路径，RUN_*全部恢复False；注释应描述实际冻结实现，不保留旧兼容/rescue说明。新旧指标比较必须注明检测边界、评价对齐和预测覆盖改变。
