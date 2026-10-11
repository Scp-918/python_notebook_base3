# Physical4D Notebook 操作与复现

入口 `run_batch_adaptive_protocol.ipynb` 由 `scripts/update_change6_notebook.py` 生成。使用PPG_sensor_env，在仓库根或notebooks目录打开。参数定义集中于params.py；修改Notebook说明时同步生成脚本。全部RUN开关默认False，关闭时可顺序执行，无需存在输入、训练结果或输出目录。

## 配置与训练

第0节设置SUBJECT_DIRS（一/多个目录）、MOTION_TYPES、REFERENCE_COMBINATIONS（有序组tuple，一/多个组合）、ADAPTIVE_FILTERS（lms/volterra/rff_lms/klms）、SCOPES（motion_only/motion_recovery/motion_post10/global）。多组合各有独立root，每样本每组合300点。OUTPUT_PARENT默认共同数据父目录/outputs；父目录不同必须明确指定。

叶目录只含五项，例如S01-S02__wr__AH__L__mp10，其他配置写入manifest/config。同名root固定配置不一致会拒绝续写。不同policy可放不同OUTPUT_PARENT。配对按subject_motion_index_sensor/HRdata.csv，ck.mat位于受试者父目录；未配对记原因，QC异常按保存策略回退。

TRAIN_PARAMS显式列出主路径固定值、policy和默认关闭的所有复杂机制；TRAIN_PREPROCESSING、TRAIN_DETECTOR、TRAIN_ALIGNMENT、TRAIN_NUMERICAL包含全部其他参数。用dataclasses.replace显式修改，实际调用单元再次赋值对象并展示生效配置。第5节完整参数表及parameter_catalog.json说明单位、可选项、启用条件、参考/采用值及差异。不要修改固定Physical4D网格候选轴。

第1节配对，第2节预处理，第3节静息对齐，第4节原始信号/FFT，第5节配置，第6节300点查看，第7节训练，第8节只读SQLite状态，第9节补导出。预览单独开关，仅预览不进入训练账本。训练首次预处理前冻结config/fixed_hyperparameters.json及hash，展开未选滤波器/关闭机制参数；不允许用当前默认值填已有快照缺字段。

sample-level source ACC/GYRO分段只算一次；首次检测运动前截静息再生成完整8秒窗，Tdelay默认[-5,5]/.1s；不足2窗回退0并保存原因。sensor/运动边界不移动。Tdelay后读取HR_ref(t+Tdelay_s)，posthoc后读取HR_ref(t+Tdelay_s+posthoc_bias_s)。posthoc独立默认[-5,5]/1s。跨Fs共享样本状态，每cell重新初始化在线滤波/追踪，旧reference HR仅参与对齐/评价。

默认soft_visible保持候选可见+continuity protection；完整suppressed_protected_continuous_visibility实现owner生命周期，配置改变进入hash/identity，不进入网格。rise、low/high-lock、post-motion recovery/guard、dual-reset全部可用但默认关闭。Final优先dual-reset、guard、简化融合，再一次dynamics；恢复期简化handoff使用reset-FFT。RFF自适应robust sigma、KLMS归一化距离等与参考差异见目录。

SQLite cell逐次即时事务；技术失败仅attempt/error，重启补缺；unscored保存NULL，300点全部执行完成才sample_complete。完成样本优先跳过，当前Fs/样本结束清缓存及GC。每样本保存align图、signal/FFT图、固定300行P4D；每个体共享所有运动的sample_best与motion_summary，根保存overall_summary。第9节缺失图/Excel可单独补，不跑300点。

## 单样本与窗口诊断（10–11节）

RESULTS_ROOT指定已有run_root。SAMPLE_ID使用SQLite样本名，无_sensor后缀；COORDINATE_ID默认best_posthoc，或p4d_000…299。best仅完整可评分样本；显式坐标可选部分样本中已完成cell。

开启RUN_SINGLE_REPLAY：先显示状态、坐标、policy/rescue、完整保存参数、hash、Tdelay、bias及保存指标，再从起点恢复算法。REPLAY_EFFECTIVE_PARAMS来自ReplaySpec，TRAIN_PARAMS不能覆盖它。输入/标定hash必须一致，迁移可设置DIRECTORY_MAPPING。复现不做detector、对齐、oracle或网格搜索；无效保存bias保持unscored。指标绝对容差1e-6，有效窗数相等，输出双面板HR、windows.csv、metrics.json、receipt.json。图标题按原scope评价，PLOT_RANGE_S只裁显示。

RUN_WINDOW_DIAGNOSTICS复用同一ReplaySpec；DIAGNOSTIC_START_S匹配最近窗口，偏差最多半hop，越界报错。保留完整先前历史，只收集该窗完整stage：ranking/shared delay/M、group K、实际步长与权重、原始/惩罚频谱、policy/owner、tracker/fusion/rescue。诊断保存到样本replay目录下独立diagnostics子目录。

## 汇总与独立reference实验（12–15节）

第12节只读SQLite重建汇总，不重算。第13–14节显式跨reference实验及窗口诊断，固定源样本坐标、分段、Tdelay和保存bias。SEARCH_EXPERIMENT_ORACLE默认False；True另记实验bias。实验独立identity、config和experiments输出，不覆盖精确复现或训练。

第15节多样本reference比较：各自恢复自身best坐标和两个偏移，不能复用首样本参数。BATCH_SUBJECTS/MOTIONS/SAMPLE_IDS均支持字符串或列表，None表示全部。

## 批量严格复现（16节）

RUN_BATCH_REPLAY首先固定并展示每样本选择清单，再串行执行。每样本保存receipt，立即释放临时数据；匹配identity且verified、输出文件hash完整时cached_verified，中断只补尚未验证样本。失败/不一致继续后续样本；状态为verified/cached_verified/metric_mismatch/failed/skipped。默认best跳未完成或无有限best，汇总仅验证通过样本并保留失败/跳过计数。只读训练数据库，不改cell或sample_complete。

批量输出位于replay/batches/<selection_hash>，包含selection.json、逐样本progress.csv和batch_summary.xlsx。说明与数值配置分离；篡改/缺字段/不支持schema/实现身份变化明确报错，不静默降级。旧BO目录保留历史兼容，不自动迁移到新账本。

提交时重新生成Notebook，清空输出及执行计数，不保留本机绝对路径，所有RUN开关False。只做轻量300点账本验证、代表性真实坐标复算及必要机制行为测试，不设置完整批量性能或AAE提升门槛。
