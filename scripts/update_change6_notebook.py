"""Generate the Physical4D training and strict SQLite replay notebook."""
from __future__ import annotations

from pathlib import Path
import sys
import textwrap
import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'notebooks/run_batch_adaptive_protocol.ipynb'
sys.path.insert(0, str(ROOT / 'src'))


def build_notebook():
    """Return deterministic, cleared cells; disabled execution requires no input files."""
    cells = []
    def section(number, title, explanation, code):
        cells.append(nbformat.v4.new_markdown_cell(f'## {number}. {title}\n\n{explanation}', id=f'p4d-{number}-md'))
        cells.append(nbformat.v4.new_code_cell(textwrap.dedent(code).strip(), id=f'p4d-{number}-code'))
    section(0, '完整配置与执行开关', '''正式基线是 ACC + LMS。每样本冻结 source-domain motion segmentation 和 Tdelay_s；300 点只搜索 Fs、memory、步长及 penalty 半宽。所有 RUN 开关默认 False，无数据、无训练结果也可顺序执行。

固定配置真源为 config/fixed_hyperparameters.json；坐标、样本状态及 posthoc_bias_s 以 SQLite 为准。复现不使用当前训练配置补字段。完整固定超参数目录（说明、单位、可选项、启用条件与参考差异）见项目根目录 README.md；每次运行另存 parameter_catalog.json。soft_visible 默认开启 continuity protection；suppressed_protected_continuous_visibility 是可运行的完整可选策略，policy 不进入网格，但进入配置 hash 和结果 identity。复杂 rescue 默认关闭，选择完整 policy 仅启用其 advanced visibility。

训练输入支持多 subjects、多 motions，以及多个有序 reference 组合、filter、scope。每个外层组合独立根目录，每样本完整300点。不同数据父目录必须明确指定 OUTPUT_PARENT。同名目录配置不同会拒绝续写。''', '''
        import os
        import sys
        from pathlib import Path
        from dataclasses import replace
        import pandas as pd
        from IPython.display import display, Image as NotebookImage

        REPO_ROOT = Path.cwd() if (Path.cwd() / "src").is_dir() else Path.cwd().parent
        sys.path.insert(0, str(REPO_ROOT / "src"))
        from ppg_hr.params import (reference_parity_params, PreprocessingParams,
            MotionDetectorParams, Physical4DSettings, NumericalParams)
        from ppg_hr.experimental.physical4d import build_fixed_config, parameter_catalog, coordinates, run_name
        from ppg_hr.experimental.physical4d_run import run_physical4d, RunLedger, repair_run_outputs
        from ppg_hr.experimental.physical4d_export import rebuild_summaries
        from ppg_hr.experimental.physical4d_replay import (resolve_replay_spec, replay_sample,
            replay_batch, replay_reference_experiment, replay_reference_batch)

        SUBJECT_DIR = Path(os.environ.get("PPG_SUBJECT_DIR", REPO_ROOT / "testdata/data/ts"))
        SUBJECT_DIRS = [SUBJECT_DIR]  # 多个受试者目录；按名称稳定排序
        MOTION_TYPES = ["write", "gripper", "run", "rope"]  # 单选或多选
        REFERENCE_COMBINATIONS = (("ACC",),)  # 可增加 ("ACC", "HF")，组顺序影响同分排序
        ADAPTIVE_FILTERS = ("lms",)  # lms / volterra / rff_lms / klms
        SCOPES = ("motion_post10",)  # motion_only / motion_recovery / motion_post10 / global
        OUTPUT_PARENT = None  # 默认共同数据父目录 / outputs；不同父目录必须显式指定
        RESULTS_ROOT = (Path(OUTPUT_PARENT) if OUTPUT_PARENT is not None else SUBJECT_DIR.parent / "outputs") / run_name(
            [p.name for p in SUBJECT_DIRS], MOTION_TYPES, REFERENCE_COMBINATIONS[0], ADAPTIVE_FILTERS[0], SCOPES[0])

        TRAIN_PARAMS = reference_parity_params(
            adaptive_filter=ADAPTIVE_FILTERS[0], delay_estimation_mode="direct", reference_lag_ms=200,
            smooth_win_len=5, Spec_Penalty_Weight=0.4, M_base=1, C_scale=1, K_max=16,
            hr_window_s=8, hr_step_s=1, prewarm_s=0,
            spectral_penalty_policy="soft_visible", enable_continuity_protection=True,
            enable_rise_lineage=False, enable_low_lock_recovery=False, enable_high_lock_recovery=False,
            enable_post_motion_recovery=False, enable_post_motion_guard=False,
            enable_post_motion_protection=False, enable_dual_reset=False)
        TRAIN_PREPROCESSING = PreprocessingParams()  # 包括源采样率、修复、带通、UD与数值控制
        TRAIN_DETECTOR = MotionDetectorParams()  # source-domain ACC/GYRO；不随 Fs 重新检测
        TRAIN_ALIGNMENT = Physical4DSettings()  # Tdelay [-5,5]/0.1s，posthoc 由 TRAIN_PARAMS 控制
        TRAIN_NUMERICAL = NumericalParams()  # FFT、归一化、内核稳定阈值也进入保存快照
        TRAIN_FIXED_CONFIG = build_fixed_config(TRAIN_PARAMS, preprocessing=TRAIN_PREPROCESSING,
            motion_detector=TRAIN_DETECTOR, settings=TRAIN_ALIGNMENT, numerical=TRAIN_NUMERICAL,
            references=REFERENCE_COMBINATIONS[0], target_scope=SCOPES[0])
        PHYSICAL4D_GRID = coordinates(ADAPTIVE_FILTERS[0])  # 300点固定空间，不修改候选轴

        SAMPLE_ID = "ts_write_1"  # SQLite sample_id，无 _sensor 后缀
        COORDINATE_ID = "best_posthoc"  # 或 p4d_000…p4d_299；指定坐标允许已完成cell的部分样本
        DIRECTORY_MAPPING = {}  # 数据迁移：{旧目录: 新目录}；文件内容/标定身份仍必须一致
        DIAGNOSTIC_START_S = 30.0  # 最近窗口，允许半hop偏差；从样本起点保持全部历史
        PLOT_RANGE_S = None  # 如 (20,80)；仅裁图，不改变 scope 评价
        BATCH_SUBJECTS = None  # 如 ["S01","S02"]；None表示全选
        BATCH_MOTIONS = None  # 如 ["write","run"]
        BATCH_SAMPLE_IDS = None  # 如 ["S01_write_1","S02_run_2"]
        EXPERIMENT_REFERENCES = (("ACC",), ("ACC", "HF"))
        SEARCH_EXPERIMENT_ORACLE = False  # True时另记实验bias，不覆盖训练bias
        REPLAY_SPEC = None
        RUN_DISCOVERY = False
        RUN_PREPROCESS_PREVIEW = False
        RUN_ALIGNMENT_DIAGNOSTICS = False
        RUN_SIGNAL_DIAGNOSTICS = False
        RUN_CONFIG_DISPLAY = False
        RUN_GRID_DISPLAY = False
        RUN_TRAIN = False
        RUN_SQL_STATUS = False
        RUN_REPAIR_OUTPUTS = False
        RUN_SINGLE_REPLAY = False
        RUN_WINDOW_DIAGNOSTICS = False
        RUN_SUMMARY = False
        RUN_CROSS_REFERENCE = False
        RUN_CROSS_DIAGNOSTICS = False
        RUN_BATCH_REFERENCE_COMPARE = False
        RUN_BATCH_REPLAY = False

        def show_config(config):
            display(pd.DataFrame(parameter_catalog(config)).T)

        def show_figure(fig):
            from io import BytesIO
            import matplotlib.pyplot as plt
            buffer = BytesIO()
            fig.savefig(buffer, format="png", bbox_inches="tight")
            display(NotebookImage(data=buffer.getvalue()))
            plt.close(fig)

        def show_window_diagnostic(directory, fs):
            import json
            import numpy as np
            import matplotlib.pyplot as plt
            diagnostic = json.loads((directory / "window_diagnostic.json").read_text(encoding="utf-8"))
            display({key: value for key, value in diagnostic.items() if not key.endswith("_json")})
            stages = json.loads(diagnostic.get("adaptive_stages_json", "[]"))
            if stages:
                display(pd.DataFrame(stages[0].get("reference_channel_ranking", [])))
                display(pd.DataFrame([{key: value for key, value in stage.items()
                    if not isinstance(value, (list, dict))} for stage in stages]))
                fig, axes = plt.subplots(2, 1, figsize=(12, 6))
                for stage in stages:
                    signal = stage.get("output_signal", [])
                    axes[0].plot(np.arange(len(signal)) / fs,
                        signal, label=stage["channel"])
                    weights = stage.get("weight_norm_t", stage.get("theta_norm_t", []))
                    if weights:
                        axes[1].plot(weights, label=stage["channel"])
                axes[0].set_title("Cascade output; full selected window")
                axes[1].set_title("Adaptive weight norm")
                for ax in axes:
                    if ax.lines:
                        ax.legend()
                plt.tight_layout()
                show_figure(fig)
            for name in ("baseline", "adaptive", "reset_fft"):
                trace = json.loads(diagnostic.get(name + "_spectrum_tracking_json", "{}"))
                evidence = trace.pop("spectrum_evidence", {})
                reference_spectrum = trace.pop("penalty_reference_spectrum", {})
                display({"path": name, "tracking_penalty_owner_rescue": trace})
                if evidence:
                    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
                    frequency = evidence["frequency_hz"]
                    axes[0].plot(frequency, evidence["raw_amplitude"], label="raw")
                    axes[0].plot(frequency, evidence["penalized_amplitude"], label="penalized")
                    axes[1].plot(frequency, evidence["penalty_weights"], label="penalty weights")
                    if reference_spectrum:
                        axes[0].plot(reference_spectrum["frequency_hz"], reference_spectrum["amplitude"],
                            label="penalty reference", alpha=.5)
                    axes[0].set_title(name + " spectrum / policy / visibility")
                    axes[1].set_xlabel("Hz")
                    for ax in axes:
                        ax.set_xlim(.2, 5)
                        ax.legend()
                    plt.tight_layout()
                    show_figure(fig)
        ''')
    section(1, '严格配对与输入选择', '命名为 subject_motion_index_sensor.csv / HRdata.csv；标定读取受试者父目录 ck.mat。未配对文件记录原因，不参加训练。', '''
        if RUN_DISCOVERY:
            from ppg_hr.experimental.batch_pairing import discover_sample_pairs_with_unpaired
            for subject in SUBJECT_DIRS:
                discovery = discover_sample_pairs_with_unpaired(subject)
                display(pd.DataFrame([vars(p) for p in discovery.pairs]))
                display(pd.DataFrame([vars(p) for p in discovery.unpaired]))
        ''')
    section(2, '样本预处理预览', '本节只在显式开启时读样本。预览使用当前训练配置；精确复现使用保存配置。QC 异常按保存回退策略处理。', '''
        PREVIEW_DATASET = None
        PREVIEW_CONTEXT = None
        def load_training_preview():
            from ppg_hr.experimental.batch_pairing import discover_sample_pairs
            from ppg_hr.experimental.preprocess_protocol import load_and_preprocess_protocol
            pairs = [p for subject in SUBJECT_DIRS for p in discover_sample_pairs(subject) if p.sample_id == SAMPLE_ID]
            if len(pairs) != 1:
                raise ValueError("预览选择必须唯一匹配样本")
            pair = pairs[0]
            dataset = load_and_preprocess_protocol(pair.sensor_csv, pair.ref_csv,
                fs_origin=TRAIN_PREPROCESSING.fs_origin, preprocessing=TRAIN_PREPROCESSING,
                ud_calculation_mode=TRAIN_PREPROCESSING.ud_calculation_mode,
                ud_smoothing_window_s=TRAIN_PREPROCESSING.ud_smoothing_window_s)
            dataset.sample_stem = pair.sample_id
            return dataset, pair
        if RUN_PREPROCESS_PREVIEW:
            PREVIEW_DATASET, PREVIEW_PAIR = load_training_preview()
            display(PREVIEW_DATASET.source_metadata)
        ''')
    section(3, '静息 Tdelay 与冻结分段诊断', '首次检测运动前先截静息片段，再生成完整8秒窗。少于2个可比较窗口时 Tdelay=0并记录原因。sensor与motion boundary保持不变，评价读取 HR_ref(t + Tdelay_s)。', '''
        if RUN_ALIGNMENT_DIAGNOSTICS:
            from ppg_hr.experimental.sample_context import prepare_sample_context, save_sample_diagnostics
            from ppg_hr.experimental.physical4d_run import file_identity
            if PREVIEW_DATASET is None:
                PREVIEW_DATASET, PREVIEW_PAIR = load_training_preview()
            PREVIEW_INPUT_ID = {key: file_identity(path) for key, path in dict(sensor=PREVIEW_PAIR.sensor_csv,
                reference=PREVIEW_PAIR.ref_csv, calibration=PREVIEW_PAIR.sensor_csv.parent.parent / "ck.mat").items()}
            PREVIEW_CONTEXT = prepare_sample_context(PREVIEW_DATASET, TRAIN_PARAMS,
                PREVIEW_INPUT_ID, fixed_config=TRAIN_FIXED_CONFIG)
            save_sample_diagnostics(RESULTS_ROOT / "preview", PREVIEW_PAIR.subject, PREVIEW_DATASET,
                PREVIEW_CONTEXT, REFERENCE_COMBINATIONS[0])
            display(PREVIEW_CONTEXT.to_dict())
        ''')
    section(4, '原始信号与 FFT 基础诊断', '训练每样本保存一次 signal/<sample>__fft_signal.png，含原始PPG、启用reference、源域IMU和基础频谱，并标明处理状态。与坐标无关；缺图可在第9节补导出。', '''
        if RUN_SIGNAL_DIAGNOSTICS:
            from ppg_hr.experimental.sample_context import prepare_sample_context, save_sample_diagnostics
            from ppg_hr.experimental.physical4d_run import file_identity
            if PREVIEW_DATASET is None:
                PREVIEW_DATASET, PREVIEW_PAIR = load_training_preview()
            if PREVIEW_CONTEXT is None:
                PREVIEW_INPUT_ID = {key: file_identity(path) for key, path in dict(sensor=PREVIEW_PAIR.sensor_csv,
                    reference=PREVIEW_PAIR.ref_csv, calibration=PREVIEW_PAIR.sensor_csv.parent.parent / "ck.mat").items()}
                PREVIEW_CONTEXT = prepare_sample_context(PREVIEW_DATASET, TRAIN_PARAMS,
                    PREVIEW_INPUT_ID, fixed_config=TRAIN_FIXED_CONFIG)
            save_sample_diagnostics(RESULTS_ROOT / "preview", PREVIEW_PAIR.subject, PREVIEW_DATASET,
                PREVIEW_CONTEXT, REFERENCE_COMBINATIONS[0])
        ''')
    section(5, '固定配置查看', '''完整固定超参数目录已移至项目根目录 README.md，包含默认值、单位、可选项、启用条件与参考差异。当前训练的实际固定配置可在开启 RUN_CONFIG_DISPLAY 后查看；每次运行以保存的 config/fixed_hyperparameters.json 为准。''', '''
        if RUN_CONFIG_DISPLAY:
            DISPLAY_FIXED_CONFIG = build_fixed_config(TRAIN_PARAMS, preprocessing=TRAIN_PREPROCESSING,
                motion_detector=TRAIN_DETECTOR, settings=TRAIN_ALIGNMENT, numerical=TRAIN_NUMERICAL,
            references=REFERENCE_COMBINATIONS[0], target_scope=SCOPES[0])
            show_config(DISPLAY_FIXED_CONFIG)
        ''')
    section(6, '确定性 Physical4D 网格', 'Fs → memory_ms → step_value → penalty_half_width_bpm；最后一维最快。max_order=round(Fs×memory/1000)，Hz半宽=bpm/60。每个滤波器独立五点步长；policy、seed、rescue不搜索。', '''
        if RUN_GRID_DISPLAY:
            for filter_name in ADAPTIVE_FILTERS:
                display(pd.DataFrame(coordinates(filter_name)))
        ''')
    section(7, '训练与逐样本续跑', '开启后逐cell事务写入SQLite，技术失败仅记attempt/error，下次只补缺点，每次调用每缺失点尝试一次。300点全部执行完成才标记sample_complete；unscored保存NULL。根目录只含subjects/motions/references/filter/scope五项；名称不含policy等。', '''
        if RUN_TRAIN:
            EFFECTIVE_TRAIN_PARAMS = replace(TRAIN_PARAMS)
            EFFECTIVE_PREPROCESSING = replace(TRAIN_PREPROCESSING)
            EFFECTIVE_DETECTOR = replace(TRAIN_DETECTOR)
            EFFECTIVE_ALIGNMENT = replace(TRAIN_ALIGNMENT)
            EFFECTIVE_NUMERICAL = replace(TRAIN_NUMERICAL)
            show_config(build_fixed_config(EFFECTIVE_TRAIN_PARAMS, preprocessing=EFFECTIVE_PREPROCESSING,
                motion_detector=EFFECTIVE_DETECTOR, settings=EFFECTIVE_ALIGNMENT, numerical=EFFECTIVE_NUMERICAL,
                references=REFERENCE_COMBINATIONS[0], target_scope=SCOPES[0]))
            TRAIN_ROOTS = run_physical4d(SUBJECT_DIRS, MOTION_TYPES,
                references=REFERENCE_COMBINATIONS, adaptive_filters=ADAPTIVE_FILTERS, scopes=SCOPES,
                fixed_params=EFFECTIVE_TRAIN_PARAMS, output_parent=OUTPUT_PARENT,
                preprocessing=EFFECTIVE_PREPROCESSING, motion_detector=EFFECTIVE_DETECTOR,
                settings=EFFECTIVE_ALIGNMENT, numerical=EFFECTIVE_NUMERICAL, save_diagnostics=True)
            display(TRAIN_ROOTS)
        ''')
    section(8, '只读 SQLite 状态查询', '完成样本优先跳过；未完成样本可查看缺失cell与attempt。不要手工修改训练状态。', '''
        if RUN_SQL_STATUS:
            ledger = RunLedger(RESULTS_ROOT, readonly=True)
            try:
                display(pd.read_sql_query("SELECT sample_id,subject,motion,sample_complete FROM samples", ledger.connection))
                display(pd.read_sql_query("SELECT sample_id,count(*) AS completed_cells FROM cell_metrics GROUP BY sample_id", ledger.connection))
                display(pd.read_sql_query("SELECT * FROM attempt_events", ledger.connection))
            finally:
                ledger.close()
        ''')
    section(9, '缺失输出补导出', '只读账本重建完整300行P4D、sample_best、motion_summary、overall_summary及缺失诊断图，不重跑detector、Tdelay或300点。', '''
        if RUN_REPAIR_OUTPUTS:
            repair_run_outputs(RESULTS_ROOT, diagnostics=True)
        ''')
    section(10, '单样本严格复现', '先恢复并显示样本状态、自身坐标、配置hash、policy/rescue、Tdelay、保存bias、完整参数及预期指标，再重算。best_posthoc要求完整且有有限最佳AAE；指定坐标可选未完成样本中已完成的cell。保存bias无效保持unscored，不重搜或补零。有效窗口数必须一致，指标绝对容差1e-6；生成双面板HR、windows.csv、metrics.json、receipt.json。', '''
        if RUN_SINGLE_REPLAY:
            REPLAY_SPEC = resolve_replay_spec(RESULTS_ROOT, SAMPLE_ID, COORDINATE_ID)
            REPLAY_EFFECTIVE_PARAMS = REPLAY_SPEC.params
            REPLAY_FIXED_CONFIG = REPLAY_SPEC.fixed_config
            display(REPLAY_SPEC.preview())
            show_config(REPLAY_FIXED_CONFIG)
            SINGLE_RECEIPT = replay_sample(REPLAY_SPEC, directory_mapping=DIRECTORY_MAPPING,
                plot_range_s=PLOT_RANGE_S, use_cache=True)
            display(SINGLE_RECEIPT)
            if (REPLAY_SPEC.output_dir / "hr.png").is_file():
                display(NotebookImage(filename=str(REPLAY_SPEC.output_dir / "hr.png")))
        ''')
    section(11, '同一 ReplaySpec 的窗口诊断', '从样本起点执行完整历史，只收集指定窗完整stage。请求映射最近窗口，允许半hop偏差，超出范围报错；诊断输出包含correlation排名、shared delay/M、group K、实际步长、频谱/penalty/owner、tracker/fusion/rescue。裁图不改变scope评价。', '''
        if RUN_WINDOW_DIAGNOSTICS:
            if REPLAY_SPEC is None:
                REPLAY_SPEC = resolve_replay_spec(RESULTS_ROOT, SAMPLE_ID, COORDINATE_ID)
            DIAGNOSTIC_PARAMS = REPLAY_SPEC.params
            DIAGNOSTIC_FIXED_CONFIG = REPLAY_SPEC.fixed_config
            display(REPLAY_SPEC.preview())
            WINDOW_RECEIPT = replay_sample(REPLAY_SPEC, directory_mapping=DIRECTORY_MAPPING,
                diagnostic_start_s=DIAGNOSTIC_START_S, plot_range_s=PLOT_RANGE_S, use_cache=False)
            display(WINDOW_RECEIPT)
            if WINDOW_RECEIPT.get("status") in {"verified", "metric_mismatch"}:
                show_window_diagnostic(REPLAY_SPEC.output_dir / "diagnostics" / str(DIAGNOSTIC_START_S).replace(".", "_"), DIAGNOSTIC_PARAMS.Fs_Target)
        ''')
    section(12, '从 SQLite 重建汇总', '不运行求解器。best按有限posthoc-oracle final AAE再按coordinate_id选择；汇总保留完成/失败/可评分数，平均AAE/accuracy和样本标准差。各受试者自身结果不被整体汇总代替。', '''
        if RUN_SUMMARY:
            ledger = RunLedger(RESULTS_ROOT, readonly=True)
            try:
                rebuild_summaries(ledger)
            finally:
                ledger.close()
        ''')
    section(13, '独立跨 reference 实验', '复用源样本自身坐标、冻结分段、Tdelay与保存bias。新reference顺序进入独立identity，输出experiments。显式重搜oracle仅记录experiment_posthoc_bias_s，不覆盖精确复现或训练。', '''
        if RUN_CROSS_REFERENCE:
            EXPERIMENT_SPEC = resolve_replay_spec(RESULTS_ROOT, SAMPLE_ID, COORDINATE_ID)
            EXPERIMENT_PARAMS = EXPERIMENT_SPEC.params
            display(EXPERIMENT_SPEC.preview())
            EXPERIMENT_RECEIPTS = [replay_reference_experiment(EXPERIMENT_SPEC, tuple(refs),
                oracle_search=SEARCH_EXPERIMENT_ORACLE, directory_mapping=DIRECTORY_MAPPING)
                for refs in EXPERIMENT_REFERENCES]
            display(EXPERIMENT_RECEIPTS)
        ''')
    section(14, '跨 reference 窗口诊断', '完整历史执行后仅收集选定stage；诊断实验使用独立identity，保持源坐标与时间偏移。', '''
        if RUN_CROSS_DIAGNOSTICS:
            EXPERIMENT_SPEC = resolve_replay_spec(RESULTS_ROOT, SAMPLE_ID, COORDINATE_ID)
            EXPERIMENT_DIAGNOSTIC_PARAMS = EXPERIMENT_SPEC.params
            display(EXPERIMENT_SPEC.preview())
            EXPERIMENT_DIAGNOSTICS = [replay_reference_experiment(EXPERIMENT_SPEC, tuple(refs),
                oracle_search=SEARCH_EXPERIMENT_ORACLE, diagnostic_start_s=DIAGNOSTIC_START_S,
                directory_mapping=DIRECTORY_MAPPING) for refs in EXPERIMENT_REFERENCES]
            display(EXPERIMENT_DIAGNOSTICS)
            for receipt in EXPERIMENT_DIAGNOSTICS:
                show_window_diagnostic(RESULTS_ROOT / "experiments" / (SAMPLE_ID + "__" + receipt["identity"]), EXPERIMENT_DIAGNOSTIC_PARAMS.Fs_Target)
        ''')
    section(15, '逐样本自身坐标的批量 reference 对比', '每个样本分别恢复自身best或指定坐标、分段与两个偏移，再比较reference组合；失败继续。禁止把第一个样本的best参数用于其他样本。输出experiments/batches。', '''
        if RUN_BATCH_REFERENCE_COMPARE:
            REFERENCE_BATCH_SELECTOR = dict(subjects=BATCH_SUBJECTS, motions=BATCH_MOTIONS,
                sample_ids=BATCH_SAMPLE_IDS, coordinate_id=COORDINATE_ID)
            REFERENCE_COMPARISON = replay_reference_batch(RESULTS_ROOT, EXPERIMENT_REFERENCES,
                **REFERENCE_BATCH_SELECTOR, oracle_search=SEARCH_EXPERIMENT_ORACLE,
                directory_mapping=DIRECTORY_MAPPING,
                selection_callback=lambda choices: display(pd.json_normalize(choices)))
            display(REFERENCE_COMPARISON)
        ''')
    section(16, '多 subjects/motions 批量精确复现', '先固定并展示每个样本的坐标清单，串行执行并立即保存receipt、释放内存。身份一致、verified且文件完整才cached_verified；中断补该样本。记录verified/cached_verified/metric_mismatch/failed/skipped，默认best跳过未完成或无有限best。汇总仅统计验证通过样本，保留失败与跳过计数，训练库始终只读。', '''
        if RUN_BATCH_REPLAY:
            EXACT_BATCH_SELECTOR = dict(subjects=BATCH_SUBJECTS, motions=BATCH_MOTIONS,
                sample_ids=BATCH_SAMPLE_IDS, coordinate_id=COORDINATE_ID)
            BATCH_REPORT = replay_batch(RESULTS_ROOT, **EXACT_BATCH_SELECTOR,
                directory_mapping=DIRECTORY_MAPPING, use_cache=True,
                selection_callback=lambda choices: display(pd.json_normalize(choices)))
            display(BATCH_REPORT)
        ''')
    return nbformat.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name':'Python (PPG_sensor_env)', 'language':'python', 'name':'python3'},
        'language_info': {'name':'python', 'version':'3.10'}})


if __name__ == '__main__':
    nbformat.write(build_notebook(), NOTEBOOK)
