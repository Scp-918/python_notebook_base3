"""Rebuild the batch notebook with the change6 single-subject contract."""

from __future__ import annotations

from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "run_batch_adaptive_protocol.ipynb"


def code(source: str):
    return nbformat.v4.new_code_cell(source.strip() + "\n")


def markdown(source: str):
    return nbformat.v4.new_markdown_cell(source.strip() + "\n")


cells = [
    markdown(
        """
# change6 单受试者自适应 PPG-HR 协议

本 Notebook 只接受一个 `SUBJECT_DIR`。文件名必须为
`{subject}_{motion}_{index}_sensor.csv` 与 `{subject}_{motion}_{index}_HRdata.csv`，
其中 motion 为 `write/gripper/run/rope`。默认配置只运行 LMS；正式训练前先运行静态发现与标定检查。
"""
    ),
    code(
        r'''
from pathlib import Path
import os
import sys

PROJECT_ROOT = Path.cwd().resolve()
if PROJECT_ROOT.name.lower() == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ppg_hr.params import CascadeScheme, MotionType, TargetScope
from ppg_hr.experimental.batch_pairing import discover_sample_pairs_with_unpaired
from ppg_hr.experimental.preprocess_protocol import load_and_preprocess_protocol
from ppg_hr.experimental.run_batch_protocol import run_batch_adaptive_protocol
from ppg_hr.preprocess.calibration import load_subject_calibration

# 唯一数据输入；可通过环境变量覆盖，不在仓库中保存本机绝对路径。
SUBJECT_DIR = Path(os.environ.get("PPG_SUBJECT_DIR", PROJECT_ROOT / "subject_data")).resolve()
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "change6"
MOTION_TYPES = ("write", "gripper", "run", "rope")
assert MOTION_TYPES == tuple(item.value for item in MotionType)

TRACKER_MODE = "enhanced"
ENABLE_DIRECTIONAL_TRACKING = True
ENABLE_DYNAMIC_PENALTY = True
ENABLE_CONTINUITY_PROTECTION = True
ENABLE_LOW_LOCK_RECOVERY = True
ENABLE_HIGH_LOCK_RECOVERY = True
ENABLE_POST_MOTION_PROTECTION = True

TARGET_SCOPES = [TargetScope.MOTION_POST10.value]
CASCADE_SCHEMES = [
    CascadeScheme.ACC.value,
    CascadeScheme.HF2.value,
    CascadeScheme.UD2.value,
    CascadeScheme.ACC_HF2.value,
    CascadeScheme.HF2_CF2.value,
    CascadeScheme.ACC_UD2.value,
]
ADAPTIVE_FILTERS = ["lms"]
MAX_ITERATIONS = 200
NUM_REPEATS = 1
RANDOM_STATE = 42
OBJECTIVE_MODE = "posthoc_aae"
DATA_SPLIT_MODE = "all_train"
PPG_INPUT_TRANSFORM = "log_absorbance"

TRACKING_OVERRIDES = {
    "tracker_mode": TRACKER_MODE,
    "enable_directional_tracking": ENABLE_DIRECTIONAL_TRACKING,
    "enable_dynamic_penalty": ENABLE_DYNAMIC_PENALTY,
    "enable_continuity_protection": ENABLE_CONTINUITY_PROTECTION,
    "enable_low_lock_recovery": ENABLE_LOW_LOCK_RECOVERY,
    "enable_high_lock_recovery": ENABLE_HIGH_LOCK_RECOVERY,
    "enable_post_motion_protection": ENABLE_POST_MOTION_PROTECTION,
    "ppg_input_transform": PPG_INPUT_TRANSFORM,
}
RUN_TRAINING = False
'''
    ),
    markdown("## 1. 只读发现、严格配对与标定检查"),
    code(
        r'''
if not SUBJECT_DIR.is_dir():
    print("请先设置 PPG_SUBJECT_DIR 或修改 SUBJECT_DIR:", SUBJECT_DIR)
    discovery = None
    calibration = None
else:
    calibration_logs = []
    calibration = load_subject_calibration(SUBJECT_DIR, on_log=calibration_logs.append)
    discovery = discover_sample_pairs_with_unpaired(SUBJECT_DIR)
    print(*calibration_logs, sep="\n")
    print("配对数:", len(discovery.pairs))
    print("运动计数:", {
        motion: sum(pair.motion_type == motion for pair in discovery.pairs)
        for motion in MOTION_TYPES
    })
    for issue in discovery.unpaired:
        print(issue.category, issue.file_name, issue.reason)
'''
    ),
    markdown("## 2. 单对静态预处理检查（不训练）"),
    code(
        r'''
if discovery is not None and discovery.pairs:
    pair = discovery.pairs[0]
    preview = load_and_preprocess_protocol(
        pair.sensor_csv,
        pair.ref_csv,
        fs_origin=100,
        calibration=calibration,
    )
    print(pair.sample_id, preview.to_frame().columns.tolist())
    print(preview.source_metadata)
else:
    print("没有可预览的完整配对。")
'''
    ),
    markdown("## 3. 正式模式级优化（默认关闭，避免误跑 200 trials）"),
    code(
        r'''
if RUN_TRAINING:
    result = run_batch_adaptive_protocol(
        subject_dir=SUBJECT_DIR,
        output_root=OUTPUT_ROOT,
        max_iterations=MAX_ITERATIONS,
        num_repeats=NUM_REPEATS,
        random_state=RANDOM_STATE,
        target_scopes=TARGET_SCOPES,
        cascade_schemes=CASCADE_SCHEMES,
        adaptive_filters=ADAPTIVE_FILTERS,
        objective_mode=OBJECTIVE_MODE,
        data_split_mode=DATA_SPLIT_MODE,
        trial_param_overrides=TRACKING_OVERRIDES,
    )
    print(result.output_root)
else:
    print("RUN_TRAINING=False：未创建 Optuna study，未运行训练。")
'''
    ),
    markdown(
        """
## 4. 回放、窗口诊断与跨运动汇总

训练完成后使用 `replay_best_record_hr_curves`、`plot_window_diagnostics_from_records`
和 `build_cross_motion_summary_table`。这些接口会从 `param_*`/JSON 恢复 tracker 总开关、
六项独立开关、六种 scheme 和 `write/gripper/run/rope` 元数据，不会重新执行 Optuna。
"""
    ),
]

notebook = nbformat.v4.new_notebook(cells=cells)
notebook.metadata["kernelspec"] = {
    "display_name": "PPG_sensor_env",
    "language": "python",
    "name": "python3",
}
notebook.metadata["language_info"] = {"name": "python", "version": "3.10"}
nbformat.write(notebook, NOTEBOOK)
