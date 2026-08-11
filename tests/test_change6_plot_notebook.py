from __future__ import annotations

from pathlib import Path

import nbformat
import numpy as np

from ppg_hr.experimental.preprocess_protocol import ProtocolDataset
from ppg_hr.experimental.protocol_outputs import _plot_groups, _plot_motion_bandpass
from ppg_hr.experimental.run_batch_protocol import _scan_stage6_motion_dirs
from ppg_hr.experimental.segmentation import SegmentInfo
from ppg_hr.params import CascadeScheme, MotionType


def _dataset() -> ProtocolDataset:
    t = np.arange(300, dtype=float) / 100.0
    wave = np.sin(2 * np.pi * 1.2 * t)
    zeros = np.zeros_like(t)
    return ProtocolDataset(
        sample_stem="subject_name_write_1",
        fs=100,
        time_s=t,
        ppg_green=wave,
        ppg_red=wave,
        ppg_ir=wave,
        hf1=wave,
        hf2=0.8 * wave,
        cf1=0.3 * wave,
        cf2=0.2 * wave,
        hfcomp1=0.9 * wave,
        hfcomp2=0.7 * wave,
        ud1=0.4 * wave,
        ud2=0.5 * wave,
        accx=zeros,
        accy=zeros,
        accz=zeros,
        gyrox=zeros,
        gyroy=zeros,
        gyroz=zeros,
        ref_time_s=np.asarray([0.0, 1.0, 2.0]),
        ref_hr_bpm=np.asarray([72.0, 72.0, 72.0]),
    )


def test_new_signal_groups_and_motion_plot_smoke(tmp_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    groups = _plot_groups()
    assert [item[0] for item in groups[:4]] == ["HF2", "HF2comp", "CF2", "UD2"]
    assert groups[0][2] == ["hf1", "hf2"]
    assert groups[1][2] == ["hfcomp1", "hfcomp2"]
    assert groups[3][2] == ["ud1", "ud2"]
    path = tmp_path / "signals.png"
    segment = SegmentInfo(
        status="ok",
        reason="",
        motion_start_s=0.5,
        motion_end_s=2.5,
        motion_threshold=0.1,
        window_starts_s=np.asarray([0.0]),
        window_centers_s=np.asarray([1.0]),
        window_std=np.asarray([1.0]),
        motion_flags=np.asarray([True]),
        labels=np.asarray(["motion"], dtype=object),
    )

    _plot_motion_bandpass(plt, _dataset(), segment, path, "subject_name_write_1", "write")

    assert path.exists() and path.stat().st_size > 0
    assert [scheme.display_name for scheme in CascadeScheme] == [
        "ACC", "HF2", "UD2", "ACC+HF2", "HF2+CF2", "ACC+UD2"
    ]


def test_stage6_motion_directories_use_canonical_order(tmp_path: Path) -> None:
    root = tmp_path / "motion_types"
    for motion in reversed(list(MotionType)):
        directory = root / motion.value
        directory.mkdir(parents=True)
        (directory / "best_metrics.csv").write_text("motion_type\n" + motion.value, encoding="utf-8")

    assert [path.name for path in _scan_stage6_motion_dirs(tmp_path)] == [
        motion.value for motion in MotionType
    ]


def test_notebook_has_single_subject_and_tracker_switches_and_all_code_compiles() -> None:
    path = Path("notebooks/run_batch_adaptive_protocol.ipynb")
    notebook = nbformat.read(path, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    first = str(code_cells[0].source)

    assert "SUBJECT_DIR" in first
    assert "INPUT_DIR" not in first and "TESTDATA_DIR" not in first
    assert "TRACKER_MODE = \"enhanced\"" in first
    for name in (
        "ENABLE_DIRECTIONAL_TRACKING",
        "ENABLE_DYNAMIC_PENALTY",
        "ENABLE_CONTINUITY_PROTECTION",
        "ENABLE_LOW_LOCK_RECOVERY",
        "ENABLE_HIGH_LOCK_RECOVERY",
        "ENABLE_POST_MOTION_PROTECTION",
    ):
        assert f"{name} = True" in first
    assert all(motion.value in first for motion in MotionType)
    assert "multi_" not in "\n".join(str(cell.source) for cell in code_cells)
    assert "D:\\" not in "\n".join(str(cell.source) for cell in code_cells)
    for index, cell in enumerate(code_cells):
        compile(str(cell.source), f"{path.name}:cell-{index}", "exec")
