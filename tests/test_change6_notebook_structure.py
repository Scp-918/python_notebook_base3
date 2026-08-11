from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import nbformat
import pytest


NOTEBOOK_PATH = Path("notebooks/run_batch_adaptive_protocol.ipynb")


def _notebook() -> nbformat.NotebookNode:
    return nbformat.read(NOTEBOOK_PATH, as_version=4)


def _sources(kind: str) -> list[str]:
    return [str(cell.source) for cell in _notebook().cells if cell.cell_type == kind]


def test_notebook_restores_sections_zero_through_nine() -> None:
    markdown = _sources("markdown")
    headings = [line for source in markdown for line in source.splitlines() if line.startswith("## ")]

    for section in range(10):
        assert any(line.startswith(f"## {section}.") for line in headings), section

    joined = "\n".join(_sources("code"))
    for call in (
        "plot_rest_alignment_diagnostics_by_motion_type(",
        "plot_unaligned_fullfield_ppg_hr_by_motion_type(",
        "plot_raw_ppg_and_unaligned_hr_by_motion_type(",
        "run_batch_adaptive_protocol(",
    ):
        assert call in joined
    assert "84 \u6a21\u5f0f" not in joined


def test_notebook_has_layered_protocol_configuration() -> None:
    code = _sources("code")
    first = code[0]
    joined = "\n".join(code)

    assert "build_subject_output_dir(" in first
    assert "OUTPUT_ROOT = SUBJECT_DIR.parent / \"outputs\"" in first
    assert "RUN_OUTPUT_DIR = build_subject_output_dir(" in first
    for name in (
        "FS_ORIGIN",
        "TW_F",
        "NORMALIZATION_MODE",
        "QC_POLICY",
        "DELAY_ESTIMATION_MODE",
        "ALIGNMENT_TW",
        "PPG_INPUT_TRANSFORM",
        "TRAIN_HR_POSTPROCESS_METHOD",
        "SSR_NUM_ATOMS",
        "RFF_UPDATE_MODE",
        "KLMS_MAX_DICTIONARY_SIZE",
        "CASCADE_GUARD_POLICY",
        "SEARCH_SPACE",
        "TRIAL_PARAM_OVERRIDES",
        "CLEAN_OUTPUTS",
    ):
        assert name in joined

    for flag in (
        "RUN_ALIGNMENT_DIAGNOSTICS",
        "RUN_FULLFIELD_PLOT",
        "RUN_RAW_PPG_PLOT",
        "RUN_ALL_TRAIN",
        "RUN_CONNECTIVITY_CHECK",
    ):
        assert f"{flag} = False" in joined


def test_trial_overrides_do_not_freeze_optuna_search_fields() -> None:
    override_cell = next(source for source in _sources("code") if "TRIAL_PARAM_OVERRIDES =" in source)
    for searched_name in (
        "Fs_Target",
        "LMS_Mu_Base",
        "Rest_HR_Track_Band_BPM",
        "Rest_HR_Slew_Limit_BPM",
        "Rest_HR_Slew_Step_BPM",
        "alpha_u",
        "M2",
        "rff_D",
        "rff_sigma_scale",
        "klms_step_size",
        "klms_sigma",
        "klms_epsilon",
    ):
        assert f'"{searched_name}":' not in override_cell


def test_notebook_documents_cell_risk_and_compiles_zero_through_nine() -> None:
    notebook = _notebook()
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    joined = "\n".join(str(cell.source) for cell in code_cells)

    assert joined.count("# \u7528\u9014：") >= 10
    assert "# \u662f\u5426\u5199\u6587\u4ef6：" in joined
    assert "# \u8017\u65f6\u98ce\u9669：" in joined
    assert "multi_" not in joined
    assert "D:\\" not in joined
    for index, cell in enumerate(code_cells):
        compile(str(cell.source), f"{NOTEBOOK_PATH.name}:cell-{index}", "exec")


def test_notebook_restores_replay_diagnostics_sections_ten_through_fifteen() -> None:
    markdown = _sources("markdown")
    headings = [line for source in markdown for line in source.splitlines() if line.startswith("## ")]
    joined = "\n".join(_sources("code"))

    for section in range(10, 16):
        assert any(line.startswith(f"## {section}.") for line in headings), section
    assert joined.count("replay_best_record_hr_curves(") >= 2
    assert joined.count("plot_window_diagnostics_from_records(") >= 2
    assert "build_cross_motion_summary_table(" in joined
    assert "run_batch_reference_compare(" in joined


def test_replay_uses_subject_pair_selectors_and_run_subdirectories() -> None:
    joined = "\n".join(_sources("code"))

    assert "def select_subject_pair(" in joined
    assert 'REPLAY_MOTION_TYPE = "write"' in joined
    assert "REPLAY_MOTION_INDEX = 1" in joined
    assert "REPLAY_SIGNAL_CSV" not in joined
    assert "REPLAY_REF_CSV" not in joined
    for assignment in (
        'REPLAY_OUTPUT_DIR = Path(RESULTS_ROOT) / "replay"',
        'DIAGNOSTIC_OUTPUT_DIR = Path(RESULTS_ROOT) / "window_diagnostics"',
        'SUMMARY_TABLE_OUTPUT_DIR = Path(RESULTS_ROOT) / "summary_tables"',
        'CROSS_OUTPUT_DIR = Path(RESULTS_ROOT) / "cross_replay"',
        'CROSS_DIAGNOSTIC_OUTPUT_DIR = Path(RESULTS_ROOT) / "cross_window_diagnostics"',
        'REFERENCE_COMPARE_OUTPUT_DIR = Path(RESULTS_ROOT) / "batch_reference_compare"',
    ):
        assert assignment in joined

    for flag in (
        "RUN_STAGE7_REPLAY",
        "RUN_WINDOW_DIAGNOSTICS",
        "RUN_CROSS_MOTION_SUMMARY",
        "RUN_CROSS_STAGE7_REPLAY",
        "RUN_CROSS_WINDOW_DIAGNOSTICS",
        "RUN_BATCH_REFERENCE_COMPARE",
    ):
        assert f"{flag} = False" in joined


def test_notebook_generator_matches_tracked_sources() -> None:
    script_path = Path("scripts/update_change6_notebook.py")
    spec = importlib.util.spec_from_file_location("update_change6_notebook", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    generated = module.build_notebook()
    tracked = _notebook()
    assert generated.metadata == tracked.metadata
    assert [(cell.cell_type, str(cell.source)) for cell in generated.cells] == [
        (cell.cell_type, str(cell.source)) for cell in tracked.cells
    ]


def test_section_two_documents_every_preview_parameter() -> None:
    notebook = _notebook()
    heading_index = next(
        index
        for index, cell in enumerate(notebook.cells)
        if cell.cell_type == "markdown" and str(cell.source).startswith("## 2.")
    )
    markdown_source = str(notebook.cells[heading_index].source)
    code_source = str(notebook.cells[heading_index + 1].source)

    for name in (
        "RUN_PREPROCESS_PREVIEW",
        "PREVIEW_MOTION_TYPE",
        "PREVIEW_MOTION_INDEX",
        "FS_ORIGIN",
        "TW",
        "SUBJECT_DIR",
        "calibration",
    ):
        assert f"`{name}`" in markdown_source
        assert f"# {name}:" in code_source
    assert "write/gripper/run/rope" in markdown_source
    assert "100 Hz" in markdown_source
    assert "Optuna" in markdown_source


def test_layered_config_cells_two_three_four_have_inline_parameter_comments() -> None:
    config_cells = _sources("code")[1:4]
    configurable_line = re.compile(
        r'^\s*(?:[A-Z][A-Z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?\s*=|"[A-Za-z_][A-Za-z0-9_]*"\s*:)'
    )

    missing: list[str] = []
    for cell_number, source in enumerate(config_cells, start=2):
        for line_number, line in enumerate(source.splitlines(), start=1):
            if configurable_line.match(line) and "#" not in line:
                missing.append(f"cell {cell_number}, line {line_number}: {line.strip()}")

    assert not missing, "\n".join(missing)


def test_notebook_uses_reference_phase_tracking_and_dynamic_guard_defaults() -> None:
    joined = "\n".join(_sources("code"))
    override_cell = next(source for source in _sources("code") if "TRIAL_PARAM_OVERRIDES =" in source)

    expected_assignments = (
        "MOTION_TRACKING_RANGE_UP_BPM = 35.0",
        "MOTION_TRACKING_RANGE_DOWN_BPM = 15.0",
        "MOTION_TRACKING_SLEW_LIMIT_UP_BPM = 5.5",
        "MOTION_TRACKING_SLEW_STEP_UP_BPM = 3.5",
        "MOTION_TRACKING_SLEW_LIMIT_DOWN_BPM = 2.0",
        "MOTION_TRACKING_SLEW_STEP_DOWN_BPM = 1.5",
        "RECOVERY_TRACKING_RANGE_UP_BPM = 20.0",
        "RECOVERY_TRACKING_RANGE_DOWN_BPM = 25.0",
        "RECOVERY_TRACKING_SLEW_LIMIT_UP_BPM = 1.5",
        "RECOVERY_TRACKING_SLEW_STEP_UP_BPM = 1.5",
        "RECOVERY_TRACKING_SLEW_LIMIT_DOWN_BPM = 3.5",
        "RECOVERY_TRACKING_SLEW_STEP_DOWN_BPM = 3.0",
        "CANDIDATE_PEAK_THRESHOLD_RATIO = 0.30",
        "FULL_CANDIDATE_PEAK_THRESHOLD_RATIO = 0.15",
        "POST_MOTION_GUARD_SECONDS = None",
        "POST_MOTION_GUARD_RISING_WINDOWS = 3",
        "POST_MOTION_GUARD_RISING_SLOPE_BPM_PER_WINDOW = 1.5",
        "SEARCH_SPACE.Rest_HR_Track_Band_BPM = [20.0, 30.0, 60.0, 80.0]",
        "SEARCH_SPACE.Rest_HR_Slew_Limit_BPM = [1.0, 3.0, 6.0, 8.0]",
        "SEARCH_SPACE.Rest_HR_Slew_Step_BPM = [0.5, 2.0, 4.0]",
    )
    for assignment in expected_assignments:
        assert assignment in joined

    for legacy_name in (
        "TRACKING_RANGE_UP_BPM",
        "TRACKING_RANGE_DOWN_BPM",
        "TRACKING_SLEW_LIMIT_UP_BPM",
        "TRACKING_SLEW_STEP_UP_BPM",
        "TRACKING_SLEW_LIMIT_DOWN_BPM",
        "TRACKING_SLEW_STEP_DOWN_BPM",
    ):
        assert f"{legacy_name} = None" in joined

    for field_name in (
        "motion_tracking_range_up_bpm",
        "recovery_tracking_range_down_bpm",
        "candidate_peak_threshold_ratio",
        "full_candidate_peak_threshold_ratio",
        "post_motion_guard_rising_windows",
        "post_motion_guard_rising_slope_bpm_per_window",
    ):
        assert f'"{field_name}":' in override_cell


def test_readme_distinguishes_motion_post10_from_dynamic_guard_lifetime() -> None:
    readme = Path("notebooks/readme.md").read_text(encoding="utf-8")

    assert "adaptive_rising_rescue" in readme
    assert "motion_post10" in readme
    assert "不作为运动后保护的固定退出时间" in readme


def test_notebook_safe_defaults_execute_without_creating_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_subject = tmp_path / "total_data" / "subject_not_present"
    monkeypatch.setenv("PPG_SUBJECT_DIR", str(missing_subject))
    namespace: dict[str, object] = {"__name__": "__notebook_smoke__"}

    for index, cell in enumerate(_notebook().cells):
        if cell.cell_type == "code":
            exec(compile(str(cell.source), f"safe-smoke-cell-{index}", "exec"), namespace)

    assert not (missing_subject.parent / "outputs").exists()
    for name in (
        "RUN_ALL_TRAIN",
        "RUN_CONNECTIVITY_CHECK",
        "RUN_STAGE7_REPLAY",
        "RUN_WINDOW_DIAGNOSTICS",
        "RUN_CROSS_MOTION_SUMMARY",
        "RUN_CROSS_STAGE7_REPLAY",
        "RUN_CROSS_WINDOW_DIAGNOSTICS",
        "RUN_BATCH_REFERENCE_COMPARE",
    ):
        assert namespace[name] is False
