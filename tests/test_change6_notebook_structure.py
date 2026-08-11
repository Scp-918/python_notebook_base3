from __future__ import annotations

from pathlib import Path

import nbformat


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
