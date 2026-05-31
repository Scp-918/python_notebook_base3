import json
from pathlib import Path


def test_training_notebook_progress_prints_mode_done_acc3_compare() -> None:
    notebook_path = Path("notebooks/run_batch_adaptive_protocol.ipynb")
    payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in payload.get("cells", [])
        if cell.get("cell_type") == "code"
    )

    assert 'stage == "optimization_mode_done"' in source
    mode_done_source = source.split('stage == "optimization_mode_done"', maxsplit=1)[1].split(
        'stage == "optimization_parallel_repeats_start"',
        maxsplit=1,
    )[0]

    assert "current_posthoc_final_aae_bpm" in mode_done_source
    assert "current_posthoc_final_acc_pct" in mode_done_source
    assert "acc3_compare_posthoc_final_aae_bpm" in mode_done_source
    assert "acc3_compare_posthoc_final_acc_pct" in mode_done_source
    assert "current_final_aae_bpm" not in mode_done_source
    assert "acc3_compare_aae_bpm" not in mode_done_source
    assert "acc3_compare_status" in source


def test_training_notebook_batch_reference_compare_displays_posthoc_columns() -> None:
    notebook_path = Path("notebooks/run_batch_adaptive_protocol.ipynb")
    payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in payload.get("cells", [])
        if cell.get("cell_type") == "code"
    )
    batch_compare_source = source.split("run_batch_reference_compare(", maxsplit=1)[1]

    assert "source_posthoc_final_aae_bpm" in batch_compare_source
    assert "source_posthoc_final_accuracy_pct" in batch_compare_source
    assert "actual_posthoc_final_aae_bpm" in batch_compare_source
    assert "actual_posthoc_final_accuracy_pct" in batch_compare_source
