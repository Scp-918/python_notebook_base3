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
    assert "current_final_aae_bpm" in source
    assert "current_final_acc_pct" in source
    assert "acc3_compare_aae_bpm" in source
    assert "acc3_compare_accuracy_pct" in source
    assert "acc3_compare_status" in source
