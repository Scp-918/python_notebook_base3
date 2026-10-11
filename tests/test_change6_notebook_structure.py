from pathlib import Path
import importlib.util
import nbformat

NOTEBOOK_PATH = Path("notebooks/run_batch_adaptive_protocol.ipynb")


def test_physical4d_sections_and_snapshot_replay_contract():
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    headings = [str(c.source).splitlines()[0] for c in notebook.cells if c.cell_type == "markdown"]
    for number in range(17):
        assert any(h.startswith(f"## {number}.") for h in headings)
    joined = "\n".join(str(c.source) for c in notebook.cells if c.cell_type == "code")
    for call in ("run_physical4d(","resolve_replay_spec(","replay_sample(","replay_batch(",
                 "rebuild_summaries(","replay_reference_experiment(","replay_reference_batch("):
        assert call in joined
    assert "run_batch_adaptive_protocol(" not in joined
    assert "reference_parity_params(" in joined
    assert "REPLAY_EFFECTIVE_PARAMS = REPLAY_SPEC.params" in joined
    assert "DIAGNOSTIC_PARAMS = REPLAY_SPEC.params" in joined
    assert "selection_callback=" in joined


def test_generator_matches_and_cells_are_clean():
    spec = importlib.util.spec_from_file_location("generator", "scripts/update_change6_notebook.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    generated = module.build_notebook()
    tracked = nbformat.read(NOTEBOOK_PATH, as_version=4)
    assert [(c.cell_type,c.source) for c in generated.cells] == [(c.cell_type,c.source) for c in tracked.cells]
    for i, cell in enumerate(tracked.cells):
        if cell.cell_type == "code":
            assert not cell.outputs and cell.execution_count is None
            compile(cell.source, f"cell-{i}", "exec")
            assert "D:\\" not in cell.source


def test_all_disabled_notebook_executes_without_results_or_subject(tmp_path, monkeypatch):
    absent = tmp_path / "absent" / "S01"
    monkeypatch.setenv("PPG_SUBJECT_DIR", str(absent))
    namespace = {"__name__":"__notebook_smoke__"}
    for i, cell in enumerate(nbformat.read(NOTEBOOK_PATH,as_version=4).cells):
        if cell.cell_type == "code":
            exec(compile(cell.source,f"cell-{i}","exec"),namespace)
    assert all(value is False for key,value in namespace.items() if key.startswith("RUN_"))
    assert not absent.parent.exists()


def test_fixed_parameters_have_catalog_and_policy_explanation():
    notebook = nbformat.read(NOTEBOOK_PATH,as_version=4)
    markdown = "\n".join(c.source for c in notebook.cells if c.cell_type == "markdown")
    for text in ("fixed_hyperparameters.json", "suppressed_protected_continuous_visibility",
                 "soft_visible", "Tdelay_s", "posthoc_bias_s", "1e-6", "rff_seed", "启用条件"):
        assert text in markdown
    readme = Path("notebooks/readme.md").read_text(encoding="utf-8")
    assert "motion_post10" in readme and "SQLite" in readme and "reset-FFT" in readme
