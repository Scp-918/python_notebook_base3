from ppg_hr.experimental.physical4d import build_fixed_config, coordinates
from ppg_hr.experimental.physical4d_run import RunLedger, execute_sample_cells
from ppg_hr.params import reference_parity_params


def test_three_hundred_transactional_cells_and_missing_only_resume(tmp_path):
    config = build_fixed_config(reference_parity_params())
    ledger = RunLedger.create(tmp_path, config, ('ACC',), 'global', ['S01'], ['run'])
    ledger.add_sample('S01_run_1', 'S01', 'run', {'input': 'abc'}, {'Tdelay_s': 0})
    calls = []
    def runner(coordinate):
        calls.append(coordinate['coordinate_id'])
        if coordinate['coordinate_id'] == 'p4d_005' and len(calls) < 301:
            raise RuntimeError('temporary failure')
        return dict(tdelay_aae_bpm=1., posthoc_aae_bpm=0.5, posthoc_bias_s=2., n_valid=4)
    execute_sample_cells(ledger, 'S01_run_1', runner)
    assert not ledger.sample_complete('S01_run_1')
    assert len(ledger.cells('S01_run_1')) == 299
    execute_sample_cells(ledger, 'S01_run_1', runner)
    assert calls[-1] == 'p4d_005' and len(calls) == 301
    assert ledger.sample_complete('S01_run_1')
    execute_sample_cells(ledger, 'S01_run_1', runner)
    assert len(calls) == 301
    assert len(ledger.connection.execute('select * from attempt_events').fetchall()) == 1
    assert len(coordinates('lms')) == 300
    from ppg_hr.experimental.physical4d_export import export_sample_grid, rebuild_summaries
    import pandas as pd
    path = export_sample_grid(ledger, 'S01_run_1')
    assert len(pd.read_excel(path)) == 300
    assert rebuild_summaries(ledger).coordinate_id.tolist() == ['p4d_000']
    ledger.close()
