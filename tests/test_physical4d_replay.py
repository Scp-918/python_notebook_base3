import json
import numpy as np
from ppg_hr.experimental.physical4d_run import RunLedger, result_metrics, file_identity
from ppg_hr.experimental.physical4d import build_fixed_config
from ppg_hr.experimental.physical4d_replay import resolve_replay_spec, replay_sample, replay_batch
from ppg_hr.params import reference_parity_params
from ppg_hr.experimental.sample_context import prepare_sample_context
from ppg_hr.experimental.cascade_solver import run_protocol_trial, clear_all_caches
from test_core_phase1 import dataset, source


def test_each_sample_restores_own_coordinate_bias_and_verified_cache(tmp_path, monkeypatch):
    root = tmp_path / 'run'
    params = reference_parity_params(Fs_Target=25)
    ledger = RunLedger.create(root, build_fixed_config(params), ('ACC',), 'global', ['S01'], ['run'])
    data_by_id = {}
    for index, coordinate_index, bias in [(1, 0, -2.), (2, 7, 3.)]:
        data = dataset(source(duration=32, bouts=()))
        data.sample_stem = f'S01_run_{index}'
        data_by_id[data.sample_stem] = data
        paths = {}
        for name in ('sensor', 'reference', 'calibration'):
            path = tmp_path / f'{index}_{name}.txt'
            path.write_text(name)
            paths[name] = file_identity(path)
        context = prepare_sample_context(data, params, paths)
        ledger.add_sample(data.sample_stem, 'S01', 'run', paths, context.to_dict())
        coordinate = ledger.manifest['coordinates'][coordinate_index]
        from ppg_hr.experimental.physical4d import trial_from_snapshot
        trial = trial_from_snapshot(ledger.config, coordinate)
        result = run_protocol_trial(data, ('ACC',), 'global', trial, sample_context=context,
                                    posthoc_mode='saved', saved_posthoc_bias_s=bias,
                                    collect_stages=False)
        ledger.record_cell(data.sample_stem, coordinate, result_metrics(result))
        for other in ledger.manifest['coordinates']:
            if other['coordinate_id'] != coordinate['coordinate_id']:
                ledger.record_cell(data.sample_stem, other, dict(tdelay_aae_bpm=None, posthoc_aae_bpm=None))
        assert ledger.sample_complete(data.sample_stem)
        clear_all_caches(data)
    ledger.connection.commit()
    ledger.close()
    specs = [resolve_replay_spec(root, name) for name in data_by_id]
    assert [s.coordinate['coordinate_id'] for s in specs] == ['p4d_000', 'p4d_007']
    assert [s.posthoc_bias_s for s in specs] == [-2., 3.]
    def loader(spec):
        return data_by_id[spec.sample_id]
    monkeypatch.setattr('ppg_hr.experimental.cascade_solver.search_time_bias_after', lambda *a, **k: (_ for _ in ()).throw(AssertionError('oracle search')))
    first = replay_sample(specs[0], dataset_loader=loader)
    assert first['status'] == 'verified'
    assert json.loads((specs[0].output_dir / 'receipt.json').read_text())['identity'] == specs[0].identity
    batch = replay_batch(root, dataset_loader=loader)
    assert batch.status.tolist() == ['cached_verified', 'verified']
    assert replay_batch(root, dataset_loader=loader).status.tolist() == ['cached_verified', 'cached_verified']
    assert np.isfinite(batch.posthoc_aae_bpm).all()
    # Missing files require recomputation; a failed sample must not stop its successor.
    (specs[0].output_dir / 'metrics.json').unlink()
    def broken_loader(spec):
        if spec.sample_id == specs[0].sample_id:
            raise RuntimeError('interrupted input load')
        return loader(spec)
    assert replay_batch(root, dataset_loader=broken_loader, use_cache=False).status.tolist() == ['failed', 'verified']
    assert replay_batch(root, dataset_loader=loader).status.tolist() == ['verified', 'cached_verified']
    from dataclasses import replace
    mismatched = replace(specs[0], expected_metrics=dict(specs[0].expected_metrics, n_valid=999))
    assert replay_sample(mismatched, dataset_loader=loader, use_cache=False)['status'] == 'metric_mismatch'
