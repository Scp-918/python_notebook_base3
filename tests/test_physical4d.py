from dataclasses import replace

import pytest

from ppg_hr.experimental.physical4d import (
    build_fixed_config, coordinates, load_snapshot, save_snapshot, trial_from_snapshot,
    run_name,
)
from ppg_hr.params import ProtocolTrialParams


def test_grid_and_snapshot_are_strict(tmp_path):
    for filter_name in ('lms', 'volterra', 'rff_lms', 'klms'):
        grid = coordinates(filter_name)
        assert len(grid) == 300
        assert grid[0]['coordinate_id'] == 'p4d_000'
        assert grid[-1]['coordinate_id'] == 'p4d_299'
        assert grid[-1]['max_order'] == 20
        assert grid[1]['penalty_half_width_bpm'] == 6
        config = build_fixed_config(ProtocolTrialParams(adaptive_filter=filter_name))
        root = tmp_path / filter_name
        digest = save_snapshot(root, config)
        restored = load_snapshot(root)
        assert restored == config
        trial = trial_from_snapshot(restored, grid[0])
        assert trial.Fs_Target == 25
        assert trial.history_tap_limit == 1
        assert digest != save_snapshot(tmp_path / (filter_name + '_other'),
                                       build_fixed_config(replace(trial, smooth_win_len=9)))
        del restored['trial']['smooth_win_len']
        with pytest.raises(ValueError):
            trial_from_snapshot(restored, grid[0])


def test_compact_name():
    assert run_name(['S02', 'S01'], ['run', 'write'], ('ACC', 'HF'),
                    'lms', 'motion_post10') == 'S01-S02__wr__AH__L__mp10'


def test_snapshot_tampering_policy_and_changed_defaults_are_detected(tmp_path, monkeypatch):
    import json
    from dataclasses import dataclass, fields
    import ppg_hr.experimental.physical4d as module
    from ppg_hr.params import reference_parity_params
    params = reference_parity_params()
    config = build_fixed_config(params)
    assert set(config['trial']) == {f.name for f in fields(ProtocolTrialParams)} - module.search_fields('lms')
    assert {'rff_seed', 'klms_epsilon', 'M2', 'dual_reset_reanchor_prior_guard_bpm'} <= set(config['trial'])
    digest = save_snapshot(tmp_path / 'soft', config)
    hard = build_fixed_config(replace(params, spectral_penalty_policy='suppressed_protected_continuous_visibility'))
    assert module.identity(hard) != digest
    with pytest.raises(ValueError):
        save_snapshot(tmp_path / 'soft', hard)
    @dataclass(frozen=True)
    class ChangedDefaults(ProtocolTrialParams):
        smooth_win_len: int = 99
        rff_seed: int = 123
        rff_err_clip: float | None = 10.0
    monkeypatch.setattr(module, 'ProtocolTrialParams', ChangedDefaults)
    restored = trial_from_snapshot(load_snapshot(tmp_path / 'soft'), coordinates('lms')[0])
    assert restored.smooth_win_len == params.smooth_win_len and restored.rff_seed == params.rff_seed
    assert restored.rff_err_clip is None
    path = tmp_path / 'soft/config/fixed_hyperparameters.json'
    tampered = json.loads(path.read_text(encoding='utf-8'))
    tampered['trial']['smooth_win_len'] = 9
    path.write_text(json.dumps(tampered), encoding='utf-8')
    with pytest.raises(ValueError, match='hash'):
        load_snapshot(tmp_path / 'soft')


def test_implementation_identity_ignores_defaults_but_detects_algorithm_edits(monkeypatch):
    from pathlib import Path
    from ppg_hr.experimental.physical4d_run import implementation_identity
    original = Path.read_bytes
    before = implementation_identity()
    def changed_defaults(path):
        value = original(path)
        if path.name == 'params.py':
            value = value.replace(b'rff_seed: int = 0', b'rff_seed: int = 123')
        return value
    monkeypatch.setattr(Path, 'read_bytes', changed_defaults)
    assert implementation_identity() == before
    def changed_algorithm(path):
        value = changed_defaults(path)
        if path.name == 'noncausal_lms.py':
            value = value.replace(b'+ eps)', b'+ eps * 2)')
        return value
    monkeypatch.setattr(Path, 'read_bytes', changed_algorithm)
    assert implementation_identity() != before
