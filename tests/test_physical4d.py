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
