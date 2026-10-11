"""Strict single/batch replay from fixed snapshots and read-only SQLite records."""
from __future__ import annotations
from dataclasses import dataclass, replace
import gc
import json
from pathlib import Path
from typing import Any, Callable
import numpy as np
import pandas as pd
from ppg_hr.params import ProtocolTrialParams, PreprocessingParams
from .physical4d import canonical_json, identity, trial_from_snapshot
from .physical4d_run import RunLedger, file_identity, json_safe, result_metrics
from .physical4d_export import best_cell
from .sample_context import SampleContext
from .cascade_solver import run_protocol_trial, clear_all_caches
from .preprocess_protocol import load_and_preprocess_protocol


@dataclass(frozen=True)
class ReplaySpec:
    root: Path
    sample_id: str
    subject: str
    motion: str
    sample_complete: bool
    fixed_config: dict[str, Any]
    coordinate: dict[str, Any]
    params: ProtocolTrialParams
    context: SampleContext
    references: tuple[str, ...]
    scope: str
    input_identity: dict[str, Any]
    expected_metrics: dict[str, Any]
    posthoc_bias_s: float | None
    fixed_config_hash: str
    cell_identity: str
    identity: str

    @property
    def output_dir(self) -> Path:
        return self.root / self.subject / 'replay' / (self.sample_id + '__' + self.coordinate['coordinate_id'])

    def preview(self) -> dict[str, Any]:
        """Notebook preview before any recomputation or data loading."""
        return dict(sample_id=self.sample_id, sample_complete=self.sample_complete,
            coordinate=self.coordinate, fixed_config_hash=self.fixed_config_hash,
            Tdelay_s=self.context.Tdelay_s, posthoc_bias_s=self.posthoc_bias_s,
            parameters=self.params.to_dict(), expected_metrics=self.expected_metrics,
            cell_identity=self.cell_identity, replay_identity=self.identity)


def resolve_replay_spec(root: Path, sample_id: str, coordinate_id: str = 'best_posthoc') -> ReplaySpec:
    """Restore a finished cell; default best requires the sample completion marker."""
    ledger = RunLedger(root, readonly=True)
    try:
        sample = ledger.connection.execute('SELECT * FROM samples WHERE sample_id=?', (sample_id,)).fetchone()
        if not sample:
            raise ValueError(f'Unknown sample: {sample_id}')
        if coordinate_id == 'best_posthoc':
            cell = best_cell(ledger, sample_id)
            if not cell:
                raise ValueError('Best replay requires a complete sample with finite posthoc AAE')
        else:
            row = ledger.connection.execute('SELECT * FROM cell_metrics WHERE sample_id=? AND coordinate_id=?',
                                            (sample_id, coordinate_id)).fetchone()
            if row is None:
                raise ValueError('Requested coordinate has no completed cell')
            cell = dict(row)
        row = ledger.connection.execute('SELECT coordinate_json FROM coordinates WHERE coordinate_id=?',
                                        (cell['coordinate_id'],)).fetchone()
        coordinate = json.loads(row[0])
        canonical = next((c for c in ledger.manifest['coordinates'] if c['coordinate_id'] == cell['coordinate_id']), None)
        if coordinate != canonical:
            raise ValueError('SQLite coordinate differs from manifest')
        params = trial_from_snapshot(ledger.config, coordinate)
        if canonical_json(params.to_dict()) != cell['parameters_json']:
            raise ValueError('Effective parameters differ from cell audit snapshot')
        if cell['fixed_config_hash'] != ledger.manifest['fixed_config_hash']:
            raise ValueError('Cell fixed configuration hash differs')
        context_value = json.loads(sample['context_json'])
        inputs = json.loads(sample['input_json'])
        cell_id = identity(dict(run=ledger.manifest['run_identity'], sample=sample_id,
                               input=inputs, context=context_value, coordinate=coordinate))
        if cell_id != cell['cell_identity']:
            raise ValueError('Cell identity does not match saved sample and coordinate')
        expected = json.loads(cell['metrics_json'])
        replay_id = identity(dict(cell_identity=cell_id, expected_metrics=expected,
                                 fixed_config_hash=cell['fixed_config_hash'], receipt_schema=1))
        return ReplaySpec(Path(root), sample_id, sample['subject'], sample['motion'], bool(sample['sample_complete']),
            ledger.config, coordinate, params, SampleContext.from_dict(context_value),
            tuple(ledger.manifest['reference_groups_order']), ledger.manifest['target_scope'], inputs,
            expected, cell['posthoc_bias_s'], cell['fixed_config_hash'], cell_id, replay_id)
    finally:
        ledger.close()


def _mapped_path(path: str, directory_mapping: dict | None) -> Path:
    original = Path(path)
    for old, new in sorted((directory_mapping or {}).items(), key=lambda item: -len(str(item[0]))):
        try:
            relative = original.relative_to(Path(old))
        except ValueError:
            continue
        return Path(new) / relative
    return original


def verify_inputs(spec: ReplaySpec, directory_mapping: dict | None = None) -> dict[str, Path]:
    """Relocation is allowed only when sensor/reference/calibration contents are identical."""
    paths = {}
    for name, expected in spec.input_identity.items():
        path = _mapped_path(expected['path'], directory_mapping)
        actual = file_identity(path)
        if actual['sha256'] != expected['sha256'] or actual['bytes'] != expected['bytes']:
            raise ValueError(f'Input identity mismatch: {name}')
        paths[name] = path
    return paths


def load_replay_dataset(spec: ReplaySpec, directory_mapping=None):
    paths = verify_inputs(spec, directory_mapping)
    if paths['calibration'].resolve() != (paths['sensor'].parent.parent / 'ck.mat').resolve():
        raise ValueError('Mapped calibration must remain at the sensor subject parent / ck.mat')
    preprocessing = PreprocessingParams(**spec.fixed_config['preprocessing'])
    dataset = load_and_preprocess_protocol(paths['sensor'], paths['reference'],
        fs_origin=preprocessing.fs_origin, preprocessing=preprocessing,
        ud_calculation_mode=preprocessing.ud_calculation_mode,
        ud_smoothing_window_s=preprocessing.ud_smoothing_window_s)
    dataset.sample_stem = spec.sample_id
    return dataset


def compare_metrics(expected: dict, actual: dict, tolerance: float) -> dict[str, Any]:
    """Window counts match exactly; finite metrics use absolute tolerance only."""
    differences = {}
    for name in ('tdelay_aae_bpm', 'tdelay_accuracy_pct', 'posthoc_aae_bpm',
                 'posthoc_accuracy_pct', 'n_valid', 'posthoc_n_valid', 'posthoc_bias_s'):
        saved, recomputed = expected.get(name), actual.get(name)
        if saved is None or recomputed is None:
            same = saved is None and recomputed is None
        elif name in {'n_valid', 'posthoc_n_valid'}:
            same = saved == recomputed
        else:
            same = abs(saved - recomputed) <= tolerance
        if not same:
            differences[name] = dict(saved=saved, recomputed=recomputed)
    return differences


def _valid_receipt(spec: ReplaySpec, directory: Path) -> dict | None:
    path = directory / 'receipt.json'
    if not path.is_file():
        return None
    try:
        receipt = json.loads(path.read_text(encoding='utf-8'))
        if receipt.get('identity') != spec.identity or receipt.get('status') != 'verified':
            return None
        required = {'hr.png', 'windows.csv', 'metrics.json'}
        if set(receipt.get('files', {})) != required:
            return None
        for name, expected in receipt['files'].items():
            if file_identity(directory / name)['sha256'] != expected:
                return None
        return receipt
    except (OSError, ValueError, KeyError):
        return None


def _write_hr_plot(path: Path, result, spec: ReplaySpec, *, plot_range_s=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    arrays = result.metric_arrays
    time = arrays['time_s']
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for ax, ref_key, aae, acc, title in (
        (axes[0], 'ref_hr_bpm', result.final_aae_bpm, result.final_acc_pct, 'Tdelay'),
        (axes[1], 'ref_hr_after_bpm', result.posthoc_final_aae_bpm, result.posthoc_final_acc_pct, 'saved posthoc')):
        ax.plot(time, arrays['final_hr_bpm'], label='Final')
        ax.plot(time, arrays['baseline_hr_bpm'], label='Baseline', alpha=.5)
        if ref_key in arrays:
            ax.plot(time, arrays[ref_key], label='Reference')
        ax.set(ylabel='HR (bpm)', title=f'{spec.sample_id} {title}; scope={spec.scope}; AAE={aae:.6g}, accuracy={acc:.6g}%')
        ax.legend()
        if plot_range_s:
            ax.set_xlim(*plot_range_s)
    axes[-1].set_xlabel('Sensor window center (s)')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def execute_replay(spec: ReplaySpec, dataset, *, diagnostic_start_s: float | None = None,
                   references=None, posthoc_mode='saved'):
    """Execute from sample start, collect full stages only at the requested window."""
    selected = None
    actual_start = None
    if diagnostic_start_s is not None:
        last_start = len(dataset.time_s) / dataset.fs - spec.params.hr_window_s
        starts = np.arange(0., max(0., last_start) + 1e-9, spec.params.hr_step_s) if last_start >= 0 else np.array([])
        if not starts.size:
            raise ValueError('No complete diagnostic window')
        index = int(np.argmin(abs(starts - diagnostic_start_s)))
        actual_start = float(starts[index])
        if abs(actual_start - diagnostic_start_s) > spec.params.hr_step_s / 2 + 1e-9:
            raise ValueError('Diagnostic request outside half-hop window tolerance')
        selected = {index}
    result = run_protocol_trial(dataset, references or spec.references, spec.scope, spec.params,
        collect_frame=True, collect_stages=selected is not None, sample_context=spec.context,
        posthoc_mode=posthoc_mode, saved_posthoc_bias_s=spec.posthoc_bias_s,
        diagnostic_window_indices=selected)
    return result, actual_start


def replay_sample(spec: ReplaySpec, *, directory_mapping=None, dataset_loader: Callable | None = None,
                  use_cache=True, plot_range_s=None, diagnostic_start_s=None) -> dict[str, Any]:
    """Verify one recorded cell and immediately persist all files plus a final receipt."""
    directory = spec.output_dir
    if diagnostic_start_s is not None:
        directory = directory / 'diagnostics' / str(diagnostic_start_s).replace('.', '_')
    directory.mkdir(parents=True, exist_ok=True)
    dataset = None
    try:
        verify_inputs(spec, directory_mapping)
        cached = _valid_receipt(spec, directory) if use_cache and diagnostic_start_s is None else None
        if cached:
            return dict(sample_id=spec.sample_id, coordinate_id=spec.coordinate['coordinate_id'], status='cached_verified',
                        **cached['actual_metrics'])
        dataset = dataset_loader(spec) if dataset_loader else load_replay_dataset(spec, directory_mapping)
        result, actual_start = execute_replay(spec, dataset, diagnostic_start_s=diagnostic_start_s)
        try:
            actual = result_metrics(result, spec.fixed_config)
            differences = compare_metrics(spec.expected_metrics, actual, spec.fixed_config['evaluation']['replay_metric_atol'])
            status = 'metric_mismatch' if differences else 'verified'
            result.frame.to_csv(directory / 'windows.csv', index=False)
            (directory / 'metrics.json').write_text(canonical_json(dict(saved=spec.expected_metrics,
                recomputed=actual, differences=differences)), encoding='utf-8')
            _write_hr_plot(directory / 'hr.png', result, spec, plot_range_s=plot_range_s)
            if actual_start is not None:
                selected = result.frame.iloc[int(round(actual_start / spec.params.hr_step_s))].to_dict()
                selected['requested_start_s'] = diagnostic_start_s
                selected['actual_start_s'] = actual_start
                (directory / 'window_diagnostic.json').write_text(canonical_json(json_safe(selected)), encoding='utf-8')
            receipt = dict(schema_version=1, identity=spec.identity, status=status,
                fixed_config_hash=spec.fixed_config_hash, cell_identity=spec.cell_identity,
                sample_id=spec.sample_id, coordinate=spec.coordinate,
                Tdelay_s=spec.context.Tdelay_s, posthoc_bias_s=spec.posthoc_bias_s,
                parameters=spec.params.to_dict(), actual_metrics=actual, differences=differences,
                actual_diagnostic_start_s=actual_start,
                files={name: file_identity(directory / name)['sha256'] for name in ('hr.png','windows.csv','metrics.json')})
            (directory / 'receipt.json').write_text(canonical_json(receipt), encoding='utf-8')
            return dict(sample_id=spec.sample_id, coordinate_id=spec.coordinate['coordinate_id'], status=status, **actual)
        finally:
            del result
    except Exception as exc:
        receipt = dict(identity=spec.identity, status='failed', fixed_config_hash=spec.fixed_config_hash,
                       sample_id=spec.sample_id, error=f'{type(exc).__name__}: {exc}')
        (directory / 'receipt.json').write_text(canonical_json(receipt), encoding='utf-8')
        return receipt
    finally:
        if dataset is not None:
            clear_all_caches(dataset)
        dataset = None
        gc.collect()


def replay_batch(root, *, subjects=None, motions=None, sample_ids=None, coordinate_id='best_posthoc',
                 directory_mapping=None, dataset_loader=None, use_cache=True) -> pd.DataFrame:
    """Freeze all sample-specific coordinate choices, then run serially with receipt resume."""
    def accepts(value, selection):
        return selection is None or value in ([selection] if isinstance(selection, str) else selection)
    ledger = RunLedger(root, readonly=True)
    try:
        selected = [dict(row) for row in ledger.connection.execute('SELECT * FROM samples ORDER BY sample_id')
            if accepts(row['subject'], subjects) and accepts(row['motion'], motions) and accepts(row['sample_id'], sample_ids)]
    finally:
        ledger.close()
    specs, records, choices = [], [], []
    for sample in selected:
        try:
            spec = resolve_replay_spec(root, sample['sample_id'], coordinate_id)
            specs.append(spec)
            choices.append(spec.preview())
        except Exception as exc:
            status = 'skipped' if coordinate_id == 'best_posthoc' and (not sample['sample_complete'] or 'finite posthoc' in str(exc)) else 'failed'
            records.append(dict(sample_id=sample['sample_id'], status=status, error=str(exc)))
            choices.append(dict(sample_id=sample['sample_id'], status=status, error=str(exc)))
    selection_hash = identity(choices)
    directory = Path(root) / 'replay/batches' / selection_hash
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'selection.json').write_text(canonical_json(choices), encoding='utf-8')
    for spec in specs:
        records.append(replay_sample(spec, directory_mapping=directory_mapping, dataset_loader=dataset_loader,
                                     use_cache=use_cache))
        pd.DataFrame(records).to_csv(directory / 'progress.csv', index=False)
    frame = pd.DataFrame(records).sort_values('sample_id') if records else pd.DataFrame(columns=['sample_id','status'])
    verified = frame[frame.status.isin(['verified','cached_verified'])]
    with pd.ExcelWriter(directory / 'batch_summary.xlsx') as writer:
        frame.to_excel(writer, sheet_name='all_statuses', index=False)
        verified.to_excel(writer, sheet_name='verified_samples', index=False)
        frame.status.value_counts().rename_axis('status').reset_index(name='count').to_excel(writer, sheet_name='counts', index=False)
        if not verified.empty:
            verified.select_dtypes(include='number').agg(['mean','std']).to_excel(writer, sheet_name='verified_statistics')
    return frame


def replay_reference_experiment(spec: ReplaySpec, references: tuple[str, ...], *, oracle_search=False,
                                diagnostic_start_s=None, directory_mapping=None):
    """Cross-reference experiment preserves source coordinates/segmentation and never overwrites replay."""
    experiment_identity = identity(dict(source=spec.identity, references=references, oracle_search=oracle_search))
    experiment = replace(spec, references=references, identity=experiment_identity)
    directory = spec.root / 'experiments' / (spec.sample_id + '__' + experiment_identity)
    directory.mkdir(parents=True, exist_ok=True)
    dataset = load_replay_dataset(spec, directory_mapping)
    try:
        result, actual = execute_replay(experiment, dataset, references=references,
            posthoc_mode='search' if oracle_search else 'saved', diagnostic_start_s=diagnostic_start_s)
        metrics = result_metrics(result, spec.fixed_config)
        result.frame.to_csv(directory / 'windows.csv', index=False)
        _write_hr_plot(directory / 'hr.png', result, experiment)
        receipt = dict(identity=experiment_identity, source_identity=spec.identity,
            fixed_config_hash=spec.fixed_config_hash, references=references,
            source_posthoc_bias_s=spec.posthoc_bias_s, experiment_posthoc_bias_s=metrics['posthoc_bias_s'],
            oracle_search=oracle_search, coordinate=spec.coordinate, metrics=metrics,
            actual_diagnostic_start_s=actual)
        (directory / 'receipt.json').write_text(canonical_json(receipt), encoding='utf-8')
        return receipt
    finally:
        clear_all_caches(dataset)
        del dataset
        gc.collect()
