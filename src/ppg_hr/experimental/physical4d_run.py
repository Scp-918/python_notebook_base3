"""Transactional Physical4D ledger and serial, sample-first training entry point."""
from __future__ import annotations

from dataclasses import replace
import gc
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import sqlite3
import traceback
from typing import Any, Callable
import numpy as np

from ppg_hr.params import ReferenceConfig, TargetScope, PreprocessingParams, reference_parity_params
from .physical4d import (build_fixed_config, canonical_json, coordinates, identity,
                         load_snapshot, run_name, save_snapshot, trial_from_snapshot)
from .sample_context import SampleContext, prepare_sample_context, save_sample_diagnostics
from .cascade_solver import run_protocol_trial, clear_all_caches
from .batch_pairing import discover_sample_pairs_with_unpaired
from .preprocess_protocol import load_and_preprocess_protocol


def json_safe(value: Any) -> Any:
    """Replace unavailable metrics by JSON null, never a synthetic error value."""
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def implementation_identity() -> str:
    """Hash numerical source contents rather than relying on mutable Git labels."""
    root = Path(__file__).resolve().parents[1]
    digest = sha256()
    for path in sorted(root.rglob('*.py')):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes().replace(b'\r\n', b'\n'))
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    digest = sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return dict(path=str(Path(path).resolve()), sha256=digest.hexdigest(), bytes=Path(path).stat().st_size)


class RunLedger:
    """SQLite is the runtime authority; each successful cell is an atomic receipt."""

    def __init__(self, root: Path, *, readonly: bool = False):
        self.root = Path(root)
        self.config = load_snapshot(self.root)
        self.manifest = json.loads((self.root / 'manifest.json').read_text(encoding='utf-8'))
        if identity(self.config) != self.manifest['fixed_config_hash']:
            raise ValueError('Manifest fixed configuration hash differs')
        if self.manifest['implementation_identity'] != implementation_identity():
            raise ValueError('Implementation identity differs from saved run')
        expected = dict(self.manifest)
        expected.pop('run_identity')
        if identity(expected) != self.manifest['run_identity']:
            raise ValueError('Run manifest identity mismatch')
        if readonly:
            self.connection = sqlite3.connect((self.root / 'run.sqlite').resolve().as_uri() + '?mode=ro', uri=True)
        else:
            self.connection = sqlite3.connect(self.root / 'run.sqlite')
            self.connection.execute('PRAGMA journal_mode=WAL')
        self.connection.row_factory = sqlite3.Row
        self.connection.execute('PRAGMA foreign_keys=ON')

    @classmethod
    def create(cls, root: Path, config: dict, references: tuple[str, ...], scope: str,
               subjects: list[str], motions: list[str]) -> RunLedger:
        ReferenceConfig(references)
        TargetScope(scope)
        digest = save_snapshot(root, config)
        grid = coordinates(config['trial']['adaptive_filter'])
        manifest = dict(schema_version=1, fixed_config_hash=digest,
            fixed_config_path='config/fixed_hyperparameters.json',
            implementation_identity=implementation_identity(), reference_groups_order=references,
            target_scope=scope, subjects=sorted(subjects), motions=sorted(motions),
            coordinates=grid, grid_hash=identity(grid), best_policy='finite_posthoc_final_aae_coordinate_id')
        manifest = json.loads(canonical_json(manifest))
        manifest['run_identity'] = identity(manifest)
        path = Path(root) / 'manifest.json'
        if path.exists():
            if json.loads(path.read_text(encoding='utf-8')) != manifest:
                raise ValueError('Existing run identity differs; choose another output parent')
        else:
            path.write_text(canonical_json(manifest), encoding='utf-8')
        ledger = cls(root)
        ledger.connection.executescript('''
            CREATE TABLE IF NOT EXISTS samples (
                sample_id TEXT PRIMARY KEY, subject TEXT NOT NULL, motion TEXT NOT NULL,
                input_json TEXT NOT NULL, context_json TEXT NOT NULL,
                sample_complete INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS coordinates (
                coordinate_id TEXT PRIMARY KEY, coordinate_json TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS cell_metrics (
                sample_id TEXT NOT NULL REFERENCES samples(sample_id),
                coordinate_id TEXT NOT NULL REFERENCES coordinates(coordinate_id),
                status TEXT NOT NULL, tdelay_aae_bpm REAL, tdelay_accuracy_pct REAL,
                posthoc_aae_bpm REAL, posthoc_accuracy_pct REAL, posthoc_bias_s REAL,
                n_valid INTEGER NOT NULL, posthoc_n_valid INTEGER NOT NULL,
                fixed_config_hash TEXT NOT NULL, cell_identity TEXT NOT NULL,
                parameters_json TEXT NOT NULL, metrics_json TEXT NOT NULL,
                PRIMARY KEY(sample_id, coordinate_id));
            CREATE TABLE IF NOT EXISTS attempt_events (
                event_id INTEGER PRIMARY KEY, sample_id TEXT NOT NULL,
                coordinate_id TEXT, error_type TEXT NOT NULL, error TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
        ''')
        with ledger.connection:
            for cell in grid:
                ledger.connection.execute('INSERT OR IGNORE INTO coordinates VALUES (?,?)',
                                          (cell['coordinate_id'], canonical_json(cell)))
        saved = [json.loads(row[0]) for row in ledger.connection.execute(
            'SELECT coordinate_json FROM coordinates ORDER BY coordinate_id')]
        if saved != grid:
            ledger.close()
            raise ValueError('SQLite grid differs from manifest')
        return ledger

    def close(self) -> None:
        self.connection.close()

    def add_sample(self, sample_id, subject, motion, input_identity, context) -> None:
        input_json = canonical_json(json_safe(input_identity))
        row = self.connection.execute('SELECT input_json FROM samples WHERE sample_id=?', (sample_id,)).fetchone()
        if row:
            if row[0] != input_json:
                raise ValueError('Sample input identity differs from saved run')
            return
        with self.connection:
            self.connection.execute('INSERT INTO samples VALUES (?,?,?,?,?,0)',
                (sample_id, subject, motion, input_json, canonical_json(json_safe(context))))

    def sample_complete(self, sample_id: str) -> bool:
        row = self.connection.execute('SELECT sample_complete FROM samples WHERE sample_id=?', (sample_id,)).fetchone()
        return bool(row and row[0])

    def cells(self, sample_id: str) -> list[dict[str, Any]]:
        return [dict(row) for row in self.connection.execute(
            'SELECT * FROM cell_metrics WHERE sample_id=? ORDER BY coordinate_id', (sample_id,))]

    def missing_coordinates(self, sample_id):
        return [json.loads(row[0]) for row in self.connection.execute('''
            SELECT coordinate_json FROM coordinates c WHERE NOT EXISTS
            (SELECT 1 FROM cell_metrics m WHERE m.sample_id=? AND m.coordinate_id=c.coordinate_id)
            ORDER BY coordinate_id''', (sample_id,))]

    def record_error(self, sample_id, coordinate_id, exc) -> None:
        with self.connection:
            self.connection.execute('INSERT INTO attempt_events(sample_id,coordinate_id,error_type,error) VALUES (?,?,?,?)',
                (sample_id, coordinate_id, type(exc).__name__, ''.join(traceback.format_exception_only(type(exc), exc)).strip()))

    def record_cell(self, sample_id, coordinate, metrics) -> None:
        metrics = json_safe(metrics)
        effective = trial_from_snapshot(self.config, coordinate).to_dict()
        sample = self.connection.execute('SELECT input_json,context_json FROM samples WHERE sample_id=?', (sample_id,)).fetchone()
        cell_id = identity(dict(run=self.manifest['run_identity'], sample=sample_id,
            input=json.loads(sample[0]), context=json.loads(sample[1]), coordinate=coordinate))
        scored = metrics.get('posthoc_aae_bpm') is not None
        args = (sample_id, coordinate['coordinate_id'], 'completed' if scored else 'unscored',
            metrics.get('tdelay_aae_bpm'), metrics.get('tdelay_accuracy_pct'), metrics.get('posthoc_aae_bpm'),
            metrics.get('posthoc_accuracy_pct'), metrics.get('posthoc_bias_s'), metrics.get('n_valid', 0),
            metrics.get('posthoc_n_valid', 0), self.manifest['fixed_config_hash'], cell_id,
            canonical_json(effective), canonical_json(metrics))
        with self.connection:
            self.connection.execute('INSERT INTO cell_metrics VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)', args)
            count = self.connection.execute('SELECT COUNT(*) FROM cell_metrics WHERE sample_id=?', (sample_id,)).fetchone()[0]
            if count == 300:
                self.connection.execute('UPDATE samples SET sample_complete=1 WHERE sample_id=?', (sample_id,))


def execute_sample_cells(ledger: RunLedger, sample_id: str, runner: Callable,
                         *, after_cell: Callable | None = None) -> None:
    """Try each missing coordinate once; error cells remain missing for the next invocation."""
    if ledger.sample_complete(sample_id):
        return
    for coordinate in ledger.missing_coordinates(sample_id):
        metrics = None
        try:
            metrics = runner(coordinate)
            ledger.record_cell(sample_id, coordinate, metrics)
        except Exception as exc:
            ledger.record_error(sample_id, coordinate['coordinate_id'], exc)
        finally:
            del metrics
            if after_cell:
                after_cell(coordinate)
    gc.collect()


def result_metrics(result) -> dict[str, Any]:
    if not result.success:
        raise RuntimeError(result.reason)
    arrays = result.metric_arrays
    mask = arrays['filtered_mask']
    valid = mask & np.isfinite(arrays['final_abs_err_bpm'])
    return json_safe(dict(tdelay_aae_bpm=result.final_aae_bpm, tdelay_accuracy_pct=result.final_acc_pct,
        posthoc_aae_bpm=result.posthoc_final_aae_bpm, posthoc_accuracy_pct=result.posthoc_final_acc_pct,
        n_valid=int(valid.sum()), posthoc_n_valid=result.posthoc_n_valid_windows,
        posthoc_bias_s=result.time_bias_after.time_bias_after_s if result.time_bias_after else None))


def _multiple(value):
    return [value] if isinstance(value, (str, Path)) else list(value)


def run_physical4d(subjects, motions=('write', 'gripper', 'run', 'rope'), *,
                   references=(('ACC',),), adaptive_filters=('lms',), scopes=('motion_post10',),
                   fixed_params=None, output_parent=None, save_diagnostics=True) -> list[Path]:
    """Expand outer modes; prepare one sample at a time and enumerate every 300-cell grid."""
    subjects = sorted((Path(p).resolve() for p in _multiple(subjects)), key=lambda p: p.name)
    if not subjects or len({p.name for p in subjects}) != len(subjects):
        raise ValueError('Select distinct nonempty subject directories')
    motions = _multiple(motions)
    if not set(motions) <= {'write', 'gripper', 'run', 'rope'} or not motions:
        raise ValueError('Invalid motion selection')
    if isinstance(references, tuple) and references and isinstance(references[0], str):
        references = [references]
    modes = list(product(references, _multiple(adaptive_filters), _multiple(scopes)))
    parents = {p.parent for p in subjects}
    if output_parent is None:
        if len(parents) != 1:
            raise ValueError('Different data parents require explicit OUTPUT_PARENT')
        output_parent = next(iter(parents)) / 'outputs'
    ledgers = []
    try:
        for refs, adaptive_filter, scope in modes:
            params = fixed_params or reference_parity_params()
            params = replace(params, adaptive_filter=adaptive_filter)
            if params.cascade_mode != 'reference_parity':
                raise ValueError('Physical4D requires reference_parity configuration')
            root = Path(output_parent) / run_name([p.name for p in subjects], motions, refs, adaptive_filter, scope)
            ledger = RunLedger.create(root, build_fixed_config(params), tuple(refs), scope,
                                      [p.name for p in subjects], motions)
            ledgers.append(ledger)
        for subject_dir in subjects:
            discovery = discover_sample_pairs_with_unpaired(subject_dir)
            for ledger in ledgers:
                for issue in discovery.unpaired:
                    ledger.record_error(issue.file_name, None, ValueError(issue.reason))
            for pair in discovery.pairs:
                if pair.motion_type not in motions:
                    continue
                input_id = {key: file_identity(path) for key, path in dict(sensor=pair.sensor_csv,
                    reference=pair.ref_csv, calibration=subject_dir.parent / 'ck.mat').items()}
                needed = [ledger for ledger in ledgers if not ledger.sample_complete(pair.sample_id)]
                for ledger in ledgers:
                    saved = ledger.connection.execute('SELECT input_json FROM samples WHERE sample_id=?', (pair.sample_id,)).fetchone()
                    if saved and json.loads(saved[0]) != input_id:
                        raise ValueError(f'Input changed: {pair.sample_id}')
                if not needed:
                    continue
                dataset = None
                try:
                    dataset = load_and_preprocess_protocol(pair.sensor_csv, pair.ref_csv,
                        fs_origin=ledgers[0].config['preprocessing']['fs_origin'],
                        preprocessing=PreprocessingParams(**ledgers[0].config['preprocessing']))
                    contexts = {}
                    for ledger in needed:
                        params = trial_from_snapshot(ledger.config, ledger.manifest['coordinates'][0])
                        saved = ledger.connection.execute('SELECT context_json FROM samples WHERE sample_id=?', (pair.sample_id,)).fetchone()
                        if saved:
                            context = SampleContext.from_dict(json.loads(saved[0]))
                        elif contexts:
                            context = next(iter(contexts.values()))
                        else:
                            context = prepare_sample_context(dataset, params, input_id, ledger.config)
                        contexts[context.preparation_identity] = context
                        ledger.add_sample(pair.sample_id, pair.subject, pair.motion_type, input_id, context.to_dict())
                        if save_diagnostics:
                            save_sample_diagnostics(ledger.root, pair.subject, dataset, context,
                                                    tuple(ledger.manifest['reference_groups_order']))
                        previous_fs = None
                        def run_cell(coordinate):
                            nonlocal previous_fs
                            if previous_fs != coordinate['Fs_Target']:
                                clear_all_caches(dataset)
                                previous_fs = coordinate['Fs_Target']
                            trial = trial_from_snapshot(ledger.config, coordinate)
                            result = run_protocol_trial(dataset, tuple(ledger.manifest['reference_groups_order']),
                                ledger.manifest['target_scope'], trial, collect_frame=False, collect_stages=False,
                                sample_context=context, posthoc_mode='search')
                            try:
                                return result_metrics(result)
                            finally:
                                del result
                        execute_sample_cells(ledger, pair.sample_id, run_cell)
                        clear_all_caches(dataset)
                        from .physical4d_export import export_sample_grid, rebuild_summaries
                        export_sample_grid(ledger, pair.sample_id)
                        rebuild_summaries(ledger)
                except Exception as exc:
                    for ledger in needed:
                        ledger.record_error(pair.sample_id, None, exc)
                finally:
                    if dataset is not None:
                        clear_all_caches(dataset)
                    dataset = None
                    contexts = None
                    gc.collect()
        from .physical4d_export import rebuild_summaries
        for ledger in ledgers:
            rebuild_summaries(ledger)
        return [ledger.root for ledger in ledgers]
    finally:
        for ledger in ledgers:
            ledger.close()

