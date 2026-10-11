"""Transactional Physical4D ledger and serial, sample-first training entry point."""
from __future__ import annotations

from dataclasses import replace
import ast
import gc
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import sqlite3
import traceback
from typing import Any, Callable
import numpy as np

from ppg_hr.params import (ReferenceConfig, TargetScope, PreprocessingParams, reference_parity_params,
                          DEFAULT_PREPROCESSING, DEFAULT_MOTION_DETECTOR, PHYSICAL4D_SETTINGS, DEFAULT_NUMERICS)
from .physical4d import (build_fixed_config, canonical_json, coordinates, identity,
                         load_snapshot, run_name, save_snapshot, trial_from_snapshot, effective_parameter_audit)
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
    """Hash algorithm/schema AST; saved defaults and catalog prose are not implementation changes."""
    class NumericalSource(ast.NodeTransformer):
        def __init__(self, parameters):
            self.parameters = parameters

        def visit(self, node):
            if hasattr(node, 'body') and isinstance(node.body, list) and node.body:
                first = node.body[0]
                if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                    node.body = node.body[1:]
            return super().visit(node)

        def visit_ClassDef(self, node):
            dataclass = any(isinstance(d, ast.Name) and d.id == 'dataclass' or
                isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id == 'dataclass'
                for d in node.decorator_list)
            if self.parameters and dataclass:
                for field in node.body:
                    if isinstance(field, ast.AnnAssign):
                        field.value = None  # Saved configuration supplies all field values.
            return self.generic_visit(node)

        def visit_Assign(self, node):
            if self.parameters and any(isinstance(t, ast.Name) and t.id in {
                'PARAMETER_NOTES', '_CORE_V1_VALUES', 'CORE_FEATURE_FLAGS'} for t in node.targets):
                return None
            return self.generic_visit(node)

        def visit_FunctionDef(self, node):
            if self.parameters and node.name == 'reference_parity_params':
                node.body = [ast.Pass()]  # Request defaults never override a replay snapshot.
            return self.generic_visit(node)

    root = Path(__file__).resolve().parents[1]
    digest = sha256()
    for path in sorted(root.rglob('*.py')):
        digest.update(path.relative_to(root).as_posix().encode())
        tree = ast.parse(path.read_bytes().decode('utf-8-sig'))
        tree = NumericalSource(path.name == 'params.py').visit(tree)
        digest.update(ast.dump(tree, include_attributes=False).encode('utf-8'))
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
        if self.config['execution'] != dict(reference_groups_order=self.manifest['reference_groups_order'],
                                            target_scope=self.manifest['target_scope']):
            raise ValueError('Execution configuration differs from manifest')
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
        config = dict(config, execution=dict(reference_groups_order=list(references), target_scope=scope))
        digest = save_snapshot(root, config)
        grid = coordinates(config['trial']['adaptive_filter'])
        manifest = dict(schema_version=1, fixed_config_hash=digest,
            fixed_config_path='config/fixed_hyperparameters.json',
            implementation_identity=implementation_identity(), reference_groups_order=references,
            target_scope=scope, subjects=sorted(subjects), motions=sorted(motions),
            spectral_penalty_policy=config['trial']['spectral_penalty_policy'],
            effective_mechanisms=config['effective_mechanisms'],
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
        effective = effective_parameter_audit(self.config, coordinate)
        sample = self.connection.execute('SELECT input_json,context_json FROM samples WHERE sample_id=?', (sample_id,)).fetchone()
        cell_id = identity(dict(run=self.manifest['run_identity'], sample=sample_id,
            input=json.loads(sample[0]), context=json.loads(sample[1]), coordinate=coordinate))
        scored = metrics.get('tdelay_aae_bpm') is not None
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


def result_metrics(result, config=None) -> dict[str, Any]:
    if not result.success:
        raise RuntimeError(result.reason)
    arrays = result.metric_arrays
    mask = arrays['filtered_mask']
    valid = mask & np.isfinite(arrays['final_abs_err_bpm'])
    from ppg_hr.params import PHYSICAL4D_SETTINGS
    threshold = config['evaluation']['accuracy_threshold_bpm'] if config else PHYSICAL4D_SETTINGS.accuracy_threshold_bpm
    after_errors = arrays.get('final_abs_err_after_bpm', np.full(len(mask), np.nan))[mask]
    after_errors = after_errors[np.isfinite(after_errors)]
    return json_safe(dict(tdelay_aae_bpm=result.final_aae_bpm,
        tdelay_accuracy_pct=float(np.mean(arrays['final_abs_err_bpm'][valid] <= threshold) * 100) if valid.any() else None,
        posthoc_aae_bpm=result.posthoc_final_aae_bpm,
        posthoc_accuracy_pct=float(np.mean(after_errors <= threshold) * 100) if after_errors.size else None,
        n_valid=int(valid.sum()), posthoc_n_valid=result.posthoc_n_valid_windows,
        posthoc_bias_s=result.time_bias_after.time_bias_after_s if result.time_bias_after and result.time_bias_after.status == 'ok' else None))


def _multiple(value):
    return [value] if isinstance(value, (str, Path)) else list(value)


def run_physical4d(subjects, motions=('write', 'gripper', 'run', 'rope'), *,
                   references=(('ACC',),), adaptive_filters=('lms',), scopes=('motion_post10',),
                   fixed_params=None, output_parent=None, save_diagnostics=True,
                   preprocessing=DEFAULT_PREPROCESSING, motion_detector=DEFAULT_MOTION_DETECTOR,
                   settings=PHYSICAL4D_SETTINGS, numerical=DEFAULT_NUMERICS) -> list[Path]:
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
    if not modes or len({(tuple(refs), filt, scope) for refs, filt, scope in modes}) != len(modes):
        raise ValueError('Select distinct nonempty outer combinations')
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
            config = build_fixed_config(params, preprocessing=preprocessing, motion_detector=motion_detector,
                                        settings=settings, numerical=numerical, references=tuple(refs), target_scope=scope)
            ledger = RunLedger.create(root, config, tuple(refs), scope,
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
                    dataset.sample_stem = pair.sample_id
                    contexts = {}
                    for ledger in needed:
                        params = trial_from_snapshot(ledger.config, ledger.manifest['coordinates'][0])
                        saved = ledger.connection.execute('SELECT context_json FROM samples WHERE sample_id=?', (pair.sample_id,)).fetchone()
                        if saved:
                            context = SampleContext.from_dict(json.loads(saved[0]))
                        elif contexts:
                            context = next(iter(contexts.values()))
                        else:
                            context = prepare_sample_context(dataset, params, input_id, fixed_config=ledger.config)
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
                                sample_context=context, posthoc_mode='search', numerical_config=ledger.config['numerical'])
                            try:
                                if not result.success:
                                    raise RuntimeError(result.reason)
                                return result_metrics(result, ledger.config)
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
            for sample in ledger.connection.execute('SELECT sample_id FROM samples ORDER BY sample_id'):
                from .physical4d_export import export_sample_grid
                export_sample_grid(ledger, sample[0])
            rebuild_summaries(ledger)
        return [ledger.root for ledger in ledgers]
    finally:
        for ledger in ledgers:
            ledger.close()


def repair_run_outputs(root: Path, *, diagnostics: bool = True) -> None:
    """Recreate absent plots and complete grids without detector, alignment or solver calls."""
    from .physical4d_export import export_sample_grid, rebuild_summaries
    ledger = RunLedger(root, readonly=True)
    try:
        for sample in ledger.connection.execute('SELECT * FROM samples ORDER BY sample_id'):
            export_sample_grid(ledger, sample['sample_id'])
            subject = sample['subject']
            missing = any(not (ledger.root / subject / group / (sample['sample_id'] + suffix)).is_file()
                for group, suffix in [('align','__tdelay.png'), ('signal','__fft_signal.png')])
            if diagnostics and missing:
                inputs = json.loads(sample['input_json'])
                for expected in inputs.values():
                    actual = file_identity(Path(expected['path']))
                    if actual['sha256'] != expected['sha256']:
                        raise ValueError('Input changed while regenerating diagnostics')
                prep = PreprocessingParams(**ledger.config['preprocessing'])
                dataset = load_and_preprocess_protocol(inputs['sensor']['path'], inputs['reference']['path'],
                    fs_origin=prep.fs_origin, preprocessing=prep,
                    ud_calculation_mode=prep.ud_calculation_mode, ud_smoothing_window_s=prep.ud_smoothing_window_s)
                dataset.sample_stem = sample['sample_id']
                try:
                    context = SampleContext.from_dict(json.loads(sample['context_json']))
                    save_sample_diagnostics(ledger.root, subject, dataset, context,
                                            tuple(ledger.manifest['reference_groups_order']))
                finally:
                    del dataset
                    gc.collect()
        rebuild_summaries(ledger)
    finally:
        ledger.close()

