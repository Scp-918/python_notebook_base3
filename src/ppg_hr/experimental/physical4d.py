"""Deterministic physical grid and immutable per-run fixed configuration."""
from __future__ import annotations

from dataclasses import asdict, fields
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import platform
import subprocess
from importlib.metadata import version
from typing import Any

from ppg_hr.params import (DEFAULT_MOTION_DETECTOR, DEFAULT_PREPROCESSING,
    PHYSICAL4D_SETTINGS, ProtocolTrialParams, REFERENCE_IMPLEMENTATION_SHA)

SCHEMA_VERSION = 1


def canonical_json(value: Any) -> str:
    """Canonical serialization rejects nonfinite numerical configuration."""
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def identity(value: Any) -> str:
    return sha256(canonical_json(value).encode('utf-8')).hexdigest()


def coordinates(adaptive_filter: str) -> list[dict[str, Any]]:
    """Generate exactly 300 cells, with penalty width varying fastest."""
    s = PHYSICAL4D_SETTINGS
    steps = getattr(s, adaptive_filter + '_steps', None)
    if steps is None:
        raise ValueError(f'Unsupported filter: {adaptive_filter}')
    step_name = 'klms_step_size' if adaptive_filter == 'klms' else 'LMS_Mu_Base'
    return [dict(coordinate_id=f'p4d_{i:03d}', Fs_Target=fs, memory_ms=memory,
                 step_parameter=step_name, step_value=step, mu=step,
                 penalty_half_width_bpm=width, Spec_Penalty_Width=width / 60,
                 max_order=round(fs * memory / 1000))
            for i, (fs, memory, step, width) in enumerate(product(
                s.sampling_rates, s.memories_ms, steps, s.penalty_widths_bpm))]


def search_fields(adaptive_filter: str) -> set[str]:
    return {'Fs_Target', 'memory_ms', 'Spec_Penalty_Width',
            'klms_step_size' if adaptive_filter == 'klms' else 'LMS_Mu_Base'}


def build_fixed_config(params: ProtocolTrialParams) -> dict[str, Any]:
    """Expand every fixed field, including inactive filters and mechanisms."""
    trial = {k: v for k, v in params.to_dict().items() if k not in search_fields(params.adaptive_filter)}
    settings = {k: v for k, v in asdict(PHYSICAL4D_SETTINGS).items()
                if k not in {'sampling_rates', 'memories_ms', 'penalty_widths_bpm',
                             'lms_steps', 'volterra_steps', 'rff_lms_steps', 'klms_steps'}}
    return json.loads(canonical_json(dict(schema_version=SCHEMA_VERSION, trial=trial,
        preprocessing=asdict(DEFAULT_PREPROCESSING), motion_detector=asdict(DEFAULT_MOTION_DETECTOR),
        evaluation=settings)))


def validate_fixed_config(config: dict[str, Any]) -> None:
    if config.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('Unsupported fixed configuration schema')
    try:
        filter_name = config['trial']['adaptive_filter']
        expected = build_fixed_config(ProtocolTrialParams(adaptive_filter=filter_name))
    except (KeyError, TypeError) as exc:
        raise ValueError('Incomplete fixed configuration') from exc
    def check(actual, template, path):
        if isinstance(template, dict):
            if not isinstance(actual, dict) or set(actual) != set(template):
                raise ValueError(f'Missing or unknown configuration fields: {path}')
            for key in template:
                check(actual[key], template[key], path + '.' + key)
        elif actual is not None and template is not None:
            if isinstance(template, (int, float)) and not isinstance(template, bool):
                if not isinstance(actual, (int, float)) or isinstance(actual, bool):
                    raise ValueError(f'Invalid numerical type: {path}')
            elif not isinstance(actual, type(template)):
                raise ValueError(f'Invalid configuration type: {path}')
    check(config, expected, 'fixed')
    canonical_json(config)


def trial_from_snapshot(config: dict[str, Any], coordinate: dict[str, Any]) -> ProtocolTrialParams:
    """Restore exclusively from the snapshot and coordinate; never fill defaults."""
    validate_fixed_config(config)
    values = dict(config['trial'])
    for key in search_fields(values['adaptive_filter']):
        values[key] = coordinate['step_value'] if key == coordinate['step_parameter'] else coordinate[key]
    if set(values) != {f.name for f in fields(ProtocolTrialParams)}:
        raise ValueError('Incomplete effective trial parameters')
    for key, value in values.items():
        if isinstance(value, list):
            values[key] = tuple(value)
    return ProtocolTrialParams(**values)


def parameter_catalog(config: dict[str, Any]) -> dict[str, Any]:
    """Describe fixed parameters independently of their numerical hash."""
    catalog = {}
    for group, values in config.items():
        if not isinstance(values, dict):
            continue
        for name, value in values.items():
            unit = 's' if name.lower().endswith(('_s', '_seconds')) else ('bpm' if 'bpm' in name.lower() else ('Hz' if 'hz' in name.lower() else 'dimensionless'))
            catalog[group + '.' + name] = dict(name=name, type=type(value).__name__, unit=unit,
                description=f'{group}: {name}', options=None, mechanism=group,
                enabled_when='always; inactive filter/mechanism values remain frozen',
                adopted_value=value, reference_value=None, source='baseline e526b64', difference_reason=None)
    return catalog


def save_snapshot(root: Path, config: dict[str, Any]) -> str:
    """Freeze or verify a run snapshot before loading any sensor data."""
    root = Path(root)
    validate_fixed_config(config)
    digest = identity(config)
    path = root / 'config' / 'fixed_hyperparameters.json'
    if path.exists():
        if identity(load_snapshot(root)) != digest:
            raise ValueError('Existing run fixed configuration differs; choose another output parent')
        return digest
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(config), encoding='utf-8')
    (path.parent / 'parameter_catalog.json').write_text(json.dumps(parameter_catalog(config), ensure_ascii=False, indent=2), encoding='utf-8')
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = 'unavailable'
    provenance = dict(schema_version=SCHEMA_VERSION, fixed_config_hash=digest,
        implementation='reference_parity_physical4d_v1', git_commit=commit,
        reference_commit=REFERENCE_IMPLEMENTATION_SHA, python=platform.python_version(),
        dependencies={name: version(name) for name in ('numpy', 'scipy', 'pandas', 'openpyxl')})
    (path.parent / 'provenance.json').write_text(canonical_json(provenance), encoding='utf-8')
    return digest


def load_snapshot(root: Path) -> dict[str, Any]:
    root = Path(root)
    config = json.loads((root / 'config/fixed_hyperparameters.json').read_text(encoding='utf-8'))
    validate_fixed_config(config)
    provenance = json.loads((root / 'config/provenance.json').read_text(encoding='utf-8'))
    if identity(config) != provenance['fixed_config_hash']:
        raise ValueError('Fixed configuration hash mismatch')
    return config


def run_name(subjects, motions, references, adaptive_filter, scope) -> str:
    """Compact leaf identity contains exactly the five user-selected dimensions."""
    motion_codes = dict(write='w', gripper='g', run='r', rope='p')
    reference_codes = dict(ACC='A', HF='H', CF='C', UD='U')
    filter_codes = dict(lms='L', volterra='V', rff_lms='R', klms='K')
    scope_codes = dict(motion_only='mo', motion_recovery='mr', motion_post10='mp10', global_='gl')
    scope_code = 'gl' if scope == 'global' else scope_codes[scope]
    return '__'.join(('-'.join(sorted(subjects)), ''.join(v for k, v in motion_codes.items() if k in motions),
                     ''.join(reference_codes[g] for g in references), filter_codes[adaptive_filter], scope_code))
