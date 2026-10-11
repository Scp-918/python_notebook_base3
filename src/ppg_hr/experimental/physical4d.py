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
from typing import Any, get_type_hints, get_origin, get_args

from ppg_hr.params import (DEFAULT_MOTION_DETECTOR, DEFAULT_PREPROCESSING,
    PHYSICAL4D_SETTINGS, ProtocolTrialParams, REFERENCE_IMPLEMENTATION_SHA)
from ppg_hr.params import PARAMETER_NOTES, DEFAULT_NUMERICS

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


def build_fixed_config(params: ProtocolTrialParams, *, preprocessing=DEFAULT_PREPROCESSING,
                       motion_detector=DEFAULT_MOTION_DETECTOR, settings=PHYSICAL4D_SETTINGS,
                       numerical=DEFAULT_NUMERICS, references=('ACC',), target_scope='motion_post10') -> dict[str, Any]:
    """Expand every fixed field, including inactive filters and mechanisms."""
    axes = {'sampling_rates', 'memories_ms', 'penalty_widths_bpm',
            'lms_steps', 'volterra_steps', 'rff_lms_steps', 'klms_steps'}
    if any(getattr(settings, key) != getattr(PHYSICAL4D_SETTINGS, key) for key in axes):
        raise ValueError('Physical4D grid axes are fixed; alternative grids are not supported')
    trial = {k: v for k, v in params.to_dict().items() if k not in search_fields(params.adaptive_filter)}
    settings = {k: v for k, v in asdict(settings).items()
                if k not in {'sampling_rates', 'memories_ms', 'penalty_widths_bpm',
                             'lms_steps', 'volterra_steps', 'rff_lms_steps', 'klms_steps'}}
    return json.loads(canonical_json(dict(schema_version=SCHEMA_VERSION, trial=trial,
        execution=dict(reference_groups_order=references, target_scope=target_scope),
        preprocessing=asdict(preprocessing), motion_detector=asdict(motion_detector),
        evaluation=settings, numerical=asdict(numerical), effective_mechanisms={
            **{key: value for key, value in trial.items() if key.startswith('enable_')},
            'advanced_visibility': params.spectral_penalty_policy == 'suppressed_protected_continuous_visibility'})))


def validate_fixed_config(config: dict[str, Any]) -> None:
    if config.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('Unsupported fixed configuration schema')
    try:
        filter_name = config['trial']['adaptive_filter']
        expected = build_fixed_config(ProtocolTrialParams(adaptive_filter=filter_name))
    except (KeyError, TypeError) as exc:
        raise ValueError('Incomplete fixed configuration') from exc
    annotations = {group: get_type_hints(cls) for group, cls in {
        'trial': ProtocolTrialParams, 'preprocessing': type(DEFAULT_PREPROCESSING),
        'motion_detector': type(DEFAULT_MOTION_DETECTOR), 'evaluation': type(PHYSICAL4D_SETTINGS),
        'numerical': type(DEFAULT_NUMERICS)}.items()}
    def compatible(value, annotation):
        origin, args = get_origin(annotation), get_args(annotation)
        if origin is tuple:
            if not isinstance(value, (list, tuple)):
                return False
            if args[-1] is Ellipsis:
                return all(compatible(item, args[0]) for item in value)
            return len(value) == len(args) and all(compatible(item, kind) for item, kind in zip(value, args))
        if args:
            return any(compatible(value, kind) for kind in args)
        if annotation is float:
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if annotation is int:
            return isinstance(value, int) and not isinstance(value, bool)
        return isinstance(value, annotation)
    def check(actual, template, path):
        if isinstance(template, dict):
            if not isinstance(actual, dict) or set(actual) != set(template):
                raise ValueError(f'Missing or unknown configuration fields: {path}')
            for key in template:
                check(actual[key], template[key], path + '.' + key)
        elif len(path.split('.')) == 3 and path.split('.')[1] in annotations:
            _, group, name = path.split('.')
            if not compatible(actual, annotations[group][name]):
                raise ValueError(f'Invalid fixed parameter type: {path}')
        elif actual is None and template is not None:
            raise ValueError(f'Non-null fixed parameter required: {path}')
        elif actual is not None and template is not None:
            if isinstance(template, (int, float)) and not isinstance(template, bool):
                if not isinstance(actual, (int, float)) or isinstance(actual, bool):
                    raise ValueError(f'Invalid numerical type: {path}')
                if isinstance(template, int) and not isinstance(actual, int):
                    raise ValueError(f'Integer fixed parameter required: {path}')
            elif not isinstance(actual, type(template)):
                raise ValueError(f'Invalid configuration type: {path}')
    check(config, expected, 'fixed')
    from ppg_hr.params import ReferenceConfig, TargetScope
    ReferenceConfig(tuple(config['execution']['reference_groups_order']))
    TargetScope(config['execution']['target_scope'])
    effective = {**{k: v for k, v in config['trial'].items() if k.startswith('enable_')},
        'advanced_visibility': config['trial']['spectral_penalty_policy'] == 'suppressed_protected_continuous_visibility'}
    if config['effective_mechanisms'] != effective:
        raise ValueError('Contradictory effective mechanism configuration')
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


def effective_parameter_audit(config: dict[str, Any], coordinate: dict[str, Any]) -> dict[str, Any]:
    """Complete cell audit includes all fixed groups, its coordinate and the merged trial."""
    return dict(fixed_config=config, coordinate=coordinate,
                effective_trial=trial_from_snapshot(config, coordinate).to_dict())


def parameter_catalog(config: dict[str, Any]) -> dict[str, Any]:
    """Describe fixed parameters independently of their numerical hash."""
    catalog = {}
    options = dict(delay_estimation_mode=['direct','envelope'],
        reference_groups_order=['ACC','HF','CF','UD'],
        target_scope=['motion_only','motion_recovery','motion_post10','global'],
        spectral_penalty_policy=['soft_visible','suppressed_protected_continuous_visibility'],
        adaptive_filter=['lms','volterra','rff_lms','klms'], postprocess_method=['fft','ssr'],
        motion_policy=['single_bout','multi_bout'], ppg_input_transform=['raw_bandpass','log_absorbance'],
        normalization_mode=['minmax','zscore','none'], ud_calculation_mode=['raw','smoothed'],
        rff_update_mode=['nlms','lms'], klms_distance_mode=['normalized','absolute_squared'],
        klms_center_prune_policy=['freeze_new_centers','prune_oldest','prune_smallest_weight'])
    reference = dict(M_base=1,C_scale=1,K_max=16,Spec_Penalty_Weight=.4,smooth_win_len=5,
        rff_D=100,rff_seed=42,rff_sigma_scale=1.,klms_sigma=1.,klms_epsilon=.1,M2=3,
        visibility_match_half_width_bpm=3.5,visibility_motion_invocations=3,visibility_edge_guard_bpm=1.)
    differences = dict(rff_sigma_scale='保留robust tap距离自适应带宽；参考采用固定sigma=1。',
        klms_epsilon='数值相同；保留当前归一化距离语义。', M2='二阶长度一致；保留当前归一化线性/二阶更新。')
    for group, values in config.items():
        if not isinstance(values, dict):
            continue
        for name, value in values.items():
            unit = 's' if name.lower().endswith(('_s', '_seconds')) else ('bpm' if 'bpm' in name.lower() else ('Hz' if 'hz' in name.lower() else 'dimensionless'))
            if name == 'fs_origin':
                unit = 'Hz'
            if name == 'memory_ms' or name.endswith('_ms'):
                unit = 'ms'
            if name.endswith(('_windows','_hits','_count')) or name in {'smooth_win_len','Rest_HR_Smooth_Win'}:
                unit = 'windows'
            description = PARAMETER_NOTES.get(name.upper(), f'{group}：{name}；实际采用值见adopted_value。')
            enabled = '固定记录；关闭机制或未选滤波器仍保留参数'
            mechanism = group
            for prefix, condition in [('rff_', 'adaptive_filter == rff_lms'), ('klms_', 'adaptive_filter == klms'),
                ('low_lock_', 'enable_low_lock_recovery'), ('high_lock_', 'enable_high_lock_recovery'),
                ('rise_lineage_', 'enable_rise_lineage'), ('dual_reset_', 'enable_dual_reset'),
                ('post_motion_guard_', 'enable_post_motion_guard'), ('visibility_', '完整spectral_penalty_policy')]:
                if name.startswith(prefix):
                    enabled, mechanism = condition, prefix[:-1]
            if name in {'M2','alpha_u'}:
                enabled, mechanism = 'adaptive_filter == volterra', 'volterra'
            if name.startswith(('motion_tracking_', 'recovery_tracking_', 'final_')):
                description = '对应阶段的HR候选搜索范围或每窗变化阈值/步长；超过limit按step修正。'
            catalog[group + '.' + name] = dict(name=name, type=type(value).__name__, unit=unit,
                description=description, options=([False,True] if isinstance(value,bool) else options.get(name)), mechanism=mechanism,
                enabled_when=enabled, adopted_value=value, reference_value=reference.get(name),
                source=(f'reference {REFERENCE_IMPLEMENTATION_SHA}' if name in reference else 'baseline e526b64'),
                difference_reason=differences.get(name))
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
    from .physical4d_run import implementation_identity
    provenance['implementation_identity'] = implementation_identity()
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
