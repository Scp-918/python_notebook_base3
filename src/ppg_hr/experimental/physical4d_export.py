"""Rebuild readable grids and sample/motion summaries exclusively from SQLite."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from .physical4d_run import RunLedger


def sample_grid(ledger: RunLedger, sample_id: str) -> pd.DataFrame:
    """Always return all 300 coordinates, including technical failures and missing cells."""
    sample = ledger.connection.execute('SELECT * FROM samples WHERE sample_id=?', (sample_id,)).fetchone()
    if sample is None:
        raise KeyError(sample_id)
    context = json.loads(sample['context_json'])
    cells = {row['coordinate_id']: row for row in ledger.cells(sample_id)}
    errors = {row['coordinate_id']: row['error'] for row in ledger.connection.execute(
        'SELECT coordinate_id,error FROM attempt_events WHERE sample_id=? ORDER BY event_id', (sample_id,))}
    records = []
    for coordinate in ledger.manifest['coordinates']:
        cell = cells.get(coordinate['coordinate_id'])
        records.append(dict(sample_id=sample_id, subject=sample['subject'], motion=sample['motion'],
            **coordinate, penalty_half_width_hz=coordinate['Spec_Penalty_Width'],
            Tdelay_s=context['Tdelay_s'], status=cell['status'] if cell else ('failed' if coordinate['coordinate_id'] in errors else 'missing'),
            error=errors.get(coordinate['coordinate_id']) if not cell else None,
            fixed_config_hash=ledger.manifest['fixed_config_hash'], run_identity=ledger.manifest['run_identity'],
            **({k: v for k, v in cell.items() if k not in {'sample_id','coordinate_id','status','fixed_config_hash'}} if cell else {})))
    return pd.DataFrame(records)


def export_sample_grid(ledger: RunLedger, sample_id: str) -> Path:
    grid = sample_grid(ledger, sample_id)
    target = ledger.root / grid.subject.iloc[0] / 'grid' / (sample_id + '__p4d.xlsx')
    target.parent.mkdir(parents=True, exist_ok=True)
    grid.to_excel(target, index=False)
    return target


def best_cell(ledger: RunLedger, sample_id: str, *, require_complete: bool = True):
    if require_complete and not ledger.sample_complete(sample_id):
        return None
    row = ledger.connection.execute('''SELECT * FROM cell_metrics WHERE sample_id=?
        AND posthoc_aae_bpm IS NOT NULL ORDER BY posthoc_aae_bpm,coordinate_id LIMIT 1''', (sample_id,)).fetchone()
    return dict(row) if row else None


def rebuild_summaries(ledger_or_root) -> pd.DataFrame:
    """Write per-subject summaries and the cross-subject overall table; no solver calls."""
    own = not isinstance(ledger_or_root, RunLedger)
    ledger = RunLedger(ledger_or_root, readonly=True) if own else ledger_or_root
    try:
        samples = [dict(row) for row in ledger.connection.execute('SELECT * FROM samples ORDER BY subject,motion,sample_id')]
        best_rows, summary_rows = [], []
        for sample in samples:
            best = best_cell(ledger, sample['sample_id'])
            row = dict(sample_id=sample['sample_id'], subject=sample['subject'], motion=sample['motion'],
                sample_complete=bool(sample['sample_complete']), best_status='scored' if best else ('unscored' if sample['sample_complete'] else 'incomplete'),
                fixed_config_hash=ledger.manifest['fixed_config_hash'])
            if best:
                coordinate = next(c for c in ledger.manifest['coordinates'] if c['coordinate_id'] == best['coordinate_id'])
                row.update(coordinate)
                row.update(best)
                row['Tdelay_s'] = json.loads(sample['context_json'])['Tdelay_s']
            best_rows.append(row)
        frame = pd.DataFrame(best_rows)
        if frame.empty:
            frame = pd.DataFrame(columns=['sample_id','subject','motion','sample_complete','best_status','fixed_config_hash','posthoc_aae_bpm','posthoc_accuracy_pct'])
        for name in ('posthoc_aae_bpm', 'posthoc_accuracy_pct'):
            if name not in frame:
                frame[name] = float('nan')
        for subject in ledger.manifest['subjects']:
            selected = frame[frame.subject == subject]
            target = ledger.root / subject / 'summary'
            target.mkdir(parents=True, exist_ok=True)
            selected.to_excel(target / 'sample_best.xlsx', index=False)
            rows = []
            for motion in ledger.manifest['motions']:
                group = selected[selected.motion == motion]
                valid = group[group.best_status == 'scored']
                failed = ledger.connection.execute('SELECT COUNT(DISTINCT sample_id) FROM attempt_events WHERE sample_id IN (SELECT sample_id FROM samples WHERE subject=? AND motion=? AND sample_complete=0)', (subject, motion)).fetchone()[0]
                rows.append(dict(subject=subject, motion=motion, sample_count=len(group),
                    complete_count=int(group.sample_complete.sum()), failed_count=failed, scored_count=len(valid),
                    mean_aae_bpm=valid.posthoc_aae_bpm.mean(), std_aae_bpm=valid.posthoc_aae_bpm.std(ddof=1),
                    mean_accuracy_pct=valid.posthoc_accuracy_pct.mean(), std_accuracy_pct=valid.posthoc_accuracy_pct.std(ddof=1),
                    fixed_config_hash=ledger.manifest['fixed_config_hash']))
            pd.DataFrame(rows).to_excel(target / 'motion_summary.xlsx', index=False)
            summary_rows.extend(rows)
        with pd.ExcelWriter(ledger.root / 'overall_summary.xlsx') as writer:
            frame.to_excel(writer, sheet_name='sample_best', index=False)
            pd.DataFrame(summary_rows).to_excel(writer, sheet_name='subject_motion', index=False)
            if not frame.empty:
                frame.groupby('motion').agg(sample_count=('sample_id','size'), scored_count=('posthoc_aae_bpm','count'),
                    mean_aae_bpm=('posthoc_aae_bpm','mean'), std_aae_bpm=('posthoc_aae_bpm','std'),
                    mean_accuracy_pct=('posthoc_accuracy_pct','mean'), std_accuracy_pct=('posthoc_accuracy_pct','std')).reset_index().to_excel(writer, sheet_name='cross_subject', index=False)
        return frame
    finally:
        if own:
            ledger.close()
