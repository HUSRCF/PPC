from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from paper_plot_style import COLORS, save_fig

ROOT = Path('.')
FIG = Path('figures')
METRICS = [
    ('f1_best', 'Val F1'),
    ('pr_auc', 'Val PR-AUC'),
    ('auroc', 'Val AUROC'),
    ('mcc_best', 'Val MCC'),
    ('top5', 'Val Top-5% P'),
    ('top10', 'Val Top-10% P'),
]
CURATED_MANIFESTS = [
    Path('runs/job_manifests/hpcue_predstruct_batch_20260619_002109.tsv'),
    Path('runs/job_manifests/hpcue_predstruct_moreseeds_20260619_002328.tsv'),
]
STAR_METRICS = Path('figures/raw/star_final_predstruct_context_nograph_20260618_171453/metrics.jsonl')
STAR_CONFIG = Path('configs/experiments/20260618_star/predicted_structure_sequence/no_contact_context/train_contact_site_esm_final_predstruct_context_nograph_seed42_star_20260618_171453.yaml')


def read_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_run_config(run_dir: Path) -> dict:
    cfg_path = run_dir / 'config.json'
    if not cfg_path.exists():
        return {}
    cfg = json.loads(cfg_path.read_text())
    return cfg


def load_yaml_config(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def split_cfg(cfg: dict) -> tuple[dict, dict, dict]:
    y = cfg.get('yaml_config') if isinstance(cfg.get('yaml_config'), dict) else cfg
    data = y.get('data', {}) if isinstance(y, dict) else {}
    model = cfg.get('model_config') or (y.get('model', {}) if isinstance(y, dict) else {})
    training = y.get('training', {}) if isinstance(y, dict) else {}
    return data or {}, model or {}, training or {}


def best_from_metrics(metrics_path: Path) -> tuple[dict | None, dict | None, bool]:
    best = None
    last = None
    done = False
    for obj in read_jsonl(metrics_path):
        if obj.get('event') == 'epoch':
            val = obj.get('val', {})
            rec = {
                'best_epoch': obj.get('epoch'),
                'f1_best': val.get('f1_best_threshold'),
                'f1_at_0_5': val.get('f1_at_0_5'),
                'best_threshold': val.get('best_threshold'),
                'pr_auc': val.get('pr_auc'),
                'auroc': val.get('auroc'),
                'mcc_best': val.get('mcc_best_threshold'),
                'mcc_at_0_5': val.get('mcc_at_0_5'),
                'top5': val.get('top_5pct_precision_micro'),
                'top10': val.get('top_10pct_precision_micro'),
                'val_loss': val.get('loss'),
            }
            last = rec
            if rec['f1_best'] is not None and (best is None or rec['f1_best'] > best['f1_best']):
                best = rec
        elif obj.get('event') == 'done':
            done = True
    return best, last, done


def feature_kind(data: dict, model: dict) -> str:
    root = str(data.get('sequence_feature_root') or '')
    if not model.get('use_seq_features', True):
        return 'No scalar'
    if 'pred_struct_sequence_scalar' in root:
        return 'PredStruct scalar'
    if root and root.lower() not in {'none', 'null'}:
        return 'Seq scalar'
    return 'AA scalar'


def esm_kind(model: dict, data: dict, run_name: str) -> str:
    d_esm = int(model.get('d_esm') or 0)
    root = str(data.get('esm_root') or '')
    if d_esm >= 3000 or 'mlc' in root or '_mlc_' in run_name:
        return 'MLC ESM'
    return 'Final ESM'


def context_kind(model: dict) -> str:
    ctx = any(bool(model.get(k)) for k in ('use_chain_embedding', 'use_position_features', 'use_global_context'))
    contact = bool(model.get('use_contact_graph'))
    root = str(model.get('_esm_root_for_contact_check') or '')
    has_contact_payload = ('_contact' in root) or ('_mlc' in root)
    if contact and not has_contact_payload:
        return '+context empty-graph' if ctx else 'no-context empty-graph'
    if ctx and contact:
        return '+context +contact'
    if ctx and not contact:
        return '+context no-contact'
    if (not ctx) and contact:
        return 'no-context +contact'
    return 'no-context no-contact'


def label_for(data: dict, model: dict, training: dict, run_name: str, source: str) -> str:
    model = dict(model)
    model['_esm_root_for_contact_check'] = str(data.get('esm_root') or '')
    esm = esm_kind(model, data, run_name)
    feat = feature_kind(data, model)
    ctx = context_kind(model)
    reg = ''
    if 'nostrongreg' in run_name or float(training.get('weight_decay') or 0.05) < 0.05 or float(training.get('label_smoothing') or 0.03) == 0.0:
        reg = ' weak-reg'
    return f'{esm} | {feat} | {ctx}{reg}'


def short_label(label: str) -> str:
    repl = {
        'MLC ESM | PredStruct scalar | +context +contact': 'MLC + PS + contact',
        'MLC ESM | PredStruct scalar | +context no-contact': 'MLC + PS + no graph',
        'MLC ESM | PredStruct scalar | +context +contact weak-reg': 'MLC + PS + contact weak-reg',
        'MLC ESM | No scalar | +context +contact': 'MLC + no scalar + contact',
        'Final ESM | PredStruct scalar | +context +contact': 'Final + PS + contact',
        'Final ESM | No scalar | +context +contact': 'Final + no scalar + contact',
        'Final ESM | PredStruct scalar | +context empty-graph': 'Final + PS empty graph',
        'Final ESM | No scalar | +context empty-graph': 'Final + no scalar empty graph',
        'Final ESM | PredStruct scalar | no-context no-contact': 'Final + PS no-context',
        'Final ESM | No scalar | no-context no-contact': 'Final + no scalar no-context',
        'Final ESM | PredStruct scalar | +context no-contact': 'Final + PS + context no graph',
    }
    return repl.get(label, label)


def row_from_run(run_dir: Path, source: str, config_override: dict | None = None) -> dict | None:
    metrics = run_dir / 'metrics.jsonl'
    if not metrics.exists():
        return None
    best, last, done = best_from_metrics(metrics)
    if best is None:
        return None
    cfg = config_override or load_run_config(run_dir)
    data, model, training = split_cfg(cfg)
    run_name = run_dir.name
    label = label_for(data, model, training, run_name, source)
    row = {
        'source': source,
        'run_name': run_name,
        'run_path': str(run_dir),
        'model_label': label,
        'model_short': short_label(label),
        'seed': training.get('seed'),
        'split_dir': data.get('split_dir'),
        'esm_root': data.get('esm_root'),
        'sequence_feature_root': data.get('sequence_feature_root'),
        'metric_scope': 'validation',
        'done_event': done,
        'last_epoch': last.get('best_epoch') if last else None,
    }
    row.update(best)
    return row


def collect_all_runs() -> list[dict]:
    rows = []
    for metrics in sorted(Path('runs').glob('contact_site_esm*/metrics.jsonl')):
        row = row_from_run(metrics.parent, 'hpc2')
        if row:
            rows.append(row)
    if STAR_METRICS.exists():
        cfg = load_yaml_config(STAR_CONFIG)
        row = row_from_run(STAR_METRICS.parent, 'star', cfg)
        if row:
            row['run_name'] = 'star_final_predstruct_context_nograph_20260618_171453'
            row['run_path'] = str(STAR_METRICS.parent)
            rows.append(row)
    return rows


def collect_curated(rows_by_path: dict[str, dict]) -> list[dict]:
    curated = []
    for man in CURATED_MANIFESTS:
        if not man.exists():
            continue
        for line in man.read_text().splitlines()[1:]:
            parts = line.split('\t')
            if len(parts) < 2:
                continue
            outdir = parts[1]
            row = rows_by_path.get(outdir)
            if row:
                curated.append(dict(row, curated=True))
    if STAR_METRICS.exists():
        cfg = load_yaml_config(STAR_CONFIG)
        row = row_from_run(STAR_METRICS.parent, 'star', cfg)
        if row:
            row['run_name'] = 'star_final_predstruct_context_nograph_20260618_171453'
            row['run_path'] = str(STAR_METRICS.parent)
            curated.append(dict(row, curated=True))
    return curated


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text('')
        return
    keys = list(dict.fromkeys(k for row in rows for k in row.keys()))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> pd.DataFrame:
    split_dirs = {str(row.get('split_dir')) for row in rows if row.get('split_dir')}
    if len(split_dirs) != 1:
        raise ValueError(f'Curated rows must use exactly one split_dir, got {sorted(split_dirs)}')
    scopes = {str(row.get('metric_scope')) for row in rows if row.get('metric_scope')}
    if scopes != {'validation'}:
        raise ValueError(f'Curated rows must be validation metrics, got {sorted(scopes)}')
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['model_short']].append(row)
    out = []
    for label, vals in grouped.items():
        item = {'model': label, 'n': len(vals)}
        for key, _ in METRICS:
            xs = [float(v[key]) for v in vals if v.get(key) is not None]
            item[f'{key}_mean'] = mean(xs) if xs else np.nan
            item[f'{key}_std'] = pstdev(xs) if len(xs) > 1 else 0.0
        item['family'] = 'mlc' if label.startswith('MLC') else ('star' if any(v['source']=='star' for v in vals) else 'final')
        out.append(item)
    df = pd.DataFrame(out).sort_values('f1_best_mean', ascending=False)
    return df


def plot_bar(df: pd.DataFrame) -> None:
    df = df.sort_values('f1_best_mean', ascending=True)
    height = max(3.8, 0.36 * len(df) + 0.8)
    fig, ax = plt.subplots(figsize=(7.2, height))
    colors = [COLORS.get(f, COLORS['other']) for f in df['family']]
    ax.barh(df['model'], df['f1_best_mean'], xerr=df['f1_best_std'], color=colors, alpha=0.9, error_kw={'lw': 0.8, 'capsize': 2})
    ax.set_xlabel('Validation F1 at best validation threshold')
    ax.set_xlim(max(0.68, float(df['f1_best_mean'].min()) - 0.01), min(0.74, float(df['f1_best_mean'].max()) + 0.006))
    ax.grid(axis='x', color='0.9', linewidth=0.6)
    for y, (_, row) in enumerate(df.iterrows()):
        ax.text(row['f1_best_mean'] + 0.0007, y, f"{row['f1_best_mean']:.3f}", va='center', ha='left', fontsize=8)
    save_fig(fig, 'fig_model_f1_horizontal_bar')
    plt.close(fig)


def plot_matrix(df: pd.DataFrame) -> None:
    df = df.sort_values('f1_best_mean', ascending=False)
    vals = df[[f'{k}_mean' for k, _ in METRICS]].to_numpy(dtype=float)
    labels = [name for _, name in METRICS]
    fig_w = 7.2
    fig_h = max(3.8, 0.36 * len(df) + 1.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(vals, aspect='auto', cmap='viridis', vmin=np.nanmin(vals), vmax=np.nanmax(vals))
    ax.set_xticks(range(len(labels)), labels=labels, rotation=35, ha='right')
    ax.set_yticks(range(len(df)), labels=df['model'])
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            ax.text(j, i, f'{v:.3f}', ha='center', va='center', color='white' if v < 0.80 else 'black', fontsize=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.set_ylabel('Validation metric value', rotation=270, labelpad=11)
    save_fig(fig, 'fig_metrics_matrix_auc_topk')
    plt.close(fig)


def write_latex(df: pd.DataFrame) -> None:
    cols = ['model', 'n'] + [f'{k}_mean' for k, _ in METRICS]
    table = df[cols].copy()
    rename = {'model': 'Model', 'n': 'Seeds'} | {f'{k}_mean': name for k, name in METRICS}
    table = table.rename(columns=rename)
    for col in table.columns:
        if col not in {'Model', 'Seeds'}:
            table[col] = table[col].map(lambda x: f'{x:.3f}')
    tex = table.to_latex(index=False, escape=True, column_format='lrrrrrrr')
    Path('figures/TABLE_model_results_relaxed.tex').write_text(tex)
    includes = r'''
% === Model F1 comparison ===
\begin{figure}[t]
    \centering
    \includegraphics[width=0.95\textwidth]{figures/fig_model_f1_horizontal_bar.pdf}
    \caption{Validation F1 comparison across sequence-only and predicted-structure-derived feature models on the relaxed MMseq30 split. Error bars denote standard deviation across seeds when multiple seeds are available.}
    \label{fig:model_f1_bar}
\end{figure}

% === Metrics matrix ===
\begin{figure}[t]
    \centering
    \includegraphics[width=0.95\textwidth]{figures/fig_metrics_matrix_auc_topk.pdf}
    \caption{Validation metric matrix for model variants on the relaxed MMseq30 split, including threshold-selected F1, PR-AUC, AUROC, MCC, and top-k precision.}
    \label{fig:metrics_matrix}
\end{figure}
'''.strip() + '\n'
    Path('figures/latex_includes.tex').write_text(includes)


def main() -> None:
    rows = collect_all_runs()
    rows_by_path = {r['run_path']: r for r in rows}
    # Also map HPC run paths without './'.
    rows_by_path.update({str(Path(r['run_path'])): r for r in rows})
    curated = collect_curated(rows_by_path)
    write_csv(FIG / 'model_results_all_runs.csv', rows)
    write_csv(FIG / 'model_results_curated_relaxed.csv', curated)
    summary = summarize(curated)
    summary.to_csv(FIG / 'model_results_curated_relaxed_summary.csv', index=False)
    plot_bar(summary)
    plot_matrix(summary)
    write_latex(summary)
    print('all_runs', len(rows))
    print('curated_runs', len(curated))
    print(summary[['model','n','f1_best_mean','pr_auc_mean','auroc_mean','top5_mean','top10_mean']].to_string(index=False))


if __name__ == '__main__':
    main()
