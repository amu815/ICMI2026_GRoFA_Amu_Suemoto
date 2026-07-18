#!/usr/bin/env python3
"""Phase 1: Wilcoxon signed-rank + bootstrap CI + per-seed BM> analysis.

Computes for FairFace (Ens5+WFT0.95) and UTKFace (Ens5+WFT0.60):
  - Per-seed BM> (DART vs Baseline) over 66 conditions
  - Wilcoxon signed-rank test (paired, over 66 conditions, per-seed mean)
  - Paired bootstrap CI (1000 resamples) for BM> count
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import json
import os

BASE = '.'
OUT_DIR = 'results_phase1'
os.makedirs(OUT_DIR, exist_ok=True)

RESULTS = {
    'FairFace': 'logs/fairface_5seed_wiseft095/raw_per_seed.csv',
    'UTKFace':  'logs/utkface_5seed_wiseft060/raw_per_seed.csv',
}

rng = np.random.default_rng(0)

def paired_bootstrap_bm(baseline_acc, ours_acc, n_boot=1000):
    """Bootstrap CI for BM> count."""
    n = len(baseline_acc)
    counts = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        b = baseline_acc[idx]
        o = ours_acc[idx]
        counts.append(int(np.sum(o > b)))
    counts = np.array(counts)
    return counts.mean(), np.percentile(counts, 2.5), np.percentile(counts, 97.5)


def paired_bootstrap_delta(baseline_acc, ours_acc, n_boot=1000):
    """Bootstrap CI for mean accuracy delta."""
    n = len(baseline_acc)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas.append(float(np.mean(ours_acc[idx] - baseline_acc[idx])))
    deltas = np.array(deltas)
    return deltas.mean(), np.percentile(deltas, 2.5), np.percentile(deltas, 97.5)


all_results = {}
for dataset, path in RESULTS.items():
    df = pd.read_csv(path)
    print(f"\n=== {dataset} ===")
    print(f"Loaded {len(df)} rows, seeds={sorted(df['Seed'].unique())}, "
          f"tasks={sorted(df['Task'].unique())}, models={sorted(df['Model'].unique())}")

    bm = df[df['Model'] == 'Baseline'][['Task','Noise_Type','Noise_Level','Seed','Acc']].copy()
    our = df[df['Model'] == 'Ours'][['Task','Noise_Type','Noise_Level','Seed','Acc']].copy()
    bm = bm.rename(columns={'Acc': 'bm'})
    our = our.rename(columns={'Acc': 'our'})
    m = pd.merge(bm, our, on=['Task','Noise_Type','Noise_Level','Seed'])
    print(f"Merged rows: {len(m)}")

    per_seed_bm = {}
    for s in sorted(m['Seed'].unique()):
        sub = m[m['Seed'] == s]
        wins = int(np.sum(sub['our'].values > sub['bm'].values))
        per_seed_bm[int(s)] = wins
    print(f"Per-seed BM> (raw, before ensemble): {per_seed_bm}")

    agg = m.groupby(['Task','Noise_Type','Noise_Level']).agg(
        bm_mean=('bm', 'mean'),
        our_mean=('our', 'mean'),
    ).reset_index()
    bm_arr = agg['bm_mean'].values
    our_arr = agg['our_mean'].values
    n_cond = len(agg)
    wins = int(np.sum(our_arr > bm_arr))
    losses = int(np.sum(our_arr < bm_arr))
    ties = int(np.sum(our_arr == bm_arr))
    print(f"Per-condition seed-mean: {n_cond} conds, wins={wins}, losses={losses}, ties={ties}")

    try:
        w_stat, w_p = wilcoxon(our_arr, bm_arr, alternative='greater', zero_method='wilcox')
        w_stat2, w_p2 = wilcoxon(our_arr, bm_arr, alternative='two-sided', zero_method='wilcox')
    except Exception as e:
        print(f"Wilcoxon error: {e}")
        w_stat = w_p = w_stat2 = w_p2 = None

    bm_mean_cnt, bm_lo, bm_hi = paired_bootstrap_bm(bm_arr, our_arr)
    d_mean, d_lo, d_hi = paired_bootstrap_delta(bm_arr, our_arr)

    print(f"Wilcoxon (greater):  stat={w_stat:.4f}, p={w_p:.4e}")
    print(f"Wilcoxon (two-side): stat={w_stat2:.4f}, p={w_p2:.4e}")
    print(f"Bootstrap BM> count: {bm_mean_cnt:.1f} [95% CI: {bm_lo:.0f}, {bm_hi:.0f}]")
    print(f"Bootstrap mean Delta Acc: {d_mean:+.4f} [95% CI: {d_lo:+.4f}, {d_hi:+.4f}]")

    per_task_wilcoxon = {}
    for task in ['race', 'gender', 'age']:
        sub = agg[agg['Task'] == task]
        b = sub['bm_mean'].values
        o = sub['our_mean'].values
        try:
            ws, wp = wilcoxon(o, b, alternative='greater', zero_method='wilcox')
            per_task_wilcoxon[task] = {
                'n': len(b),
                'wins': int(np.sum(o > b)),
                'mean_delta': float(np.mean(o - b)),
                'wilcoxon_stat': float(ws),
                'wilcoxon_p': float(wp),
            }
        except Exception as e:
            per_task_wilcoxon[task] = {'error': str(e)}

    all_results[dataset] = {
        'n_conditions': int(n_cond),
        'wins': wins,
        'losses': losses,
        'ties': ties,
        'per_seed_bm_raw': per_seed_bm,
        'wilcoxon_greater': {'stat': float(w_stat), 'p': float(w_p)},
        'wilcoxon_two_sided': {'stat': float(w_stat2), 'p': float(w_p2)},
        'bootstrap_bm_count': {'mean': float(bm_mean_cnt), 'ci_lo': float(bm_lo), 'ci_hi': float(bm_hi)},
        'bootstrap_delta_acc': {'mean': float(d_mean), 'ci_lo': float(d_lo), 'ci_hi': float(d_hi)},
        'per_task': per_task_wilcoxon,
    }

with open(f'{OUT_DIR}/phase1_stats.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved: {OUT_DIR}/phase1_stats.json")
