"""
F4 Step 9 — Final evaluation of the trained agent on the TEST set (1–14 Oct)
============================================================================
Run ONCE, after training and tuning are finished. Routes every test order
with the best model and compares it with all eight baselines on exactly the
same orders.

Input : models/ppo_best.zip, orders_test.json, baseline_results_test.csv,
        training_log.csv (for the learning curve)
Output: rl_results_test.csv        one row per test order, every method
        rl_summary_test.json       totals, head-to-head counts, significance
                                   tests and breakdown by order size
        charts/learning_curve.png  validation distance vs training steps
Run   : python evaluate_agent.py
Needs : scipy (optional, for the Wilcoxon signed-rank test: pip install scipy)
"""

import csv
import json
import os
import time
from collections import defaultdict

from sb3_contrib import MaskablePPO

from train_ppo import evaluate
from warehouse_env_real import RealWarehouseEnv

BASE = ['location_order', 'sshape', 'return', 'midpoint', 'largest_gap', 'combined', 'nearest_neighbour', 'optimal']
METHODS = BASE[:-1] + ['rl_agent', 'optimal']
LABEL = {'location_order': 'Location order', 'sshape': 'S-shape', 'return': 'Return', 'midpoint': 'Midpoint',
         'largest_gap': 'Largest gap', 'combined': 'Combined', 'nearest_neighbour': 'Nearest neighbour',
         'rl_agent': 'RL agent (PPO)', 'optimal': 'Optimal'}
SHORT = {'location_order': 'loc', 'sshape': 'S', 'return': 'ret', 'midpoint': 'mid', 'largest_gap': 'gap',
         'combined': 'comb', 'nearest_neighbour': 'NN', 'rl_agent': 'RL', 'optimal': 'opt'}
VERSUS = ['location_order', 'combined', 'nearest_neighbour']     # head-to-head opponents for RL


def bucket(k):
    return '1' if k == 1 else '2-3' if k <= 3 else '4-7' if k <= 7 else '8-15' if k <= 15 else '16+'


def head_to_head(rows, other):
    diff = [r['rl_agent_m'] - r[f'{other}_m'] for r in rows]
    out = {'rl_shorter': sum(d < -0.01 for d in diff), 'tie': sum(abs(d) <= 0.01 for d in diff),
           'rl_longer': sum(d > 0.01 for d in diff),
           'mean_diff_m': round(sum(diff) / len(diff), 2)}
    nz = [d for d in diff if abs(d) > 0.01]
    try:
        from scipy.stats import wilcoxon
        out['wilcoxon_p'] = float(wilcoxon(nz).pvalue) if len(nz) >= 10 else None
    except ImportError:
        out['wilcoxon_p'] = None
    return out


def summary(rows):
    tot = {m: sum(r[f'{m}_m'] for r in rows) for m in METHODS}
    return {'orders': len(rows),
            'mean_m': {m: round(tot[m] / len(rows), 1) for m in METHODS},
            'total_km': {m: round(tot[m] / 1000, 2) for m in METHODS},
            'pct_above_optimal': {m: round(100 * (tot[m] / tot['optimal'] - 1), 2) for m in METHODS},
            'rl_saving_pct': {m: round(100 * (1 - tot['rl_agent'] / tot[m]), 2) for m in METHODS if m != 'rl_agent'},
            'head_to_head': {m: head_to_head(rows, m) for m in VERSUS}}


def learning_curve(show=True):
    """Plot the validation learning curve from training_log.csv, save it, and open it in a window."""
    if not os.path.exists('training_log.csv'):
        print('training_log.csv not found, skipping the learning curve')
        return
    import matplotlib
    if not show:
        matplotlib.use('Agg')                    
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(open('training_log.csv')))
    x = [int(r['timesteps']) / 1000 for r in rows]
    rl = [float(r['val_mean_m']) for r in rows]
    nn = float(rows[-1]['nn_mean_m'])
    opt = float(rows[-1]['opt_mean_m'])
    comb = float(rows[-1]['combined_mean_m']) if 'combined_mean_m' in rows[-1] else None
    best_i = min(range(len(rl)), key=rl.__getitem__)

    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.plot(x, rl, color='#C8102E', lw=2.4, marker='o', ms=4, label='RL agent (PPO)', zorder=3)
    if comb is not None:
        ax.axhline(comb, color='#E0A43A', lw=1.6, ls='-.', label=f'Combined (classic rule)  {comb:.1f} m')
    ax.axhline(nn, color='#2F6DB5', lw=1.6, ls='--', label=f'Nearest neighbour  {nn:.1f} m')
    ax.axhline(opt, color='#111111', lw=1.6, ls=':', label=f'Optimal  {opt:.1f} m')

    # mark the saved best model
    bx, by = x[best_i], rl[best_i]
    gap = 100 * (by / opt - 1)
    ax.scatter([bx], [by], s=90, color='#C8102E', edgecolor='white', linewidth=1.5, zorder=4)
    ax.annotate(f'best model: {by:.1f} m (+{gap:.2f}% vs optimal)\nat {int(bx)}k steps',
                xy=(bx, by), xytext=(0, 34), textcoords='offset points', ha='center', fontsize=10,
                arrowprops=dict(arrowstyle='-', color='#888888', lw=1))

    # mark where the agent first beats each baseline
    for level, name in [(comb, 'Combined'), (nn, 'nearest neighbour')]:
        if level is None:
            continue
        first = next((i for i, v in enumerate(rl) if v < level), None)
        if first is not None:
            ax.axvline(x[first], color='#BBBBBB', lw=0.8, ls=(0, (2, 3)), zorder=1)
            ax.text(x[first], max(rl) - 0.4, f' beats {name}\n {int(x[first])}k steps',
                    fontsize=9, color='#555555', va='top')

    n_val = len(rows) and 600
    ax.set_xlabel('Training steps (thousands)')
    ax.set_ylabel('Mean trip per order (m)')
    ax.set_title(f'RL agent learning curve on {n_val} September validation orders', loc='left', weight='bold')
    ax.set_ylim(opt - 2, max(rl) + 2)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, loc='upper right')
    fig.tight_layout()

    os.makedirs('charts', exist_ok=True)
    fig.savefig('charts/learning_curve.png', dpi=200, bbox_inches='tight')
    print('Wrote charts/learning_curve.png')
    if show:
        print('Close the chart window to finish.')
        plt.show()

def main():
    base = {r['so_no']: r for r in csv.DictReader(open('baseline_results_test.csv', encoding='utf-8'))}
    if 'combined_m' not in next(iter(base.values())):
        raise SystemExit('baseline_results_test.csv is from the old baselines.py: rerun python baselines.py (step 7).')
    env = RealWarehouseEnv('orders_test.json', min_stops=1)
    model = MaskablePPO.load('models/ppo_best.zip')
    t0 = time.perf_counter()
    res = {r['so_no']: r for r in evaluate(model, env)}
    ms_per_order = 1000 * (time.perf_counter() - t0) / len(res)

    rows = []
    for so, b in base.items():
        r = res[so]
        row = {'so_no': so, 'lines': int(b['lines']), 'stops': int(b['stops']), 'size_bucket': bucket(int(b['lines']))}
        for m in BASE:
            row[f'{m}_m'] = float(b[f'{m}_m'])
        row['rl_agent_m'] = r['trip_m']
        for m in BASE:
            row[f'{m}_stairs'] = int(b[f'{m}_stairs'])
        row['rl_agent_stairs'] = r['stairs']
        rows.append(row)
    with open('rl_results_test.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    by_b = defaultdict(list)
    for r in rows:
        by_b[r['size_bucket']].append(r)
    out = {'all': summary(rows), 'rl_decision_ms_per_order': round(ms_per_order, 2),
           'by_order_size': {b: summary(by_b[b]) for b in ['1', '2-3', '4-7', '8-15', '16+'] if by_b[b]}}
    json.dump(out, open('rl_summary_test.json', 'w'), indent=1)

    a = out['all']
    print('=' * 78)
    print(f"  TEST set (1–14 Oct): {a['orders']} orders · RL plans a route in {out['rl_decision_ms_per_order']} ms per order")
    print(f"  {'method':20s} {'mean m':>8s} {'total km':>9s} {'vs optimal':>11s} {'RL saves':>9s}")
    for m in METHODS:
        save = '' if m == 'rl_agent' else f"{a['rl_saving_pct'][m]:8.2f}%"
        print(f"  {LABEL[m]:20s} {a['mean_m'][m]:8.1f} {a['total_km'][m]:9.2f} {a['pct_above_optimal'][m]:10.2f}% {save:>9s}")
    print('-' * 78)
    print('  RL head to head (orders where RL is shorter / same / longer; Wilcoxon signed-rank p)')
    for m in VERSUS:
        h = a['head_to_head'][m]
        p = 'n/a (pip install scipy)' if h['wilcoxon_p'] is None else f"{h['wilcoxon_p']:.2e}"
        print(f"  vs {LABEL[m]:18s} {h['rl_shorter']:5d} / {h['tie']:5d} / {h['rl_longer']:5d}   "
              f"mean {h['mean_diff_m']:+.2f} m per order   p = {p}")
    print('-' * 78)
    print('  % above optimal by order size')
    print(f"  {'lines':8s} {'orders':>6s} " + ' '.join(f'{SHORT[m]:>6s}' for m in METHODS[:-1]))
    for b, s in out['by_order_size'].items():
        print(f"  {b:8s} {s['orders']:6d} " + ' '.join(f"{s['pct_above_optimal'][m]:+6.1f}" for m in METHODS[:-1]))
    print('=' * 78)
    learning_curve()


if __name__ == '__main__':
    main()
