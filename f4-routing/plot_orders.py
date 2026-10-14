"""
F4 Order Charts — Fine Enterprises WIS
======================================
Charts of the cleaned September orders for the report and the slides.

Input : orders_clean.json, warehouse_graph.json, order_profile.json
Output: charts/dark/*.png   (black-and-red slide template)
        charts/light/*.png  (white background, for the written report)
Run   : python plot_orders.py            (both themes)
        python plot_orders.py dark       (one theme)
Needs : matplotlib  (pip install matplotlib)
"""

import datetime as dt
import json
import os
import sys
from collections import Counter, defaultdict
from itertools import combinations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

# Palette validated with the dataviz validator (2 series, all pairs, both modes):
#   dark : ground #E5383B / mezzanine #5B8DC9 on #161616
#   light: ground #C8102E / mezzanine #2F6DB5 on #FCFCFB
THEMES = {
    'dark': dict(bg='#161616', panel='#1E1E1E', ink='#F2F2F2', ink2='#B8B8B8', muted='#7A7A7A',
                 grid='#2E2E2E', rack='#3A3A3A', ground='#E5383B', mezz='#5B8DC9',
                 seq=['#2A1416', '#6B1F22', '#A82A2E', '#E5383B', '#FF8A8C']),
    'light': dict(bg='#FCFCFB', panel='#FFFFFF', ink='#111111', ink2='#4A4A4A', muted='#8A8A8A',
                  grid='#E6E6E4', rack='#D9D9D6', ground='#C8102E', mezz='#2F6DB5',
                  seq=['#FBE9EA', '#F2B8BC', '#E06A72', '#C8102E', '#7A0A1C']),
}
DPI = 200
SIZE = (12, 6.75)   # 16:9

# Rack outlines from warehouse_editor_v3 (same as validate_layout.py)
OUTLINES = [
    ('B/C', 65, 0, 80, 340), ('A', 195, 0, 40, 340), ('D', 235, 0, 45, 340),
    ('E', 325, 0, 35, 340), ('R', 370, 0, 40, 185), ('Q', 370, 190, 40, 70),
    ('P', 370, 265, 40, 70), ('X-3/X-5', 470, 0, 150, 335),
    ('O', 665, 0, 50, 40), ('O', 720, 0, 40, 170), ('N', 720, 175, 40, 115),
    ('M', 720, 295, 40, 140), ('L', 720, 440, 40, 110),
    ('G', 65, 385, 50, 175), ('G', 185, 410, 35, 140), ('G', 222, 410, 28, 140),
    ('G', 320, 385, 50, 175), ('T', 375, 385, 40, 90), ('U', 375, 480, 40, 80),
    ('X-2/4', 510, 385, 45, 175), ('X6-9', 560, 385, 50, 175), ('X-1', 100, 585, 645, 110),
    ('I', 65, 590, 30, 110), ('J', 65, 730, 345, 30), ('K', 415, 730, 350, 30),
    ('W', 810, 0, 50, 600),
]
MEZZ_OUTLINES = {'X-3/X-5', 'X-2/4', 'W', 'T', 'U'}


def style(t):
    plt.rcParams.update({
        'figure.facecolor': t['bg'], 'axes.facecolor': t['bg'], 'savefig.facecolor': t['bg'],
        'text.color': t['ink'], 'axes.labelcolor': t['ink2'], 'axes.edgecolor': t['grid'],
        'xtick.color': t['ink2'], 'ytick.color': t['ink2'], 'axes.titlecolor': t['ink'],
        'font.family': 'DejaVu Sans', 'font.size': 12, 'axes.titlesize': 17,
        'axes.titleweight': 'bold', 'axes.titlelocation': 'left', 'axes.titlepad': 30,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
        'grid.color': t['grid'], 'grid.linewidth': 0.8, 'axes.axisbelow': True,
        'legend.frameon': False, 'legend.labelcolor': t['ink2'],
    })


def subtitle(ax, text, t):
    ax.text(0, 1.012, text, transform=ax.transAxes, color=t['ink2'], fontsize=11, va='bottom')


def save(fig, out, name):
    fig.savefig(os.path.join(out, name), dpi=DPI, bbox_inches='tight', pad_inches=0.35)
    plt.close(fig)


def floor_of(slot_code, slots):
    return slots[slot_code]['floor']


# ─────────────────────────────────────────────────────────────────────────────

def chart_order_sizes(orders, t, out):
    lines = [o['n_lines'] for o in orders]
    c = Counter(lines)
    xs = list(range(1, max(lines) + 1))
    fig, ax = plt.subplots(figsize=SIZE)
    ax.bar(xs, [c.get(x, 0) for x in xs], width=0.8, color=t['ground'], edgecolor=t['bg'], linewidth=1)
    med = sorted(lines)[len(lines) // 2]
    mean = sum(lines) / len(lines)
    for v, lab, ls in ((med, f'median {med}', '-'), (mean, f'mean {mean:.1f}', '--')):
        ax.axvline(v, color=t['ink2'], lw=1.2, ls=ls)
        ax.text(v + 0.3, max(c.values()) * (0.55 if ls == '-' else 0.47), lab, color=t['ink2'], fontsize=11)
    ax.set_title('Most orders are small, but a long tail of large orders remains')
    subtitle(ax, f'Order lines per order · {len(orders):,} cleaned orders, September 2026', t)
    ax.set_xlabel('Lines in the order')
    ax.set_ylabel('Orders')
    ax.set_xlim(0.3, max(xs) + 0.7)
    ax.grid(axis='x', visible=False)
    one = c.get(1, 0)
    ax.annotate(f'{one} single-line orders ({100 * one / len(orders):.0f}%)', xy=(1, one),
                xytext=(9, one * 0.85), color=t['ink'], fontsize=11,
                arrowprops=dict(arrowstyle='-', color=t['muted'], lw=1))
    big = sum(1 for x in lines if x >= 16)
    ax.text(23, max(c.values()) * 0.45, f'{big} orders ({100 * big / len(orders):.0f}%)\nhave 16+ lines',
            color=t['ink'], fontsize=11, ha='center')
    save(fig, out, '01_order_sizes.png')


def chart_orders_per_day(orders, t, out):
    per_day = Counter(o['date'] for o in orders)
    days = sorted(per_day)
    dates = [dt.date.fromisoformat(d) for d in days]
    weekend = [d.weekday() >= 5 for d in dates]
    fig, ax = plt.subplots(figsize=SIZE)
    ax.bar(range(len(days)), [per_day[d] for d in days], width=0.78,
           color=[t['mezz'] if w else t['ground'] for w in weekend], edgecolor=t['bg'], linewidth=1)
    ax.set_xticks(range(len(days)))
    ax.set_xticklabels([f'{d.day}\n{d:%a}' for d in dates], fontsize=8.5)
    avg = sum(per_day.values()) / len(per_day)
    ax.axhline(avg, color=t['ink2'], lw=1.2, ls='--')
    ax.text(len(days) - 0.5, avg + 2, f'average {avg:.0f}/day', color=t['ink2'], ha='right', fontsize=11)
    peak = max(days, key=per_day.get)
    i = days.index(peak)
    ax.text(i, per_day[peak] + 3, f'{per_day[peak]}', ha='center', color=t['ink'], fontsize=11, weight='bold')
    ax.set_title('Daily order volume swings from under 10 to over 160')
    subtitle(ax, 'Cleaned orders per day, September 2026', t)
    ax.set_ylabel('Orders')
    ax.grid(axis='x', visible=False)
    ax.legend(handles=[Patch(color=t['ground'], label='Weekday'), Patch(color=t['mezz'], label='Weekend')],
              loc='upper left', ncol=2)
    save(fig, out, '02_orders_per_day.png')


def chart_lines_by_section(orders, slots, t, out):
    g, m = Counter(), Counter()
    for o in orders:
        for l in o['lines']:
            sec = l['location'].split('-')[0]
            (m if l['floor'] == 'M' else g)[sec] += 1
    secs = sorted(set(g) | set(m), key=lambda s: g[s] + m[s])
    fig, ax = plt.subplots(figsize=(12, 8))
    y = range(len(secs))
    ax.barh(y, [g[s] for s in secs], height=0.72, color=t['ground'], edgecolor=t['bg'], linewidth=1.5,
            label='Ground floor')
    ax.barh(y, [m[s] for s in secs], left=[g[s] for s in secs], height=0.72, color=t['mezz'],
            edgecolor=t['bg'], linewidth=1.5, label='Mezzanine (stairs needed)')
    for i, s in enumerate(secs):
        ax.text(g[s] + m[s] + 18, i, f'{g[s] + m[s]:,}', va='center', color=t['ink2'], fontsize=10)
    ax.set_yticks(list(y))
    ax.set_yticklabels(secs)
    ax.grid(axis='y', visible=False)
    tot_m = sum(m.values())
    tot = tot_m + sum(g.values())
    ax.set_title(f'{100 * tot_m / tot:.0f}% of picks are on the mezzanine — Z and W rank 2nd and 3rd')
    subtitle(ax, 'Order lines by location section, September 2026', t)
    ax.set_xlabel('Order lines')
    ax.legend(loc='lower right')
    save(fig, out, '03_lines_by_section.png')


def chart_floors_and_stairs(orders, slots, t, out):
    mix = Counter('+'.join(sorted({l['floor'] for l in o['lines']})) for o in orders)
    groups = {'SC1': {'MZV', 'MZS', 'MZX5', 'MZV4'}, 'SC2': {'MZTU', 'MZZ'}, 'SC3': {'MZW'}}
    per = Counter()
    for o in orders:
        k = sum(1 for walks in groups.values()
                if any(l['floor'] == 'M' and slots[l['location']]['walkway'] in walks for l in o['lines']))
        per[k] += 1
    n = len(orders)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=SIZE, gridspec_kw={'wspace': 0.35})
    labels = [('G', 'Ground only'), ('G+M', 'Ground + mezzanine'), ('M', 'Mezzanine only')]
    vals = [mix.get(k, 0) for k, _ in labels]
    cols = [t['ground'], t['muted'], t['mezz']]
    a1.bar(range(3), vals, color=cols, width=0.62, edgecolor=t['bg'], linewidth=1.5)
    for i, v in enumerate(vals):
        a1.text(i, v + n * 0.01, f'{v:,}\n{100 * v / n:.0f}%', ha='center', va='bottom', color=t['ink'], fontsize=11)
    a1.set_xticks(range(3))
    a1.set_xticklabels([l for _, l in labels], fontsize=10.5)
    a1.set_ylim(0, max(vals) * 1.25)
    a1.set_ylabel('Orders')
    a1.grid(axis='x', visible=False)
    a1.set_title('Floors an order touches', fontsize=14)
    ks = [0, 1, 2, 3]
    vals2 = [per.get(k, 0) for k in ks]
    a2.bar(ks, vals2, color=[t['ground'], t['mezz'], t['mezz'], t['mezz']], width=0.62,
           edgecolor=t['bg'], linewidth=1.5)
    for k, v in zip(ks, vals2):
        a2.text(k, v + n * 0.01, f'{v:,}', ha='center', va='bottom', color=t['ink'], fontsize=11)
    a2.set_xticks(ks)
    a2.set_xticklabels(['None', 'One', 'Two', 'Three'])
    a2.set_ylim(0, max(vals2) * 1.25)
    a2.set_xlabel('Different staircases needed')
    a2.grid(axis='x', visible=False)
    a2.set_title('Staircases per order', fontsize=14)
    mezz = n - mix.get('G', 0)
    fig.suptitle(f'{100 * mezz / n:.0f}% of orders need the mezzanine; {per[2] + per[3]} need two or more staircases',
                 x=0.07, y=1.02, ha='left', fontsize=17, weight='bold', color=t['ink'])
    save(fig, out, '04_floors_and_stairs.png')


def chart_pareto(orders, t, out):
    c = Counter(l['sku'] for o in orders for l in o['lines'])
    vals = sorted(c.values(), reverse=True)
    tot, n = sum(vals), len(vals)
    cum, xs, ys = 0, [0], [0]
    for i, v in enumerate(vals, 1):
        cum += v
        xs.append(100 * i / n)
        ys.append(100 * cum / tot)
    x80 = next(x for x, y in zip(xs, ys) if y >= 80)
    fig, ax = plt.subplots(figsize=SIZE)
    ax.plot([0, 100], [0, 100], color=t['muted'], lw=1, ls=':')
    ax.text(62, 56, 'every item equally popular', color=t['muted'], fontsize=10, rotation=28)
    ax.plot(xs, ys, color=t['ground'], lw=2.2)
    ax.plot([x80, x80, 0], [0, 80, 80], color=t['ink2'], lw=1, ls='--')
    ax.scatter([x80], [80], s=60, color=t['ground'], edgecolor=t['bg'], linewidth=2, zorder=5)
    ax.text(x80 + 2, 74, f'{x80:.0f}% of items make up 80% of picks', color=t['ink'], fontsize=12)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 102)
    ax.set_xlabel('Share of ordered items (most-picked first, %)')
    ax.set_ylabel('Share of order lines (%)')
    ax.set_title('Demand is spread thin — no small group of fast movers')
    subtitle(ax, f'Cumulative share of order lines across {n:,} items, September 2026', t)
    save(fig, out, '05_item_pareto.png')


def chart_section_pairs(orders, t, out, top=14):
    sec_lines = Counter(l['location'].split('-')[0] for o in orders for l in o['lines'])
    secs = [s for s, _ in sec_lines.most_common(top)]
    pair = Counter()
    for o in orders:
        ss = sorted({l['location'].split('-')[0] for l in o['lines']} & set(secs))
        pair.update(combinations(ss, 2))
    import numpy as np
    M = np.full((len(secs), len(secs)), np.nan)
    for i, a in enumerate(secs):
        for j, b in enumerate(secs):
            if i > j:
                M[i, j] = pair.get(tuple(sorted((a, b))), 0)
    cmap = LinearSegmentedColormap.from_list('seq', t['seq'])
    cmap.set_bad(t['bg'])
    M = M[1:, :-1]
    rows, cols = secs[1:], secs[:-1]
    fig, ax = plt.subplots(figsize=(10.5, 9))
    im = ax.imshow(M, cmap=cmap)
    mx = np.nanmax(M)
    for i in range(len(rows)):
        for j in range(i + 1):
            v = int(M[i, j])
            dark_cell = (v / mx) > 0.55 if t is THEMES['light'] else (v / mx) < 0.75
            ax.text(j, i, v, ha='center', va='center', fontsize=8.5,
                    color='#FFFFFF' if dark_cell else '#111111')
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label('Orders containing both sections', color=t['ink2'])
    cb.outline.set_visible(False)
    ax.set_title('Sections ordered together cross floors and ends of the warehouse')
    subtitle(ax, f'Orders that contain both sections · {top} busiest sections, September 2026', t)
    save(fig, out, '06_section_pairs.png')


def chart_pick_heatmap(orders, graph, t, out):
    nodes = graph['nodes']
    hits = Counter(l['pick_node'] for o in orders for l in o['lines'])
    cmap = LinearSegmentedColormap.from_list('seq', t['seq'][1:])
    mx = max(hits.values())
    fig, axes = plt.subplots(1, 2, figsize=(16, 8.6), gridspec_kw={'wspace': 0.04})
    for ax, floor, label in ((axes[0], 'G', 'Ground floor'), (axes[1], 'M', 'Mezzanine')):
        ax.set_aspect('equal')
        ax.set_xlim(-25, 885)
        ax.set_ylim(800, -40)
        ax.axis('off')
        ax.add_patch(Rectangle((0, 0), 860, 780, fill=False, edgecolor=t['muted'], lw=1))
        for name, x, y, w, h in OUTLINES:
            on = (name != 'W') if floor == 'G' else (name == 'W')
            ax.add_patch(Rectangle((x, y), w, h, color=t['rack'], alpha=1 if on else 0.35, lw=0))
            if on and w * h > 2500:
                ax.text(x + w / 2, y + h / 2, name, ha='center', va='center', fontsize=8,
                        color=t['ink2'], rotation=90 if h > 2 * w else 0)
        pts = [(n['x'], n['y'], hits.get(k, 0), k) for k, n in nodes.items()
               if n['floor'] == floor and n['pick_slots']]
        pts.sort(key=lambda p: p[2])
        ax.scatter([p[0] for p in pts], [p[1] for p in pts],
                   s=[12 + 260 * (p[2] / mx) for p in pts], c=[p[2] for p in pts],
                   cmap=cmap, vmin=0, vmax=mx, edgecolor=t['bg'], linewidth=1, zorder=4)
        for p in sorted(pts, key=lambda p: -p[2])[:3]:
            ax.text(p[0] + 12, p[1] - 12, f'{p[3]} · {p[2]}', fontsize=8.5, color=t['ink'], zorder=5)
        if floor == 'G':
            ax.scatter([0], [362], s=120, marker='s', color=t['ink'], zorder=5)
            ax.text(8, 392, 'DEPOT', color=t['ink'], fontsize=10, weight='bold')
        tot = sum(p[2] for p in pts)
        ax.set_title(f'{label} · {tot:,} lines', fontsize=14, loc='left')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, mx))
    cb = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.03)
    cb.set_label('Order lines picked at this spot', color=t['ink2'])
    cb.outline.set_visible(False)
    fig.suptitle('Where pickers go: pick density by stand-point', x=0.06, y=0.98, ha='left',
                 fontsize=17, weight='bold', color=t['ink'])
    fig.text(0.06, 0.935, 'Each dot is a place a picker stands; bigger and brighter = more order lines, September 2026',
             color=t['ink2'], fontsize=11)
    save(fig, out, '07_pick_heatmap.png')


def run(themes=('dark', 'light')):
    orders = json.load(open('orders_clean.json', encoding='utf-8'))['orders']
    graph = json.load(open('warehouse_graph.json', encoding='utf-8'))
    slots = graph['slots']
    for name in themes:
        t = THEMES[name]
        out = os.path.join('charts', name)
        os.makedirs(out, exist_ok=True)
        style(t)
        chart_order_sizes(orders, t, out)
        chart_orders_per_day(orders, t, out)
        chart_lines_by_section(orders, slots, t, out)
        chart_floors_and_stairs(orders, slots, t, out)
        chart_pareto(orders, t, out)
        chart_section_pairs(orders, t, out)
        chart_pick_heatmap(orders, graph, t, out)
        print(f'Wrote 7 charts to {out}/')


if __name__ == '__main__':
    run(tuple(sys.argv[1:]) or ('dark', 'light'))
