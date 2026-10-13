"""
F4 Order Profile — Fine Enterprises WIS
=======================================
Step 2 of the order pipeline. Describes the cleaned September orders so that
  (a) the results chapter can report what real demand looks like, and
  (b) the synthetic order generator (Step 3) can copy the same patterns.

Input : orders_clean.json, warehouse_graph.json
Output: order_profile.json  (all statistics + sampling tables for Step 3)
Run   : python analyze_orders.py
"""

import datetime as dt
import json
import statistics as st
from collections import Counter, defaultdict
from itertools import combinations


def pct(a, b):
    return round(100 * a / b, 1) if b else 0.0


def quantiles(xs):
    xs = sorted(xs)
    q = lambda p: xs[min(len(xs) - 1, int(p * len(xs)))]
    return {'min': xs[0], 'p25': q(.25), 'median': q(.5), 'p75': q(.75),
            'p90': q(.9), 'max': xs[-1], 'mean': round(st.mean(xs), 2)}


def run(orders_path='orders_clean.json', graph_path='warehouse_graph.json'):
    orders = json.load(open(orders_path, encoding='utf-8'))['orders']
    graph = json.load(open(graph_path, encoding='utf-8'))
    slots = graph['slots']
    n = len(orders)

    # ── order size ───────────────────────────────────────────────────────────
    lines = [o['n_lines'] for o in orders]
    stops = [len({l['pick_node'] for l in o['lines']}) for o in orders]
    size_hist = Counter(lines)
    buckets = Counter('1' if k == 1 else '2-3' if k <= 3 else '4-7' if k <= 7
                      else '8-15' if k <= 15 else '16+' for k in lines)

    # ── time ────────────────────────────────────────────────────────────────
    per_day = Counter(o['date'] for o in orders)
    weekday = Counter(dt.date.fromisoformat(o['date']).strftime('%a') for o in orders)

    # ── floors and stairs ───────────────────────────────────────────────────
    floor_mix = Counter('+'.join(sorted({l['floor'] for l in o['lines']})) for o in orders)
    stair_use = Counter()
    for o in orders:
        used = set()
        for l in o['lines']:
            if l['floor'] == 'M':
                walk = slots[l['location']]['walkway']
                used.add({'MZV': 'SC1', 'MZS': 'SC1', 'MZX5': 'SC1', 'MZV4': 'SC1', 'MZ1': 'SC1',
                          'MZTU': 'SC2', 'MZZ': 'SC2', 'MZ2': 'SC2', 'MZW': 'SC3'}.get(walk, '?'))
        for s in used:
            stair_use[s] += 1
    stairs_per_order = Counter()
    for o in orders:
        k = 0
        for s, walks in (('SC1', {'MZV', 'MZS', 'MZX5', 'MZV4'}), ('SC2', {'MZTU', 'MZZ'}), ('SC3', {'MZW'})):
            if any(l['floor'] == 'M' and slots[l['location']]['walkway'] in walks for l in o['lines']):
                k += 1
        stairs_per_order[k] += 1

    # ── item popularity (ABC by order lines) ────────────────────────────────
    sku_lines = Counter(l['sku'] for o in orders for l in o['lines'])
    ranked = sku_lines.most_common()
    total = sum(sku_lines.values())
    cum, abc = 0, {}
    for sku, c in ranked:
        cum += c
        abc[sku] = 'A' if cum <= .8 * total else 'B' if cum <= .95 * total else 'C'
    abc_count = Counter(abc.values())
    sku_name = {l['sku']: l['item_name'] for o in orders for l in o['lines']}

    # where the A items live
    sku_loc = {l['sku']: l['location'] for o in orders for l in o['lines']}
    a_sections = Counter(sku_loc[s].split('-')[0] for s, c in abc.items() if c == 'A')
    node_lines = Counter(l['pick_node'] for o in orders for l in o['lines'])

    # ── section demand ──────────────────────────────────────────────────────
    section_lines = Counter(l['location'].split('-')[0] for o in orders for l in o['lines'])

    # ── co-occurrence (pairs of sections and of SKUs in the same order) ─────
    pair_sec, pair_sku = Counter(), Counter()
    for o in orders:
        secs = sorted({l['location'].split('-')[0] for l in o['lines']})
        pair_sec.update(combinations(secs, 2))
        skus = sorted({l['sku'] for l in o['lines']})
        if len(skus) <= 30:
            pair_sku.update(combinations(skus, 2))

    # ── quantities ──────────────────────────────────────────────────────────
    qty = [l['qty'] for o in orders for l in o['lines'] if l['qty'] > 0]

    profile = {
        'orders': n,
        'lines': sum(lines),
        'period': f"{min(per_day)} to {max(per_day)}",
        'lines_per_order': quantiles(lines),
        'stops_per_order': quantiles(stops),
        'order_size_buckets': {k: {'orders': buckets[k], 'pct': pct(buckets[k], n)}
                               for k in ['1', '2-3', '4-7', '8-15', '16+']},
        'orders_by_series': dict(Counter(o['series'] for o in orders)),
        'orders_by_delivery_type': dict(Counter(o['delivery_type'] for o in orders)),
        'orders_per_day': quantiles(list(per_day.values())),
        'orders_per_weekday': {d: weekday.get(d, 0) for d in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']},
        'floor_mix': {k: {'orders': v, 'pct': pct(v, n)} for k, v in floor_mix.items()},
        'orders_using_staircase': dict(stair_use),
        'staircases_per_order': {str(k): v for k, v in sorted(stairs_per_order.items())},
        'qty_per_line': quantiles(qty),
        'distinct_skus': len(sku_lines),
        'abc': {c: {'skus': abc_count[c], 'pct_of_skus': pct(abc_count[c], len(sku_lines))}
                for c in 'ABC'},
        'a_items_by_section': dict(a_sections.most_common()),
        'top_20_skus': [{'sku': s, 'item': sku_name[s], 'location': sku_loc[s], 'order_lines': c}
                        for s, c in ranked[:20]],
        'lines_by_section': dict(section_lines.most_common()),
        'busiest_stand_points': dict(node_lines.most_common(15)),
        'top_section_pairs': [{'pair': '+'.join(p), 'orders': c} for p, c in pair_sec.most_common(15)],
        'top_sku_pairs': [{'pair': [sku_name[a], sku_name[b]], 'orders': c}
                          for (a, b), c in pair_sku.most_common(10)],
        # sampling tables used by the synthetic generator (Step 3)
        'sampling': {
            'lines_per_order_hist': dict(sorted(size_hist.items())),
            'sku_line_counts': dict(ranked),
            'sku_location': sku_loc,
        },
    }
    json.dump(profile, open('order_profile.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)

    p = profile
    print('=' * 66)
    print(f"  {p['orders']:,} orders · {p['lines']:,} lines · {p['period']}")
    print(f"  Lines/order : {p['lines_per_order']}")
    print(f"  Stops/order : {p['stops_per_order']}")
    print('  Order sizes : ' + ' | '.join(f"{k}: {v['orders']} ({v['pct']}%)" for k, v in p['order_size_buckets'].items()))
    print(f"  Orders/day  : {p['orders_per_day']}")
    print(f"  Weekdays    : {p['orders_per_weekday']}")
    print('  Floors      : ' + ' | '.join(f"{k}: {v['orders']} ({v['pct']}%)" for k, v in p['floor_mix'].items()))
    print(f"  Stair use   : {p['orders_using_staircase']}   staircases/order {p['staircases_per_order']}")
    print(f"  Qty/line    : {p['qty_per_line']}")
    print(f"  SKUs        : {p['distinct_skus']:,}  ABC " +
          ' '.join(f"{c}={v['skus']} ({v['pct_of_skus']}%)" for c, v in p['abc'].items()))
    print(f"  A items by section: {p['a_items_by_section']}")
    print(f"  Lines by section  : {p['lines_by_section']}")
    print('  Top section pairs : ' + ', '.join(f"{x['pair']} {x['orders']}" for x in p['top_section_pairs'][:8]))
    print('  Top 5 SKUs:')
    for x in p['top_20_skus'][:5]:
        print(f"    {x['order_lines']:4d}  {x['location']:6s} {x['item']}")
    print('=' * 66)
    print('Wrote order_profile.json')
    return profile


if __name__ == '__main__':
    run()
