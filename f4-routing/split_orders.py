"""
F4 Step 3 — Train / validation / test split by date
===================================================
Real orders only (June–October 2026). Splitting by date, not at random,
means the agent is always tested on orders from AFTER the period it learned
from — the way it would be used in the warehouse.

  TRAIN       1 Jun – 31 Aug   the agent learns from these
  VALIDATION  1 – 30 Sep       used while training to pick the best model
                               and tune settings
  TEST        1 – 14 Oct       touched once, for the final results.
                               Closest to the Stock Report date, so item
                               locations are most accurate here.

Synthetic orders are no longer generated: with ~6,000 real training orders
they are not needed (see the One-Month Results Report for why they were
considered and how they were checked).

Input : orders_clean.json
Output: orders_train.json, orders_val.json, orders_test.json
Run   : python split_orders.py
"""

import json
from collections import Counter

TRAIN_TO = '2026-08-31'
VAL_TO = '2026-09-30'


def describe(orders):
    n = len(orders)
    lines = [o['n_lines'] for o in orders]
    stops = [len({l['pick_node'] for l in o['lines']}) for o in orders]
    floors = Counter('+'.join(sorted({l['floor'] for l in o['lines']})) for o in orders)
    return {'orders': n, 'lines': sum(lines),
            'from': min(o['date'] for o in orders), 'to': max(o['date'] for o in orders),
            'mean_lines': round(sum(lines) / n, 2), 'mean_stops': round(sum(stops) / n, 2),
            'large_orders_16plus': sum(1 for k in lines if k >= 16),
            'pct_needs_mezzanine': round(100 * (n - floors.get('G', 0)) / n, 1)}


def run():
    orders = json.load(open('orders_clean.json', encoding='utf-8'))['orders']
    sets = {
        'train': [o for o in orders if o['date'] <= TRAIN_TO],
        'val': [o for o in orders if TRAIN_TO < o['date'] <= VAL_TO],
        'test': [o for o in orders if o['date'] > VAL_TO],
    }
    print('=' * 84)
    print(f"{'set':6s} {'period':24s} {'orders':>7s} {'lines':>7s} {'lines/ord':>9s} "
          f"{'stops/ord':>9s} {'16+ lines':>9s} {'mezz':>6s}")
    for name, obj in sets.items():
        d = describe(obj)
        json.dump({'meta': {'split': name, **d}, 'orders': obj},
                  open(f'orders_{name}.json', 'w', encoding='utf-8'), ensure_ascii=False)
        print(f"{name:6s} {d['from'] + ' to ' + d['to']:24s} {d['orders']:7,d} {d['lines']:7,d} "
              f"{d['mean_lines']:9.2f} {d['mean_stops']:9.2f} {d['large_orders_16plus']:9,d} "
              f"{d['pct_needs_mezzanine']:5.1f}%")
    print('=' * 84)
    print('Wrote orders_train.json, orders_val.json, orders_test.json')


if __name__ == '__main__':
    run()
