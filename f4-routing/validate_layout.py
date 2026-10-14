"""
F4 Layout Validation — Fine Enterprises WIS
===========================================
Run after build_warehouse_graph.py and location_mapper.py.

Produces:
  * automated checks printed to the console
  * spot_check.csv   — 15 random SKUs with neighbours to verify on the floor
  * layout_check.html — top view + front view of every rack face

Run: python validate_layout.py
"""

import csv
import json
import random
from collections import defaultdict

import warehouse_layout as L
import warehouse_model as WM

# Rack outlines from warehouse_editor_v3 (for the top-view drawing only)
OUTLINES = [
    ('B/C', 65, 0, 80, 340), ('A', 195, 0, 40, 340), ('D', 235, 0, 45, 340),
    ('E', 325, 0, 35, 340), ('R', 370, 0, 40, 185), ('Q', 370, 190, 40, 70),
    ('P', 370, 265, 40, 70), ('X-3 / X-5', 470, 0, 150, 335),
    ('O13-17', 665, 0, 50, 40), ('O', 720, 0, 40, 170), ('N', 720, 175, 40, 115),
    ('M', 720, 295, 40, 140), ('L', 720, 440, 40, 110),
    ('G25-30', 65, 385, 50, 175), ('G19-24', 185, 410, 35, 140),
    ('G13-18', 222, 410, 28, 140), ('G1-12', 320, 385, 50, 175),
    ('T', 375, 385, 40, 90), ('U', 375, 480, 40, 80), ('X-2/4', 510, 385, 45, 175),
    ('X6-9', 560, 385, 50, 175), ('X-1', 100, 585, 645, 110), ('I', 65, 590, 30, 110),
    ('J', 65, 730, 345, 30), ('K', 415, 730, 350, 30), ('W (mezz)', 810, 0, 50, 600),
]


def face_key(s):
    """Racks sharing the same line form one physical face (e.g. B with C above)."""
    rack = next(r for r in L.RACKS if r['id'] == s['rack_id'])
    return (rack['floor'], rack['walkway'], rack['line'])


def main():
    graph = json.load(open('warehouse_graph.json', encoding='utf-8'))
    items = json.load(open('item_locations_real.json', encoding='utf-8'))['items']
    slots = graph['slots']
    nodes, edges = graph['nodes'], graph['edges']

    # ── Automated checks ────────────────────────────────────────────────────
    checks = []
    adj = WM.adjacency(edges)
    dist = WM.shortest_paths(adj, L.ROUTE_START)
    unreach = [n for n in nodes if n not in dist]
    checks.append(('Every stand-point reachable from the DEPOT', not unreach,
                   f'{len(nodes) - len(unreach)}/{len(nodes)} reachable'))
    pick_dist = {n: d for n, d in dist.items() if nodes[n]['pick_slots']}
    far = max(pick_dist, key=pick_dist.get)
    checks.append(('Depot round trip to the farthest pick spot (info only)', True,
                   f'DEPOT → {far} → DEPOT = {2 * pick_dist[far]:.1f} m '
                   f'({", ".join(nodes[far]["pick_slots"][:4])})'))
    routable = [i for i in items if i['routable']]
    bad = [i for i in routable if i['pick_node'] not in nodes]
    checks.append(('Every routable SKU has a stand-point in the graph', not bad,
                   f'{len(routable)} routable SKUs'))
    used = defaultdict(set)
    for i in items:
        if i['section'] not in L.EXCLUDED_SECTIONS and i['slot']:
            used[i['section']].add(i['slot'])
    missing = sorted(f'{sec}-{n:02d}' for sec, ns in used.items() for n in ns
                     if f'{sec}-{n:02d}' not in slots and sec != 'UNKNOWN'
                     and f'{sec}-{n:02d}' not in L.IGNORED_LOCATIONS)
    ignored = ', '.join(f'{k} ({v})' for k, v in L.IGNORED_LOCATIONS.items())
    checks.append(('Every slot number in the stock file fits a rack', not missing,
                   (', '.join(missing) or 'all fit') + (f' · ignored: {ignored}' if ignored else '')))
    empty = sorted(code for code in slots if (code.split('-')[0], int(code.split('-')[1]))
                   not in {(i['section'], i['slot']) for i in routable})
    checks.append(('Modelled slots with no stock (info only)', True,
                   f'{len(empty)}: ' + (', '.join(empty) if empty else 'none')))
    walk_edges = [e for e in edges if e['type'] == 'walk']
    longest = max(walk_edges, key=lambda e: e['weight'])
    checks.append(('Longest walk with no pick stop on the way (info only)', True,
                   f'{longest["weight"]:.1f} m on {longest["walkway"]} ({longest["from"]} to {longest["to"]})'))

    print('\nAUTOMATED CHECKS')
    for name, ok, info in checks:
        print(f'  [{"PASS" if ok else "FAIL"}] {name} — {info}')

    # ── Neighbour lookup ────────────────────────────────────────────────────
    by_face = defaultdict(dict)
    for code, s in slots.items():
        by_face[face_key(s)][(s['column'], s['level'])] = code

    def neighbours(code):
        s = slots[code]
        f = by_face[face_key(s)]
        c, lv = s['column'], s['level']
        me = nodes[s['pick_nodes'][0]]
        opp = []
        for nid, nd in nodes.items():
            if nd['floor'] != me['floor'] or s['walkway'] not in nd['walkways']:
                continue
            if ((nd['x'] - me['x']) ** 2 + (nd['y'] - me['y']) ** 2) ** .5 * L.SCALE_M_PER_PX > 0.8:
                continue
            opp += [x for x in nd['pick_slots'] if face_key(slots[x]) != face_key(s)]
        opp = sorted(set(opp), key=lambda c: (slots[c]['section'], slots[c]['slot']))
        return {
            'above': f.get((c, lv + 1), '-'), 'below': f.get((c, lv - 1), '-'),
            'next_col_same_level': f.get((c + 1, lv), '-'),
            'prev_col_same_level': f.get((c - 1, lv), '-'),
            'across_walkway': ', '.join(opp[:6]) or '-',
        }

    # ── Spot check list ─────────────────────────────────────────────────────
    rng = random.Random(42)
    by_sec = defaultdict(list)
    for i in routable:
        by_sec[i['section']].append(i)
    secs = sorted(by_sec)
    rng.shuffle(secs)
    picks = [rng.choice(by_sec[s]) for s in secs[:15]]
    rows = []
    for i in picks:
        n = neighbours(i['slot_code'])
        rows.append({'sku': i['sku'], 'item': i['item_name'][:40], 'location': i['slot_code'],
                     'column': i['column'], 'level': i['level'], 'floor': i['floor'],
                     'stand_point': i['pick_node'], **n, 'matches_floor (Y/N)': ''})
    import os
    if os.path.exists('spot_check.csv'):
        # keep the floor-verified file; show its rows on the page instead
        rows = list(csv.DictReader(open('spot_check.csv', encoding='utf-8')))
        print(f'\nKept existing spot_check.csv ({len(rows)} SKUs, floor-verified)')
    else:
        with open('spot_check.csv', 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f'\nWrote spot_check.csv ({len(rows)} SKUs)')

    # ── Faces for the drawing ───────────────────────────────────────────────
    faces = []
    for key, cells in by_face.items():
        floor, walkway, _ = key
        codes = list(cells.values())
        secs_here = sorted({slots[c]['section'] for c in codes})
        racks = sorted({slots[c]['rack_id'] for c in codes})
        ncol = max(c for c, _ in cells)
        nlev = max(lv for _, lv in cells)
        stand = {c: slots[code]['pick_nodes'][0] for (c, _), code in cells.items()}
        faces.append({
            'name': ' + '.join(racks), 'sections': secs_here, 'floor': floor,
            'walkway': walkway, 'ncol': ncol, 'nlev': nlev,
            'cells': [[c, lv, code, slots[code]['rack_id']] for (c, lv), code in cells.items()],
            'stand': stand,
        })
    order = {w: i for i, w in enumerate(L.WALKWAYS)}
    faces.sort(key=lambda f: (f['floor'] != 'G', order[f['walkway']], f['name']))

    sku_count = defaultdict(int)
    for i in routable:
        sku_count[i['slot_code']] += 1

    data = {
        'checks': [{'name': n, 'ok': ok, 'info': info} for n, ok, info in checks],
        'assumptions': L.ASSUMPTIONS,
        'walkways': {w: {'floor': f, 'pts': p} for w, (f, p) in L.WALKWAYS.items()},
        'stairs': L.STAIRS,
        'nodes': [{'id': k, 'x': v['x'], 'y': v['y'], 'floor': v['floor'],
                   'slots': v['pick_slots']} for k, v in nodes.items()],
        'outlines': OUTLINES, 'faces': faces, 'spot': rows,
        'sku_count': sku_count,
        'stats': {'nodes': len(nodes), 'edges': len(edges), 'slots': len(slots),
                  'routable': len(routable), 'total': len(items)},
    }
    tpl = open('layout_check_template.html', encoding='utf-8').read()
    html = tpl.replace('/*__DATA__*/null', json.dumps(data, ensure_ascii=False))
    open('layout_check.html', 'w', encoding='utf-8').write(html)
    print('Wrote layout_check.html')


if __name__ == '__main__':
    main()
