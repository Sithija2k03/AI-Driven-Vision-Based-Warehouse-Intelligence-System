"""
F4 Warehouse Model (v2) — turns warehouse_layout.py into
  1. a SLOT INDEX : (section, slot) -> rack, column, level, pick stop
  2. a GRAPH      : stand-points (nodes) joined by walkway segments (edges)

Used by build_warehouse_graph.py, location_mapper.py and validate_layout.py.
No external dependencies.
"""

import math
from collections import defaultdict

import warehouse_layout as L


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _project(pt, a, b):
    """Closest point to pt on segment a-b, and its distance."""
    ax, ay = a; bx, by = b; px, py = pt
    dx, dy = bx - ax, by - ay
    seg2 = dx * dx + dy * dy
    t = 0.0 if seg2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg2))
    q = (ax + t * dx, ay + t * dy)
    return q, math.hypot(px - q[0], py - q[1])


def project_on_walkway(pt, walkway_id):
    _, pts = L.WALKWAYS[walkway_id]
    best = None
    for a, b in zip(pts, pts[1:]):
        q, d = _project(pt, a, b)
        if best is None or d < best[1]:
            best = (q, d)
    return (round(best[0][0], 1), round(best[0][1], 1))


def _arc_pos(pt, pts):
    """Distance along a polyline to point pt (pt assumed on it)."""
    acc = 0.0
    for a, b in zip(pts, pts[1:]):
        q, d = _project(pt, a, b)
        if d < 0.6:
            return acc + math.hypot(q[0] - a[0], q[1] - a[1])
        acc += math.hypot(b[0] - a[0], b[1] - a[1])
    return None


def px_to_m(d):
    return round(d * L.SCALE_M_PER_PX, 2)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Slot index
# ─────────────────────────────────────────────────────────────────────────────

def _column_level(pattern, local_n, level_offset):
    """local_n is 1-based position inside the rack face."""
    if pattern == 'col2':
        return math.ceil(local_n / 2), (1 if local_n % 2 == 1 else 2) + level_offset
    if pattern == 'row3':
        # caller passes the real slot number for row3
        if local_n <= 3:
            return local_n, 1
        return ((local_n - 4) % 3) + 1, 2 + (local_n - 4) // 3   # T-7/U-7.. stack above
    if pattern == 'stack':
        return 1, local_n + level_offset
    if pattern == 'row1':
        return local_n, 1 + level_offset
    if pattern == 'single':
        return 1, 1 + level_offset
    raise ValueError(pattern)


def _n_cols(rack):
    first, last = rack['slots']
    n = last - first + 1
    p = rack['pattern']
    if rack.get('n_cols'):
        return rack['n_cols']
    if p == 'col2':
        return math.ceil(n / 2)
    if p == 'row3':
        return 3
    if p == 'row1':
        return n
    return 1


def build_slot_index():
    index = {}
    for rack in L.RACKS:
        first, last = rack['slots']
        ncols = _n_cols(rack)
        (sx, sy), (ex, ey) = rack['line']
        for slot in range(first, last + 1):
            local = slot if rack['pattern'] == 'row3' else slot - first + 1
            col, level = _column_level(rack['pattern'], local, rack.get('levels', 0))
            f = (col - 0.5) / ncols
            sxy = (round(sx + f * (ex - sx), 1), round(sy + f * (ey - sy), 1))
            access = [rack['walkway']] + rack.get('extra_access', [])
            stops = [(w, project_on_walkway(sxy, w)) for w in access]
            key = (rack['section'], slot)
            if key in index:
                raise ValueError(f'Slot defined twice: {key}')
            index[key] = {
                'section': rack['section'], 'slot': slot,
                'code': f"{rack['section']}-{slot:02d}",
                'rack_id': rack['id'], 'pattern': rack['pattern'],
                'floor': rack['floor'], 'column': col, 'level': level,
                'n_cols': ncols, 'walkway': rack['walkway'],
                'slot_xy': sxy, 'stops': stops,
            }
    return index


# ─────────────────────────────────────────────────────────────────────────────
# 2. Graph
# ─────────────────────────────────────────────────────────────────────────────

def build_graph(slot_index):
    """
    Nodes: every point on a walkway where something happens
           (walkway ends, junctions, pick stops, stair feet/tops,
           depot, 2nd-floor door). Stops within 0.6 px are merged — that is
           how slots on opposite rack faces share a stand-point.
    Edges: consecutive points along each walkway (weight = metres),
           plus one stair edge per staircase.
    """
    # points per walkway: list of (xy, tag)
    pts_on = defaultdict(set)

    def add(w, xy):
        pts_on[w].add((round(xy[0], 1), round(xy[1], 1)))

    for w, (_, pts) in L.WALKWAYS.items():
        for p in pts:
            add(w, p)
    # junctions: an endpoint of one walkway that lies on another (same floor)
    for w1, (f1, p1) in L.WALKWAYS.items():
        for end in (p1[0], p1[-1]):
            for w2, (f2, p2) in L.WALKWAYS.items():
                if w2 != w1 and f1 == f2 and _arc_pos(end, p2) is not None:
                    add(w2, end)
    # stops
    for s in slot_index.values():
        for w, xy in s['stops']:
            add(w, xy)
    # stairs / depot / 2nd-floor door
    for st in L.STAIRS.values():
        for w, (f, p) in L.WALKWAYS.items():
            for key in ('foot', 'top'):
                want = 'G' if key == 'foot' else 'M'
                if f == want and _arc_pos(st[key], p) is not None:
                    add(w, st[key])
    for w, (f, p) in L.WALKWAYS.items():
        for xy in (L.DEPOT_XY, L.DOOR_2F_XY):
            if f == 'G' and _arc_pos(xy, p) is not None:
                add(w, xy)

    # name nodes
    node_id = {}                     # (floor, xy) -> id
    nodes = {}
    special = {('G', tuple(map(float, L.DEPOT_XY))): 'DEPOT',
               ('G', tuple(map(float, L.DOOR_2F_XY))): 'DOOR_2F'}
    for name, st in L.STAIRS.items():
        special[('G', tuple(map(float, st['foot'])))] = f'{name}_FOOT'
        special[('M', tuple(map(float, st['top'])))] = f'{name}_TOP'

    edges = []
    for w, (floor, wpts) in L.WALKWAYS.items():
        ordered = sorted(pts_on[w], key=lambda xy: _arc_pos(xy, wpts))
        k = 0
        prev = None
        for xy in ordered:
            key = (floor, (float(xy[0]), float(xy[1])))
            if key not in node_id:
                if key in special:
                    nid = special[key]
                else:
                    k += 1
                    nid = f'{w}-{k:02d}'
                    while nid in nodes:
                        k += 1
                        nid = f'{w}-{k:02d}'
                node_id[key] = nid
                nodes[nid] = {'node_id': nid, 'floor': floor, 'x': xy[0], 'y': xy[1],
                              'walkways': [w], 'pick_slots': []}
            else:
                if w not in nodes[node_id[key]]['walkways']:
                    nodes[node_id[key]]['walkways'].append(w)
            cur = node_id[key]
            if prev is not None and prev != cur:
                d = _arc_pos(xy, wpts) - _arc_pos(prev_xy, wpts)
                edges.append({'from': prev, 'to': cur, 'type': 'walk' if floor == 'G' else 'mezz_walk',
                              'walkway': w, 'weight': max(0.1, px_to_m(d))})
            prev, prev_xy = cur, xy

    for name, st in L.STAIRS.items():
        edges.append({'from': f'{name}_FOOT', 'to': f'{name}_TOP', 'type': 'stair',
                      'staircase': name, 'weight': L.STAIR_PENALTY_M})

    # attach slots to stand-points
    for s in slot_index.values():
        s['pick_nodes'] = []
        for w, xy in s['stops']:
            nid = node_id[(s['floor'], (float(xy[0]), float(xy[1])))]
            s['pick_nodes'].append(nid)
            nodes[nid]['pick_slots'].append(s['code'])

    return nodes, edges


def adjacency(edges):
    adj = defaultdict(list)
    for e in edges:
        adj[e['from']].append((e['to'], e['weight']))
        adj[e['to']].append((e['from'], e['weight']))
    return adj


def shortest_paths(adj, src):
    import heapq
    dist = {src: 0.0}
    pq = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, 1e18):
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist.get(v, 1e18):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist


def parse_location(raw):
    """'B-07' -> ('B', 7); 'X03' -> ('X', 3); returns (section, slot|None)."""
    raw = str(raw or '').strip()
    if raw in L.ALIASES:
        return L.ALIASES[raw]
    if not raw or raw.lower() == 'none':
        return ('UNKNOWN', None)
    if '-' not in raw:
        return (raw, None)
    sec, rest = raw.split('-', 1)
    digits = ''.join(ch for ch in rest.split('/')[0] if ch.isdigit())
    return (sec.strip(), int(digits) if digits else None)
