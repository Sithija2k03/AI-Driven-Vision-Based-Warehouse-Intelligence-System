"""
F4 — Classic picker-routing heuristics adapted to the real warehouse
====================================================================
S-shape, Return, Midpoint, Largest gap and Combined were written for a
rectangular warehouse: parallel aisles with a cross aisle at the front and
back. The Fine Enterprises building is not like that, so each rule is applied
to AISLE SEGMENTS of the real walkway graph:

  aisle segment = a stretch of one walkway between two places where you can
                  turn (a junction with another walkway, a staircase, the depot)
                  or between a junction and a dead end.
  through-aisle = both ends are junctions (you can walk out the other side)
  dead-end      = one end is a wall (you must come back the way you came)
  front end     = the end closer to the depot

Segments are visited in a fixed sweep order: top block left to right
(W1, W2, W3, NX, upper W4), bottom block right to left (lower W4, W7, W6, W5),
then the cross aisles and bottom corridors, then the mezzanine walkways.
Walking between two points always uses the shortest path on the graph, so
staircases are chosen automatically.

  S-SHAPE      through-aisle: enter at the nearer end, walk all the way through
               and leave by the other end. Dead-end: in and back out.
  RETURN       every aisle is entered from its front end; walk to the farthest
               pick and come back out the same end.
  MIDPOINT     picks in the front half of a through-aisle are taken from the
               front end, picks in the back half from the back end. Block by
               block (top block, bottom block, mezzanine), a first pass does the
               front halves along the sweep and a second pass does the back
               halves on the way back.
  LARGEST GAP  like midpoint, but the split is at the largest empty stretch in
               the aisle, so that stretch is never walked.
  COMBINED     aisle by aisle, choose "walk through" or "in and back", whichever
               is shorter including the walk to the next aisle (a greedy version
               of Roodbergen & De Koster's combined heuristic).

Use: from classic_heuristics import ClassicRouter
     CR = ClassicRouter('warehouse_graph.json')
     CR.route(stops, 'sshape')  -> dict(waypoints, path, dist_m, stairs, ...)
"""

import heapq
import json
import math
from collections import defaultdict

import warehouse_layout as L

SWEEP = ['W1', 'W2', 'W3', 'NX', 'W4', 'W7', 'W6', 'W5', 'MX', 'LC', 'BX1', 'BX2',
         'MZW', 'MZX5W', 'MZX5', 'MZS', 'MZV', 'MZ1', 'MZV4', 'MZH', 'MZTU', 'MZ2', 'MZZ']
METHODS = ['sshape', 'return', 'midpoint', 'largest_gap', 'combined']


def _arc(pt, pts):
    acc = 0.0
    for a, b in zip(pts, pts[1:]):
        ax, ay = a; bx, by = b; px, py = pt
        dx, dy = bx - ax, by - ay
        s2 = dx * dx + dy * dy
        t = 0 if s2 == 0 else max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / s2))
        q = (ax + t * dx, ay + t * dy)
        if math.hypot(px - q[0], py - q[1]) < 0.6:
            return acc + math.hypot(q[0] - ax, q[1] - ay)
        acc += math.hypot(bx - ax, by - ay)
    return None


class ClassicRouter:
    def __init__(self, graph_path='warehouse_graph.json'):
        g = json.load(open(graph_path, encoding='utf-8'))
        self.nodes, self.edges = g['nodes'], g['edges']
        self.adj = defaultdict(list)
        for e in self.edges:
            st = 1 if e['type'] == 'stair' else 0
            self.adj[e['from']].append((e['to'], e['weight'], st))
            self.adj[e['to']].append((e['from'], e['weight'], st))
        self._sp = {}
        self._build_segments()

    # ── shortest paths on the full graph ───────────────────────────────────
    def sp(self, src):
        if src not in self._sp:
            dist, prev = {src: 0.0}, {}
            pq = [(0.0, src)]
            while pq:
                d, u = heapq.heappop(pq)
                if d > dist[u] + 1e-9:
                    continue
                for v, w, _ in self.adj[u]:
                    if d + w < dist.get(v, 1e18) - 1e-9:
                        dist[v], prev[v] = d + w, u
                        heapq.heappush(pq, (d + w, v))
            self._sp[src] = (dist, prev)
        return self._sp[src]

    def d(self, a, b):
        return self.sp(a)[0][b]

    def path(self, a, b):
        _, prev = self.sp(a)
        out = [b]
        while out[-1] != a:
            out.append(prev[out[-1]])
        return out[::-1]

    # ── aisle segments ─────────────────────────────────────────────────────
    def _is_open(self, nid, w):
        n = self.nodes[nid]
        return len(n['walkways']) > 1 or nid == 'DEPOT' or nid.endswith(('_FOOT', '_TOP'))

    def _build_segments(self):
        self.segments = []            # dicts: walkway, nodes(ordered), a, b, a_open, b_open, pos{node: m}
        self.seg_of = {}              # node -> list of segment indexes
        for w in SWEEP:
            if w not in L.WALKWAYS:
                continue
            _, pts = L.WALKWAYS[w]
            on = [nid for nid, n in self.nodes.items() if w in n['walkways']]
            on.sort(key=lambda nid: _arc((self.nodes[nid]['x'], self.nodes[nid]['y']), pts))
            cuts = [i for i, nid in enumerate(on) if self._is_open(nid, w)]
            bounds = sorted(set([0] + cuts + [len(on) - 1]))
            for i, j in zip(bounds, bounds[1:]):
                seg_nodes = on[i:j + 1]
                pos, acc = {seg_nodes[0]: 0.0}, 0.0
                for u, v in zip(seg_nodes, seg_nodes[1:]):
                    acc += self.d(u, v)
                    pos[v] = acc
                k = len(self.segments)
                self.segments.append({'walkway': w, 'nodes': seg_nodes, 'a': seg_nodes[0], 'b': seg_nodes[-1],
                                      'a_open': self._is_open(seg_nodes[0], w),
                                      'b_open': self._is_open(seg_nodes[-1], w), 'pos': pos, 'len': acc})
                for nid in seg_nodes:
                    self.seg_of.setdefault(nid, []).append(k)

    def _assign(self, stops):
        """Pick one stand-point per stop and put it in one aisle segment.
        A stop with several access points (X-3, X-05) uses one in an aisle that
        already has picks, otherwise the one nearest the depot."""
        single = {s['alts'][0] for s in stops if len(s['alts']) == 1}
        used_w = {w for nid in single for w in self.nodes[nid]['walkways']}
        chosen = []
        for s in stops:
            alts = s['alts']
            if len(alts) > 1:
                pref = [a for a in alts if set(self.nodes[a]['walkways']) & used_w]
                alts = pref or alts
                alts = sorted(alts, key=lambda a: self.d('DEPOT', a))
            chosen.append(alts[0])
        by_seg = defaultdict(list)
        for nid in chosen:
            segs = self.seg_of[nid]
            interior = [k for k in segs if nid not in (self.segments[k]['a'], self.segments[k]['b'])]
            k = (interior or sorted(segs))[0]
            by_seg[k].append(nid)
        return dict(sorted(by_seg.items())), chosen

    # ── building blocks ────────────────────────────────────────────────────
    def _front_back(self, seg):
        a, b = seg['a'], seg['b']
        if not seg['b_open']:
            return a, b
        if not seg['a_open']:
            return b, a
        return (a, b) if self.d('DEPOT', a) <= self.d('DEPOT', b) else (b, a)

    def _order_from(self, seg, picks, end):
        p = seg['pos']
        return sorted(set(picks), key=lambda n: abs(p[n] - p[end]))

    def _through(self, seg):
        return seg['a_open'] and seg['b_open']

    # ── policies: each returns the list of waypoints after DEPOT ───────────
    def _sshape(self, segs, cur='DEPOT'):
        wp = []
        for k, picks in segs.items():
            s = self.segments[k]
            if self._through(s):
                enter = s['a'] if self.d(cur, s['a']) <= self.d(cur, s['b']) else s['b']
                leave = s['b'] if enter == s['a'] else s['a']
                wp += [enter] + self._order_from(s, picks, enter) + [leave]
                cur = leave
            else:
                f, _ = self._front_back(s)
                wp += [f] + self._order_from(s, picks, f) + [f]
                cur = f
        return wp

    def _return(self, segs):
        wp = []
        for k, picks in segs.items():
            s = self.segments[k]
            f, _ = self._front_back(s)
            wp += [f] + self._order_from(s, picks, f) + [f]
        return wp

    def _split(self, segs, how):
        first, second = [], []
        for k, picks in segs.items():
            s = self.segments[k]
            f, b = self._front_back(s)
            blk = self.block(k)
            if not self._through(s):
                first.append((f, self._order_from(s, picks, f), blk))
                continue
            p = s['pos']
            depth = {n: abs(p[n] - p[f]) for n in picks}
            if how == 'midpoint':
                cut = s['len'] / 2
                front = [n for n in picks if depth[n] <= cut]
            else:   # largest gap: never walk the biggest empty stretch
                pts = sorted(set([0.0] + list(depth.values()) + [s['len']]))
                gaps = [(pts[i + 1] - pts[i], i) for i in range(len(pts) - 1)]
                g, i = max(gaps)
                cut = pts[i]
                front = [n for n in picks if depth[n] <= cut + 1e-9]
            back = [n for n in picks if n not in front]
            if front:
                first.append((f, self._order_from(s, front, f), blk))
            if back:
                second.append((b, self._order_from(s, back, b), blk))
        # two passes per block (the classic multi-block rule): front halves
        # along the sweep, then back halves on the way back, block by block
        wp = []
        for blk in ('top', 'bottom', 'mezz'):
            for end, picks, _ in [x for x in first if x[2] == blk]:
                wp += [end] + picks + [end]
            for end, picks, _ in reversed([x for x in second if x[2] == blk]):
                wp += [end] + picks + [end]
        return wp

    def block(self, k):
        """top block (north of the main cross aisle), bottom block, or mezzanine."""
        s = self.segments[k]
        if self.nodes[s['a']]['floor'] == 'M':
            return 'mezz'
        ys = [self.nodes[n]['y'] for n in s['nodes']]
        return 'top' if sum(ys) / len(ys) < 362 else 'bottom'

    def _combined(self, segs):
        wp, cur = [], 'DEPOT'
        items = list(segs.items())
        for idx, (k, picks) in enumerate(items):
            s = self.segments[k]
            nxt = items[idx + 1][0] if idx + 1 < len(items) else None

            def to_next(pt):
                if nxt is None:
                    return self.d(pt, 'DEPOT')
                t = self.segments[nxt]
                return min(self.d(pt, t['a']), self.d(pt, t['b']))

            enter = s['a'] if self.d(cur, s['a']) <= self.d(cur, s['b']) else s['b']
            if not self._through(s):
                enter, _ = self._front_back(s)
            order = self._order_from(s, picks, enter)
            ret_wp = [enter] + order + [enter]
            ret_cost = self._cost(ret_wp, cur) + to_next(enter)
            if self._through(s):
                leave = s['b'] if enter == s['a'] else s['a']
                thr_wp = [enter] + order + [leave]
                thr_cost = self._cost(thr_wp, cur) + to_next(leave)
                if thr_cost < ret_cost:
                    wp += thr_wp
                    cur = leave
                    continue
            wp += ret_wp
            cur = enter
        return wp

    def _cost(self, wp, start='DEPOT'):
        seq = [start] + wp
        return sum(self.d(a, b) for a, b in zip(seq, seq[1:]))

    # ── public ─────────────────────────────────────────────────────────────
    def expand(self, waypoints):
        """DEPOT → waypoints → DEPOT as a full node path, with metrics."""
        seq = ['DEPOT'] + waypoints + ['DEPOT']
        full = ['DEPOT']
        for a, b in zip(seq, seq[1:]):
            if a != b:
                full += self.path(a, b)[1:]
        dist = stairs = 0
        walked = defaultdict(int)
        for u, v in zip(full, full[1:]):
            w, st = next((w, st) for x, w, st in self.adj[u] if x == v)
            dist += w
            stairs += st
            walked[tuple(sorted((u, v)))] += 1
        repeat = 0.0
        for (u, v), c in walked.items():
            if c > 1:
                w = next(w for x, w, _ in self.adj[u] if x == v)
                repeat += w * (c - 1)
        return {'path': full, 'dist_m': round(dist, 2), 'stairs': stairs, 'rewalked_m': round(repeat, 2)}

    def route(self, stops, method):
        segs, _ = self._assign(stops)
        fn = {'sshape': self._sshape, 'return': self._return, 'combined': self._combined,
              'midpoint': lambda s: self._split(s, 'midpoint'),
              'largest_gap': lambda s: self._split(s, 'gap')}[method]
        wp = fn(segs)
        out = self.expand(wp)
        out['waypoints'] = wp
        out['segments_entered'] = len(segs)
        return out
