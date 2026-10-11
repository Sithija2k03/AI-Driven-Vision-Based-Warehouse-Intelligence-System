"""
F4 Location Mapper (v3) — Fine Enterprises WIS
==============================================
Builds the item → location master and attaches every item to:
  * its rack, column and level (from the confirmed numbering patterns)
  * pick_node: the stand-point in the walkway in front of that column
  * access_nodes: all stand-points it can be picked from (e.g. X-3 island)

Sources
  1. Stock_Report.xlsx (any sheet name)                         (MASTER)
     Code | Item Name | MAIN STORE (qty) | MS L (location)
     Lists every item code, including zero-stock and deactivated ones.
  2. ACCESSORIES-STOCK_LOCATIONS.xlsx                            (FALLBACK)
     Used only when a Stock Report row is broken: item names containing
     an inch mark (") shift the columns, so the location is lost and the
     quantity lands in the MS L column.

Zero stock does NOT make an item unroutable: it still has a slot and was
ordered during the month. Only the location decides routability.

Outputs
  item_locations_real.json   one record per item code
  location_conflicts.csv     items whose location differs between the two
                             sources (Stock Report is used) — check these
Run:   python location_mapper.py
Needs: openpyxl
"""

import csv
import json
import re
import sys
from collections import Counter

import openpyxl

import warehouse_layout as L
import warehouse_model as WM

STOCK_REPORT = 'Stock_Report.xlsx'          # Code | Item Name | MAIN STORE | MS L
STOCK_SHEET = 'Stock Report'
FALLBACK = 'ACCESSORIES-STOCK_LOCATIONS.xlsx'


def find_sheet(path, preferred, required_headers):
    """Open the sheet named `preferred` (ignoring case/spaces); if it is not
    there, use the first sheet whose header row contains required_headers."""
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    want = re.sub(r'\s+', '', preferred).lower()
    for ws in wb.worksheets:
        if re.sub(r'\s+', '', ws.title).lower() == want:
            return ws
    for ws in wb.worksheets:
        first = next(ws.iter_rows(max_row=1, values_only=True), ())
        head = {re.sub(r'\s+', ' ', str(h or '')).strip().upper() for h in first}
        if all(h.upper() in head for h in required_headers):
            print(f"  Sheet '{preferred}' not found in {path}; using '{ws.title}' (matching columns)")
            return ws
    raise SystemExit(
        f"\nERROR: {path} has no sheet named '{preferred}' and no sheet with columns "
        f"{required_headers}.\nSheets in this file: {wb.sheetnames}\n"
        f"Check that you copied the right workbook into this folder.")


def clean_loc(v):
    """'A -12' -> 'A-12'; None/'None' -> ''."""
    if v is None:
        return ''
    s = re.sub(r'\s+', '', str(v)).upper()
    return '' if s in ('NONE', '') else s


def to_int(v):
    try:
        return int(float(str(v).replace(',', '')))
    except (TypeError, ValueError):
        return 0


def read_fallback(path):
    """ACCESSORIES-STOCK_LOCATIONS.xlsx: group | code | name | qty | location."""
    out = {}
    try:
        ws = openpyxl.load_workbook(path, read_only=True, data_only=True).active
    except FileNotFoundError:
        print(f'  (fallback file {path} not found — broken rows stay unlocated)')
        return out
    for row in ws.iter_rows(min_row=2, values_only=True):
        _, code, name, qty, loc = (list(row) + [None] * 5)[:5]
        if code:
            out[str(code).strip()] = {'name': str(name or '').strip(), 'qty': to_int(qty),
                                      'loc': clean_loc(loc)}
    return out


def read_stock_report(path, sheet):
    ws = find_sheet(path, sheet, ['CODE', 'ITEM NAME', 'MAIN STORE', 'MS L'])
    rows = ws.iter_rows(values_only=True)
    head = [re.sub(r'\s+', ' ', str(h or '')).strip().upper() for h in next(rows)]
    ci = {k: head.index(k) for k in ('CODE', 'ITEM NAME', 'MAIN STORE', 'MS L')}
    out = {}
    for r in rows:
        r = list(r) + [None] * 4
        code = r[ci['CODE']]
        if not code:
            continue
        name, qty, loc = r[ci['ITEM NAME']], r[ci['MAIN STORE']], r[ci['MS L']]
        broken = isinstance(qty, str) and not str(qty).strip().replace(',', '').isdigit() \
            and isinstance(loc, (int, float))
        if broken:   # name was split at an inch mark: qty sits in MS L, location lost
            out[str(code).strip()] = {'name': f'{name}{qty}', 'qty': to_int(loc), 'loc': '',
                                      'broken_row': True}
        else:
            out[str(code).strip()] = {'name': str(name or '').strip(), 'qty': to_int(qty),
                                      'loc': clean_loc(loc), 'broken_row': False}
    return out


def run(stock_path=STOCK_REPORT, fallback_path=FALLBACK, output_path='item_locations_real.json'):
    slots = WM.build_slot_index()
    WM.build_graph(slots)                       # fills slot['pick_nodes']

    master = read_stock_report(stock_path, STOCK_SHEET)
    fb = read_fallback(fallback_path)

    items, reasons, conflicts = [], Counter(), []
    src_count = Counter()
    for code, m in master.items():
        loc, src = m['loc'], 'stock_report'
        f = fb.get(code)
        if m['broken_row'] or not loc or loc.isdigit():
            if f and f['loc']:
                loc, src = f['loc'], 'fallback (broken stock report row)' if m['broken_row'] else 'fallback'
        if src == 'stock_report' and f and f['loc'] and m['loc'] and f['loc'] != m['loc']:
            conflicts.append({'code': code, 'item': m['name'], 'stock_report': m['loc'],
                              'accessories_file': f['loc']})
        src_count[src.split(' ')[0]] += 1

        section, slot = WM.parse_location(loc)
        if loc.isdigit():
            section, slot = ('INVALID', None)
        item = {'sku': code, 'item_name': m['name'], 'qty': m['qty'], 'in_stock': m['qty'] > 0,
                'deactivated': 'DEACTIVATED' in m['name'].upper(),
                'original_loc': loc, 'location_source': src, 'section': section, 'slot': slot}

        s = slots.get((section, slot))
        if section in L.EXCLUDED_SECTIONS:
            why = f'excluded section {section}'
        elif loc in L.IGNORED_LOCATIONS:
            why = f'ignored location {loc}'
        elif section in ('UNKNOWN',):
            why = 'blank location'
        elif s is None:
            why = f'not on map: {loc}'
        else:
            why = None

        if why:
            item.update(routable=False, reason=why, pick_node=None, access_nodes=[])
            reasons['not on map' if why.startswith('not on map') else why] += 1
        else:
            item.update(routable=True, slot_code=s['code'], rack_id=s['rack_id'],
                        column=s['column'], level=s['level'], floor=s['floor'],
                        is_mezzanine=s['floor'] == 'M',
                        pick_node=s['pick_nodes'][0], access_nodes=s['pick_nodes'],
                        x=s['slot_xy'][0], y=s['slot_xy'][1])
        items.append(item)

    routable = [i for i in items if i['routable']]
    out = {
        'metadata': {
            'source_file': stock_path, 'fallback_file': fallback_path,
            'version': 'v3 stock-report master, walkway-based',
            'total_items': len(items),
            'items_in_stock': sum(1 for i in items if i['in_stock']),
            'routable_items': len(routable),
            'routable_in_stock': sum(1 for i in routable if i['in_stock']),
            'unroutable_items': len(items) - len(routable),
            'mezzanine_items': sum(1 for i in routable if i['is_mezzanine']),
            'unique_pick_nodes': len({i['pick_node'] for i in routable}),
            'location_sources': dict(src_count),
            'location_conflicts': len(conflicts),
            'unroutable_reasons': dict(reasons),
        },
        'items': items,
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    with open('location_conflicts.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['code', 'item', 'stock_report', 'accessories_file'])
        w.writeheader()
        w.writerows(conflicts)

    m = out['metadata']
    print('=' * 60)
    print(f'  Output:              {output_path}')
    print(f'  Item codes:          {m["total_items"]:,}  (in stock {m["items_in_stock"]:,})')
    print(f'  Routable:            {m["routable_items"]:,}  (in stock {m["routable_in_stock"]:,})')
    print(f'  Mezzanine routable:  {m["mezzanine_items"]:,}')
    print(f'  Unique pick nodes:   {m["unique_pick_nodes"]}')
    print(f'  Location source:     {dict(src_count)}')
    print(f'  Conflicts written:   location_conflicts.csv ({len(conflicts)})')
    print('=' * 60)
    print('\nUnroutable reasons:')
    for k, v in reasons.most_common():
        print(f'  {v:5d}  {k}')
    odd = Counter(i['original_loc'] for i in items if str(i.get('reason', '')).startswith('not on map'))
    if odd:
        print('\nLocations not on the map:', dict(odd.most_common(15)))
    return out


if __name__ == '__main__':
    run(*sys.argv[1:])
