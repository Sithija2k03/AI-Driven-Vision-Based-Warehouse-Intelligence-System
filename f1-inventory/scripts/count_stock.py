r"""
F1 sub-function 1: carton counting per location and stock status.

For each frame: run the detector, assign every detected box to a location (by the
box centre), count boxes per location, and compare each count with that location's
own full-stock reference count. Writes one JSON record per location for F4.

Usage (from the f1-inventory folder):
  python scripts\count_stock.py --weights weights\day2_s1024_best.pt ^
      --config configs\locations_example.json --camera-id X_5 ^
      --frames "D:\path\to\frame1.jpg" "D:\path\to\frame2.jpg" ^
      --out-json outputs\stock_X_5.json --overlay-dir outputs\overlays

Config (see configs\locations_example.json): per camera, one reference frame (a full-stock
frame) and a list of locations. Each location has EITHER a rectangle "region": [x1,y1,x2,y2]
OR a shelf outline "polygon": [[x,y], [x,y], ...], all as fractions of the image (0-1), so it
works at any resolution. Draw shelf polygons with scripts\draw_shelves.py.
A box belongs to the location that contains its centre (or, with --anchor bottom, the point where
it stands). If shapes overlap, the smallest one wins. Boxes on no location are reported separately.
Optionally give "reference_count" to skip counting the reference frame.

Status rule (thresholds are PLACEHOLDERS to tune on real frames):
  ratio = count / reference_count
  UNKNOWN   if the reference shows fewer than --min-ref cartons (default 5): too few to judge
  STOCK-OUT if count == 0 or ratio < --out-ratio (default 0.10)
  LOW       if ratio < --low-ratio (default 0.50)
  IN STOCK  otherwise
The detector misses about half of the cartons, so absolute counts are unreliable; the
ratio against the same location's own reference frame (same detector, same camera)
cancels part of that bias. It does NOT remove it.

Sub-function 2 (optional): add --occlusion-cnn weights\occlusion_r18.pt to estimate each box's occlusion level
(low/medium/high, hybrid CNN + geometry by default) and fill "occlusion_mix", "trust" and "verify_manually"
for every location. trust = mean expected detection weight of the location's boxes (1.0 = all clearly visible,
0.63 = all heavily occluded); verify_manually = trust < --trust-threshold.
"""
import argparse, csv, json, sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = "0.4-draft"


def loc_polygon(loc):
    """Location shape as a list of (x, y) fractions. Accepts "polygon": [[x,y],...] (shelf outlines)
    or "region": [x1,y1,x2,y2] (rectangles)."""
    if "polygon" in loc:
        return [tuple(map(float, p)) for p in loc["polygon"]]
    x1, y1, x2, y2 = loc["region"]
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def polygon_area(poly):
    return abs(sum(poly[i][0] * poly[i - 1][1] - poly[i - 1][0] * poly[i][1] for i in range(len(poly)))) / 2


def point_in_polygon(x, y, poly):
    """Ray casting; points exactly on an edge count as inside."""
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if min(y1, y2) <= y <= max(y1, y2) and min(x1, x2) <= x <= max(x1, x2):
            cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
            if abs(cross) < 1e-12:
                return True
        if (y1 > y) != (y2 > y):
            xin = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xin:
                inside = not inside
    return inside


def anchor_point(b, anchor="center"):
    """Point of a box used to decide its location. 'bottom' = middle of the bottom edge, i.e. where a
    carton stands on the shelf board (useful when shelf levels are stacked)."""
    cx = (b[0] + b[2]) / 2
    if anchor == "bottom":
        return cx, b[3] - 0.1 * (b[3] - b[1])   # a little above the edge so it stays inside the level
    return cx, (b[1] + b[3]) / 2


def locate(px, py, shapes):
    """shapes: list of (location_id, polygon, area), sorted smallest area first.
    If shapes overlap, the smallest (most specific) one wins."""
    for lid, poly, _ in shapes:
        if point_in_polygon(px, py, poly):
            return lid
    return None


def make_shapes(locations):
    shapes = [(l["location_id"], loc_polygon(l), polygon_area(loc_polygon(l))) for l in locations]
    return sorted(shapes, key=lambda s: s[2])


def assign_boxes(boxes, locations, w, h, anchor="center", unassigned=None):
    """boxes: list of (x1,y1,x2,y2,conf) in pixels. Returns {location_id: [box,...]}.
    Boxes that fall on no location are appended to `unassigned` (if a list is given)."""
    out = {loc["location_id"]: [] for loc in locations}
    shapes = make_shapes(locations)
    for b in boxes:
        px, py = anchor_point(b, anchor)
        lid = locate(px / w, py / h, shapes)
        if lid is None:
            if unassigned is not None:
                unassigned.append(b)
        else:
            out[lid].append(b)
    return out


def stock_status(count, reference, low_ratio=0.5, out_ratio=0.1, min_ref=0):
    """Returns (status, ratio). ratio is None when there is no usable reference.
    UNKNOWN when the full-stock reference shows fewer than min_ref cartons: with so few cartons,
    one missed detection swings the ratio by 20-100%, so the status would be noise."""
    if reference is None or reference <= 0:
        return "UNKNOWN", None
    ratio = count / reference
    if reference < min_ref:
        return "UNKNOWN", ratio
    if count == 0 or ratio < out_ratio:
        return "STOCK-OUT", ratio
    if ratio < low_ratio:
        return "LOW", ratio
    return "IN STOCK", ratio


def build_records(camera_id, frame_name, per_loc, references, low_ratio, out_ratio, timestamp, occ=None, slots=None,
                  min_ref=0):
    recs = []
    for loc_id, bx in per_loc.items():
        o = (occ or {}).get(loc_id, {})
        count = len(bx)
        ref = references.get(loc_id)
        status, ratio = stock_status(count, ref, low_ratio, out_ratio, min_ref)
        recs.append({
            "schema_version": SCHEMA_VERSION,
            "location_id": loc_id,
            "f4_slots": (slots or {}).get(loc_id, []),   # F4 layout slot IDs this location covers (for routing)
            "camera_id": camera_id,
            "frame": frame_name,
            "timestamp": timestamp,
            "count": count,
            "reference_count": ref,
            "ratio_to_reference": None if ratio is None else round(ratio, 3),
            "status": status,
            "status_note": (f"reference has only {ref or 0} cartons (< {min_ref}); too few to judge"
                            if status == "UNKNOWN" else None),
            "occlusion_mix": o.get("occlusion_mix"),
            "trust": o.get("trust"),                  # sub-function 2 (None if occlusion not run)
            "verify_manually": o.get("verify_manually"),
        })
    return recs


def pct_unreliable(x, ref):
    """Share of the camera's reference cells that look MORE reliable than x (mid-rank). 1 = worst seen."""
    import numpy as np
    ref = np.asarray(ref, float)
    return float(((ref > x).sum() + 0.5 * (ref == x).sum()) / len(ref))


def load_camera_calibration(path, camera_id):
    """Returns (camera calibration dict, verify threshold, source label) or (None, None, None)."""
    if not path or not Path(path).exists():
        return None, None, None
    cal = json.load(open(path, encoding="utf-8"))
    thr = cal["reliability"]["verify_threshold"]
    if camera_id in cal["cameras"]:
        return cal["cameras"][camera_id], thr, "camera"
    cams = list(cal["cameras"].values())          # camera never calibrated: pool the calibrated cameras
    pooled = {k: sum((c[k] for c in cams), []) for k in ("ref_conf", "ref_trust", "ref_ypos")}
    pooled["count_factor"] = round(sum(c["count_factor"] for c in cams) / len(cams), 3)
    floors = [c["empty_floor"] for c in cams if "empty_floor" in c]
    if floors:
        pooled["empty_floor"] = round(sum(floors) / len(floors), 3)
    return pooled, thr, "pooled (camera not calibrated)"


def apply_calibration(recs, zone_stats, camcal, thr, source, low_ratio, out_ratio):
    """Adds the three research improvements to each record:
    1) reliability = combined score of detector confidence + occlusion trust + position (verify if below threshold)
    2) adjusted ratio / status with the empty-shelf floor removed (real background only)
    3) estimated true count with the camera's count factor (precision / recall)."""
    for r in recs:
        z = zone_stats.get(r["location_id"], {})
        r["calibration"] = source
        r["mean_confidence"] = None if z.get("conf") is None else round(z["conf"], 3)
        r["position_y"] = round(z.get("ypos", 0.5), 3)
        r["estimated_true_count"] = round(r["count"] * camcal["count_factor"])
        # empty-shelf correction
        floor = camcal.get("empty_floor", 0.0)
        r["raw_status"] = r["status"]
        if r["ratio_to_reference"] is not None:
            adj = max(0.0, (r["ratio_to_reference"] - floor) / (1 - floor))
            r["adjusted_ratio"] = round(adj, 3)
            if r["status"] != "UNKNOWN":
                r["status"] = ("STOCK-OUT" if r["count"] == 0 or adj < out_ratio else "LOW" if adj < low_ratio else "IN STOCK")
        else:
            r["adjusted_ratio"] = None
        # combined reliability
        if r.get("trust") is not None and z.get("conf") is not None:
            unrel = (pct_unreliable(z["conf"], camcal["ref_conf"]) + pct_unreliable(r["trust"], camcal["ref_trust"])
                     + pct_unreliable(z.get("ypos", 0.5), camcal["ref_ypos"])) / 3
            r["reliability"] = round(1 - unrel, 3)
            r["verify_manually"] = bool(unrel >= thr)
            r["verify_rule"] = f"combined score (confidence + occlusion + position), verify if reliability < {1 - thr:.2f}"
        else:
            r["reliability"] = None
    return recs


def gt_counts(labels_dir, frame_path, locations, anchor="center"):
    """Annotated (ground-truth) count per location from a YOLO label file, or None if missing."""
    if not labels_dir:
        return None
    lf = Path(labels_dir) / (Path(frame_path).stem + ".txt")
    if not lf.exists():
        return None
    boxes = []
    for line in lf.read_text().split("\n"):
        parts = line.split()
        if len(parts) < 5:
            continue
        cx, cy, bw, bh = map(float, parts[1:5])
        boxes.append((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2, 1.0))
    return {k: len(v) for k, v in assign_boxes(boxes, locations, 1.0, 1.0, anchor).items()}


def detect(model, path, conf, imgsz, max_det, iou=0.7):
    r = model.predict(str(path), conf=conf, iou=iou, imgsz=imgsz, max_det=max_det, verbose=False)[0]
    h, w = r.orig_shape
    boxes = [(*map(float, b.xyxy[0].tolist()), float(b.conf[0])) for b in r.boxes]
    return boxes, w, h


LEVEL_COLOURS = [(0, 200, 0), (0, 165, 255), (0, 0, 255)]   # low green, medium orange, high red (BGR)


def draw_overlay(path, boxes, locations, out_path, recs, levels=None, unassigned=None):
    import cv2
    import numpy as np
    img = cv2.imread(str(path))
    h, w = img.shape[:2]
    for i, b in enumerate(boxes):
        c = LEVEL_COLOURS[levels[i]] if levels is not None else (0, 0, 255)
        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), c, 1)
    for b in unassigned or []:   # boxes on no shelf (aisle / floor): thin white
        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 255), 1)
    colour = {"IN STOCK": (0, 160, 0), "LOW": (0, 165, 255), "STOCK-OUT": (0, 0, 255), "UNKNOWN": (128, 128, 128)}
    short = {"IN STOCK": "IN", "LOW": "LOW", "STOCK-OUT": "OUT", "UNKNOWN": "?"}
    by_id = {r["location_id"]: r for r in recs}
    compact = len(locations) > 4 or any("polygon" in l for l in locations)
    for loc in locations:
        poly = loc_polygon(loc)
        r = by_id[loc["location_id"]]
        c = colour.get(r["status"], (128, 128, 128))
        pts = np.array([[int(x * w), int(y * h)] for x, y in poly], np.int32)
        cv2.polylines(img, [pts], True, c, 2 if compact else 3)
        if compact:
            # one short line at the shape's centre: "G-22 12 IN .78 V"  (V = verify manually)
            txt = f'{loc["location_id"]} {r["count"]} {short.get(r["status"], "?")}'
            score = r.get("reliability") if r.get("reliability") is not None else r.get("trust")
            if score is not None:
                txt += f' {score:.2f}'.replace("0.", ".") + (" V" if r["verify_manually"] else "")
            lines, fs = [txt], 0.38
            px1, py1 = int(pts[:, 0].mean()), int(pts[:, 1].mean())
            (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
            px1, py1, px2 = max(0, px1 - tw // 2 - 5), max(0, py1 - 18), w
        else:
            lines = [loc["location_id"], f'{r["count"]} | {r["status"]}']
            score = r.get("reliability") if r.get("reliability") is not None else r.get("trust")
            if score is not None:
                lines.append(f'{"rel" if r.get("reliability") is not None else "trust"} {score:.2f}'
                             + (" | VERIFY" if r["verify_manually"] else ""))
            fs = 0.45
            px1, py1, px2 = int(pts[:, 0].min()), int(pts[:, 1].min()), int(pts[:, 0].max())
        for i, txt in enumerate(lines):
            (tw, tht), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
            tx, ty = px1 + 5, py1 + 18 + i * 18
            cv2.rectangle(img, (tx - 3, ty - tht - 4), (min(tx + tw + 3, px2 - 2), ty + 4), (0, 0, 0), -1)
            cv2.putText(img, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, c, 1, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--camera-id", required=True)
    ap.add_argument("--frames", nargs="+", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--overlay-dir", default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--max-det", type=int, default=1000)
    ap.add_argument("--iou", type=float, default=0.7, help="NMS IoU; lower (e.g. 0.5) removes more overlapping duplicate boxes")
    ap.add_argument("--labels-dir", default=None, help="optional folder of YOLO .txt labels to print annotated counts next to detected counts")
    ap.add_argument("--occlusion-cnn", default=None, help="CNN weights, e.g. weights\\occlusion_r18.pt (enables sub-function 2)")
    ap.add_argument("--occlusion-mode", default="hybrid", choices=["hybrid", "cnn"])
    ap.add_argument("--rf-model", default="weights/occlusion_geometry_rf.joblib")
    ap.add_argument("--converted", default="annotations/converted", help="used once to train the geometry model")
    ap.add_argument("--trust-threshold", type=float, default=0.80)
    ap.add_argument("--low-ratio", type=float, default=0.5)
    ap.add_argument("--out-ratio", type=float, default=0.1)
    ap.add_argument("--min-ref", type=int, default=5,
                    help="locations whose reference frame shows fewer cartons than this get status UNKNOWN")
    ap.add_argument("--anchor", default="center", choices=["center", "bottom"],
                    help="box point used to pick its location; 'bottom' = where the carton stands (good for stacked shelf levels)")
    ap.add_argument("--calibration", default="configs/calibration.json",
                    help="from build_calibration.py; adds combined reliability, empty-shelf and count corrections")
    ap.add_argument("--no-calibration", action="store_true", help="old behaviour (trust-only verify, raw ratio)")
    a = ap.parse_args()

    cfg = json.load(open(a.config, encoding="utf-8"))
    cam = cfg["cameras"][a.camera_id]
    locations = cam["locations"]

    from ultralytics import YOLO
    model = YOLO(a.weights)
    occ_model = None
    if a.occlusion_cnn:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from occlusion_model import OcclusionEstimator, location_summary
        occ_model = OcclusionEstimator(a.occlusion_cnn, mode=a.occlusion_mode, rf_path=a.rf_model, converted=a.converted)

    # reference counts: given in config, else counted on the camera's reference frame
    references = {l["location_id"]: l.get("reference_count") for l in locations}
    if any(v is None for v in references.values()):
        if "reference_frame" not in cam:
            raise SystemExit("Config needs 'reference_frame' or a reference_count for every location.")
        rb, rw, rh = detect(model, cam["reference_frame"], a.conf, a.imgsz, a.max_det, a.iou)
        ref_assigned = assign_boxes(rb, locations, rw, rh, a.anchor)
        for k, v in references.items():
            if v is None:
                references[k] = len(ref_assigned[k])
    print("Reference counts:", references)
    camcal, cal_thr, cal_src = (None, None, None) if a.no_calibration else load_camera_calibration(a.calibration, a.camera_id)
    if camcal is not None and a.occlusion_cnn:
        print(f"Calibration: {cal_src}; count factor {camcal['count_factor']}, empty-shelf floor {camcal.get('empty_floor', 0)}, "
              f"verify if reliability < {1 - cal_thr:.2f}")
    elif a.occlusion_cnn:
        print("Calibration: none (trust-only verify, raw ratio)")

    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    all_recs = []
    for fp in a.frames:
        boxes, w, h = detect(model, fp, a.conf, a.imgsz, a.max_det, a.iou)
        loose = []
        per_loc = assign_boxes(boxes, locations, w, h, a.anchor, loose)
        occ, levels = None, None
        if occ_model is not None:
            import cv2
            probs = occ_model.predict(cv2.imread(str(fp)), boxes)
            levels = probs.argmax(1) if len(probs) else None
            index = {id(b): i for i, b in enumerate(boxes)}
            occ = {lid: location_summary(probs[[index[id(b)] for b in bx]] if bx else probs[:0], a.trust_threshold)
                   for lid, bx in per_loc.items()}
        recs = build_records(a.camera_id, Path(fp).name, per_loc, references, a.low_ratio, a.out_ratio, ts, occ,
                             {l["location_id"]: l.get("slots", []) for l in locations}, a.min_ref)
        if camcal is not None and occ_model is not None:
            zone_stats = {}
            for loc in locations:
                bx = per_loc[loc["location_id"]]
                ys = [p[1] for p in loc_polygon(loc)]
                zone_stats[loc["location_id"]] = {"conf": (sum(b[4] for b in bx) / len(bx)) if bx else None,
                                                  "ypos": sum(ys) / len(ys)}
            apply_calibration(recs, zone_stats, camcal, cal_thr, cal_src, a.low_ratio, a.out_ratio)
        all_recs += recs
        gt = gt_counts(a.labels_dir, fp, locations, a.anchor)
        for r in recs:
            extra = "" if gt is None else f'   [annotated: {gt[r["location_id"]]}]'
            if r.get("reliability") is not None:
                tr = (f'  reliability {r["reliability"]:.2f}{"  VERIFY MANUALLY" if r["verify_manually"] else ""}'
                      f'  (trust {r["trust"]:.2f}, conf {r["mean_confidence"]:.2f})')
            elif r.get("trust") is not None:
                tr = f'  trust {r["trust"]:.2f}{"  VERIFY MANUALLY" if r["verify_manually"] else ""}'
            else:
                tr = ""
            cal_txt = ""
            if "estimated_true_count" in r:
                adj = "" if r.get("adjusted_ratio") is None else f', adj ratio {r["adjusted_ratio"]:.2f}'
                cal_txt = f'  [est. true count {r["estimated_true_count"]}{adj}]'
            print(f'{Path(fp).name}  {r["location_id"]}: {r["count"]} (ref {r["reference_count"]}) -> {r["status"]}{cal_txt}{extra}{tr}')
        if loose:
            print(f'{Path(fp).name}  (not on any location: {len(loose)} boxes, e.g. aisle/floor; shown white in the overlay)')
        if a.overlay_dir:
            draw_overlay(fp, boxes, locations, Path(a.overlay_dir) / f"{Path(fp).stem}_stock.jpg", recs, levels, loose)

    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(all_recs, open(a.out_json, "w", encoding="utf-8"), indent=2)
    if a.out_csv:
        with open(a.out_csv, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(all_recs[0].keys()), extrasaction="ignore")
            wr.writeheader(); wr.writerows(all_recs)
    print(f"Wrote {len(all_recs)} records to {a.out_json}")


if __name__ == "__main__":
    main()
