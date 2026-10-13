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
frame) and a list of locations. Each location has a region [x1,y1,x2,y2] as fractions of the
image (0-1), so it works at any resolution. Use [0,0,1,1] for "whole frame = one location".
Optionally give "reference_count" to skip counting the reference frame.

Status rule (thresholds are PLACEHOLDERS to tune on real frames):
  ratio = count / reference_count
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

SCHEMA_VERSION = "0.2-draft"


def assign_boxes(boxes, locations, w, h):
    """boxes: list of (x1,y1,x2,y2,conf) in pixels. Returns {location_id: [box,...]}.
    A box belongs to the first location whose region contains the box centre."""
    out = {loc["location_id"]: [] for loc in locations}
    for b in boxes:
        cx, cy = (b[0] + b[2]) / 2 / w, (b[1] + b[3]) / 2 / h
        for loc in locations:
            x1, y1, x2, y2 = loc["region"]
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                out[loc["location_id"]].append(b)
                break
    return out


def stock_status(count, reference, low_ratio=0.5, out_ratio=0.1):
    """Returns (status, ratio). ratio is None when there is no usable reference."""
    if reference is None or reference <= 0:
        return ("STOCK-OUT" if count == 0 else "UNKNOWN"), None
    ratio = count / reference
    if count == 0 or ratio < out_ratio:
        return "STOCK-OUT", ratio
    if ratio < low_ratio:
        return "LOW", ratio
    return "IN STOCK", ratio


def build_records(camera_id, frame_name, per_loc, references, low_ratio, out_ratio, timestamp, occ=None):
    recs = []
    for loc_id, bx in per_loc.items():
        o = (occ or {}).get(loc_id, {})
        count = len(bx)
        ref = references.get(loc_id)
        status, ratio = stock_status(count, ref, low_ratio, out_ratio)
        recs.append({
            "schema_version": SCHEMA_VERSION,
            "location_id": loc_id,
            "camera_id": camera_id,
            "frame": frame_name,
            "timestamp": timestamp,
            "count": count,
            "reference_count": ref,
            "ratio_to_reference": None if ratio is None else round(ratio, 3),
            "status": status,
            "occlusion_mix": o.get("occlusion_mix"),
            "trust": o.get("trust"),                  # sub-function 2 (None if occlusion not run)
            "verify_manually": o.get("verify_manually"),
        })
    return recs


def gt_counts(labels_dir, frame_path, locations):
    """Annotated (ground-truth) count per location from a YOLO label file, or None if missing."""
    if not labels_dir:
        return None
    lf = Path(labels_dir) / (Path(frame_path).stem + ".txt")
    if not lf.exists():
        return None
    out = {loc["location_id"]: 0 for loc in locations}
    for line in lf.read_text().split("\n"):
        parts = line.split()
        if len(parts) < 5:
            continue
        cx, cy = float(parts[1]), float(parts[2])
        for loc in locations:
            x1, y1, x2, y2 = loc["region"]
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                out[loc["location_id"]] += 1
                break
    return out


def detect(model, path, conf, imgsz, max_det, iou=0.7):
    r = model.predict(str(path), conf=conf, iou=iou, imgsz=imgsz, max_det=max_det, verbose=False)[0]
    h, w = r.orig_shape
    boxes = [(*map(float, b.xyxy[0].tolist()), float(b.conf[0])) for b in r.boxes]
    return boxes, w, h


LEVEL_COLOURS = [(0, 200, 0), (0, 165, 255), (0, 0, 255)]   # low green, medium orange, high red (BGR)


def draw_overlay(path, boxes, locations, out_path, recs, levels=None):
    import cv2
    img = cv2.imread(str(path))
    h, w = img.shape[:2]
    for i, b in enumerate(boxes):
        c = LEVEL_COLOURS[levels[i]] if levels is not None else (0, 0, 255)
        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), c, 1)
    colour = {"IN STOCK": (0, 160, 0), "LOW": (0, 165, 255), "STOCK-OUT": (0, 0, 255), "UNKNOWN": (128, 128, 128)}
    by_id = {r["location_id"]: r for r in recs}
    for loc in locations:
        x1, y1, x2, y2 = loc["region"]
        r = by_id[loc["location_id"]]
        c = colour.get(r["status"], (128, 128, 128))
        px1, py1, px2, py2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
        cv2.rectangle(img, (px1, py1), (px2, py2), c, 3)
        # two short label lines inside the region, on a dark background so they stay readable
        lines = [loc["location_id"], f'{r["count"]} | {r["status"]}']
        if r.get("trust") is not None:
            lines.append(f'trust {r["trust"]:.2f}' + (" | VERIFY" if r["verify_manually"] else ""))
        fs, th = 0.45, 1
        for i, txt in enumerate(lines):
            (tw, tht), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
            tx, ty = px1 + 5, py1 + 18 + i * 18
            cv2.rectangle(img, (tx - 3, ty - tht - 4), (min(tx + tw + 3, px2 - 2), ty + 4), (0, 0, 0), -1)
            cv2.putText(img, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, c, th, cv2.LINE_AA)
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
        ref_assigned = assign_boxes(rb, locations, rw, rh)
        for k, v in references.items():
            if v is None:
                references[k] = len(ref_assigned[k])
    print("Reference counts:", references)

    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    all_recs = []
    for fp in a.frames:
        boxes, w, h = detect(model, fp, a.conf, a.imgsz, a.max_det, a.iou)
        per_loc = assign_boxes(boxes, locations, w, h)
        occ, levels = None, None
        if occ_model is not None:
            import cv2
            probs = occ_model.predict(cv2.imread(str(fp)), boxes)
            levels = probs.argmax(1) if len(probs) else None
            index = {id(b): i for i, b in enumerate(boxes)}
            occ = {lid: location_summary(probs[[index[id(b)] for b in bx]] if bx else probs[:0], a.trust_threshold)
                   for lid, bx in per_loc.items()}
        recs = build_records(a.camera_id, Path(fp).name, per_loc, references, a.low_ratio, a.out_ratio, ts, occ)
        all_recs += recs
        gt = gt_counts(a.labels_dir, fp, locations)
        for r in recs:
            extra = "" if gt is None else f'   [annotated: {gt[r["location_id"]]}]'
            tr = "" if r.get("trust") is None else f'  trust {r["trust"]:.2f}{"  VERIFY MANUALLY" if r["verify_manually"] else ""}  mix {r["occlusion_mix"]}'
            print(f'{Path(fp).name}  {r["location_id"]}: {r["count"]} (ref {r["reference_count"]}) -> {r["status"]}{extra}{tr}')
        if a.overlay_dir:
            draw_overlay(fp, boxes, locations, Path(a.overlay_dir) / f"{Path(fp).stem}_stock.jpg", recs, levels)

    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(all_recs, open(a.out_json, "w", encoding="utf-8"), indent=2)
    if a.out_csv:
        with open(a.out_csv, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(all_recs[0].keys()))
            wr.writeheader(); wr.writerows(all_recs)
    print(f"Wrote {len(all_recs)} records to {a.out_json}")


if __name__ == "__main__":
    main()
