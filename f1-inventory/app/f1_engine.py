"""
F1 engine for the dashboard: loads the saved results and runs the same pipeline as scripts/count_stock.py on one frame.
Nothing new is computed here: it reuses count_stock.py and occlusion_model.py, so the dashboard and the scripts
always give the same answer.
"""
import csv, json, os, sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]            # the f1-inventory folder
sys.path.insert(0, str(ROOT / "scripts"))
import count_stock as cs                               # noqa: E402

CAMERA_NAMES = {"G_H": "Location G-H", "X_5": "Location X-5", "CP2": "Checking Part 2",
                "CPF": "Checking Part Full", "CAM18": "Camera 18"}
STATUS_KEY = {"IN STOCK": "IN", "LOW": "LOW", "STOCK-OUT": "OUT", "UNKNOWN": "UNK"}
STATUS_COLOUR = {"IN": "#16A34A", "LOW": "#D97706", "OUT": "#DC2626", "UNK": "#94A3B8"}
LEVELS = ["low", "medium", "high"]
LEVEL_BGR = [(74, 163, 22), (6, 119, 217), (38, 38, 220)]   # green / amber / red
BLUE_BGR = (235, 99, 37)

DEFAULTS = {
    "shelves": "configs/shelves.json",
    "coverage": "configs/camera_coverage.csv",
    "calibration": "configs/calibration.json",
    "weights": "weights/day2_s1024_best.pt",
    "occlusion_cnn": "weights/occlusion_r18.pt",
    "rf_model": "weights/occlusion_geometry_rf.joblib",
    "saved_json": "outputs/stock_{cam}_shelves.json",
    "saved_overlays": "outputs/overlays_shelves",
    "live_dir": "outputs/dashboard",
    "conf": 0.25, "iou": 0.5, "imgsz": 1024, "max_det": 1000,
    "low_ratio": 0.5, "out_ratio": 0.1, "min_ref": 5, "trust_threshold": 0.80, "anchor": "center",
}


def p(rel):
    q = Path(rel)
    return q if q.is_absolute() else ROOT / q


# ------------------------------------------------------------------ configs
def load_shelves(path=DEFAULTS["shelves"]):
    f = p(path)
    return json.load(open(f, encoding="utf-8"))["cameras"] if f.exists() else {}


def load_coverage(path=DEFAULTS["coverage"]):
    f = p(path)
    if not f.exists():
        return []
    rows = []
    for r in csv.DictReader(open(f, encoding="utf-8")):
        rows.append({k.strip(): (v or "").strip() for k, v in r.items() if k})
    return rows


def load_calibration(path=DEFAULTS["calibration"]):
    f = p(path)
    return json.load(open(f, encoding="utf-8")) if f.exists() else None


IMG_EXT = (".jpg", ".jpeg", ".png")


def camera_frames(folder):
    """All frames in one extracted_frames folder."""
    folder = Path(folder)
    return sorted(q for q in folder.iterdir() if q.suffix.lower() in IMG_EXT) if folder.exists() else []


def camera_folders(cam_cfg, frames_root=None):
    """Every extracted_frames folder (any day) recorded by the same camera as the reference frame.
    Same camera = frame names start with the same prefix, e.g. 'cam10' in 'cam10__frame_00038.jpg'.
    Returns [(label, folder, holds_reference_frame)]; the folder the shelves were drawn on comes first."""
    ref = cam_cfg.get("reference_frame")
    if not ref or not Path(ref).exists():
        return []
    ref = Path(ref)
    ref_dir = ref.parent
    prefix = ref.name.split("__frame_")[0] + "__" if "__frame_" in ref.name else None
    root = Path(frames_root) if frames_root else (ref.parents[3] if len(ref.parents) > 3 else ref_dir)
    found = [ref_dir]
    if prefix and root.exists():
        for dirpath, dirnames, _ in os.walk(root):
            if Path(dirpath).name == "extracted_frames":
                dirnames[:] = []                    # do not walk into the frames themselves
                d = Path(dirpath)
                if d != ref_dir and next(d.glob(prefix + "*"), None) is not None:
                    found.append(d)
    out = []
    for d in found:
        video = d.parent.name
        label = (f"{d.parents[1].name} · {'…' + video[-8:] if len(video) > 10 else video}" if len(d.parents) > 1 else d.name)
        out.append((label, d, d == ref_dir))
    return [out[0]] + sorted(out[1:], key=lambda t: t[0])


# ------------------------------------------------------------------ saved results
def load_results(cam_id):
    """Latest result for a camera. A live run from the dashboard wins over the saved count_stock.py output.
    Returns {frame_name: {"records": [...], "overlay": path or None, "source": str}}"""
    out = {}
    saved = p(DEFAULTS["saved_json"].format(cam=cam_id))
    if saved.exists():
        recs = json.load(open(saved, encoding="utf-8"))
        for r in recs:
            out.setdefault(r["frame"], {"records": [], "overlay": None, "source": "saved run (count_stock.py)"})
            out[r["frame"]]["records"].append(r)
        for fr, v in out.items():
            ov = p(DEFAULTS["saved_overlays"]) / f"{Path(fr).stem}_stock.jpg"
            v["overlay"] = str(ov) if ov.exists() else None
    live = p(DEFAULTS["live_dir"]) / f"latest_{cam_id}.json"
    if live.exists():
        d = json.load(open(live, encoding="utf-8"))
        out[d["frame"]] = {"records": d["records"], "overlay": d.get("overlay"), "source": "live run (dashboard)",
                           "frame_path": d.get("frame_path")}
        out = {d["frame"]: out.pop(d["frame"]), **out}      # newest first
    return out


# ------------------------------------------------------------------ explanations
def reliability_parts(rec, camcal):
    """How unusual each signal is for this camera (0 = most reliable seen, 1 = least reliable seen)."""
    if not camcal or rec.get("mean_confidence") is None or rec.get("trust") is None:
        return None
    return {"confidence": cs.pct_unreliable(rec["mean_confidence"], camcal["ref_conf"]),
            "occlusion": cs.pct_unreliable(rec["trust"], camcal["ref_trust"]),
            "position": cs.pct_unreliable(rec.get("position_y", 0.5), camcal["ref_ypos"])}


def verify_reasons(rec, camcal):
    parts = reliability_parts(rec, camcal)
    if not parts:
        return []
    words = {"confidence": "the detector is less sure than usual for this camera",
             "occlusion": "many cartons here look partly or heavily hidden",
             "position": "this shelf is far from the camera (top of the picture)"}
    return [words[k] for k, v in sorted(parts.items(), key=lambda kv: -kv[1]) if v >= 0.5]


# ------------------------------------------------------------------ drawing
def draw_boxes(img, boxes, colours, thick=1):
    out = img.copy()
    for b, c in zip(boxes, colours):
        cv2.rectangle(out, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), c, thick)
    return out


def draw_shelf_outlines(img, locations, colour=BLUE_BGR, label=True, fill=0.18):
    h, w = img.shape[:2]
    over, out = img.copy(), img.copy()
    for loc in locations:
        pts = np.array([[int(x * w), int(y * h)] for x, y in cs.loc_polygon(loc)], np.int32)
        cv2.fillPoly(over, [pts], colour)
        cv2.polylines(out, [pts], True, colour, 2)
    out = cv2.addWeighted(over, fill, out, 1 - fill, 0)
    if label:
        for loc in locations:
            pts = np.array([[int(x * w), int(y * h)] for x, y in cs.loc_polygon(loc)], np.int32)
            t = loc["location_id"]
            (tw, th), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
            cv2.rectangle(out, (cx - tw // 2 - 5, cy - th - 6), (cx + tw // 2 + 5, cy + 5), (54, 23, 11), -1)
            cv2.putText(out, t, (cx - tw // 2, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def crop_location(img, loc, margin=0.06):
    h, w = img.shape[:2]
    poly = np.array(cs.loc_polygon(loc))
    x1, y1 = poly.min(0) - margin; x2, y2 = poly.max(0) + margin
    X1, Y1, X2, Y2 = int(max(x1, 0) * w), int(max(y1, 0) * h), int(min(x2, 1) * w), int(min(y2, 1) * h)
    return img[Y1:Y2, X1:X2] if X2 > X1 and Y2 > Y1 else img


def rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


# ------------------------------------------------------------------ live run
def load_models(weights, occlusion_cnn, rf_model):
    from ultralytics import YOLO
    det = YOLO(str(p(weights)))
    occ = None
    if occlusion_cnn and p(occlusion_cnn).exists():
        from occlusion_model import OcclusionEstimator
        occ = OcclusionEstimator(str(p(occlusion_cnn)), mode="hybrid", rf_path=str(p(rf_model)),
                                 converted=str(p("annotations/converted")))
    return det, occ


def reference_counts(det, cam_cfg, s):
    locs = cam_cfg["locations"]
    refs = {l["location_id"]: l.get("reference_count") for l in locs}
    if any(v is None for v in refs.values()):
        rb, rw, rh = cs.detect(det, cam_cfg["reference_frame"], s["conf"], s["imgsz"], s["max_det"], s["iou"])
        got = cs.assign_boxes(rb, locs, rw, rh, s["anchor"])
        refs = {k: (v if v is not None else len(got[k])) for k, v in refs.items()}
    return refs


def run_frame(det, occ, cam_id, cam_cfg, frame_path, refs, s, calibration_path=DEFAULTS["calibration"], save=True):
    """Same steps as count_stock.py, for one frame. Returns everything the dashboard shows."""
    from occlusion_model import location_summary
    locs = cam_cfg["locations"]
    img = cv2.imread(str(frame_path))
    boxes, w, h = cs.detect(det, frame_path, s["conf"], s["imgsz"], s["max_det"], s["iou"])
    loose = []
    per_loc = cs.assign_boxes(boxes, locs, w, h, s["anchor"], loose)

    probs, levels, occ_sum = None, None, None
    if occ is not None and len(boxes):
        probs = occ.predict(img, boxes)
        levels = probs.argmax(1)
        index = {id(b): i for i, b in enumerate(boxes)}
        occ_sum = {lid: location_summary(probs[[index[id(b)] for b in bx]] if bx else probs[:0], s["trust_threshold"])
                   for lid, bx in per_loc.items()}
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    recs = cs.build_records(cam_id, Path(frame_path).name, per_loc, refs, s["low_ratio"], s["out_ratio"], ts, occ_sum,
                            {l["location_id"]: l.get("slots", []) for l in locs}, s["min_ref"])
    camcal, thr, src = cs.load_camera_calibration(str(p(calibration_path)), cam_id)
    if camcal is not None and occ_sum is not None:
        zs = {}
        for loc in locs:
            bx = per_loc[loc["location_id"]]
            ys = [q[1] for q in cs.loc_polygon(loc)]
            zs[loc["location_id"]] = {"conf": (sum(b[4] for b in bx) / len(bx)) if bx else None, "ypos": sum(ys) / len(ys)}
        cs.apply_calibration(recs, zs, camcal, thr, src, s["low_ratio"], s["out_ratio"])

    # pictures for the four steps
    det_img = draw_boxes(img, boxes, [BLUE_BGR] * len(boxes), 1)
    occ_img = draw_boxes(img, boxes, [LEVEL_BGR[int(l)] for l in levels], 2) if levels is not None else None
    live = p(DEFAULTS["live_dir"]); (live / "overlays").mkdir(parents=True, exist_ok=True)
    ov_path = live / "overlays" / f"{Path(frame_path).stem}_stock.jpg"
    cs.draw_overlay(frame_path, boxes, locs, ov_path, recs, levels, loose)
    final = cv2.imread(str(ov_path))
    if save:
        json.dump({"camera_id": cam_id, "frame": Path(frame_path).name, "frame_path": str(frame_path),
                   "overlay": str(ov_path), "records": recs},
                  open(live / f"latest_{cam_id}.json", "w", encoding="utf-8"), indent=2)
    mix = {l: 0 for l in LEVELS}
    if levels is not None:
        for l in levels:
            mix[LEVELS[int(l)]] += 1
    return {"records": recs, "raw": img, "detections": det_img, "occlusion": occ_img, "final": final,
            "n_boxes": len(boxes), "n_loose": len(loose), "mix": mix, "overlay": str(ov_path)}
