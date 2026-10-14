r"""
F1 sub-function 2, step 1: detector recall by occlusion level.

Question: does the detector miss more cartons when they are more hidden?
For every annotated carton in the chosen batches, check whether the detector found it
(a predicted box overlapping it with IoU >= --match-iou, one prediction per carton),
then report recall separately for low / medium / high occlusion.

Uses annotations\converted\<batch>\occlusion_labels.csv
(columns: image_name, box_index, xtl, ytl, xbr, ybr, occlusion_level; pixel coordinates).

Usage (from the f1-inventory folder):
  python scripts\recall_by_occlusion.py --weights weights\day2_s1024_best.pt ^
      --frames-root "D:\Research\research camera data" ^
      --batches location_G_H_b2 location_X_5_b2 location_checkingPart_2_b2 ^
      --out-csv outputs\recall_by_occlusion.csv

Default batches are the day-2 val/test batches (never used for training).
"""
import argparse, csv, math
from pathlib import Path

LEVELS = ["low", "medium", "high"]


def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match(gts, preds, thr):
    """Greedy one-to-one matching, highest IoU first. Returns a list of booleans (matched?) per GT."""
    pairs = []
    for gi, g in enumerate(gts):
        for pi, p in enumerate(preds):
            v = iou(g, p)
            if v >= thr:
                pairs.append((v, gi, pi))
    pairs.sort(reverse=True)
    used_g, used_p = set(), set()
    for v, gi, pi in pairs:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi); used_p.add(pi)
    return [gi in used_g for gi in range(len(gts))]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--frames-root", required=True)
    ap.add_argument("--converted", default="annotations/converted")
    ap.add_argument("--batches", nargs="+",
                    default=["location_G_H_b2", "location_X_5_b2", "location_checkingPart_2_b2"])
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5, help="NMS IoU (same as count_stock.py)")
    ap.add_argument("--match-iou", type=float, default=0.5, help="IoU needed to count a carton as found")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--max-det", type=int, default=1000)
    ap.add_argument("--out-csv", default="outputs/recall_by_occlusion.csv")
    a = ap.parse_args()

    # index all extracted frames by file name
    index = {}
    for p in Path(a.frames_root).rglob("*"):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png") and p.parent.name == "extracted_frames":
            index.setdefault(p.name, p)

    from ultralytics import YOLO
    model = YOLO(a.weights)

    stats = {}  # (batch, level) -> [found, total]
    for batch in a.batches:
        csv_path = Path(a.converted) / batch / "occlusion_labels.csv"
        if not csv_path.exists():
            print(f"!! missing {csv_path}, skipped"); continue
        per_img = {}
        for r in csv.DictReader(open(csv_path, encoding="utf-8")):
            lvl = r["occlusion_level"].strip().lower()
            box = tuple(float(r[k]) for k in ("xtl", "ytl", "xbr", "ybr"))
            per_img.setdefault(r["image_name"], []).append((box, lvl))
        missing = 0
        for name, items in per_img.items():
            img = index.get(Path(name).name)
            if img is None:
                missing += 1; continue
            res = model.predict(str(img), conf=a.conf, iou=a.iou, imgsz=a.imgsz,
                                max_det=a.max_det, verbose=False)[0]
            preds = [tuple(b.xyxy[0].tolist()) for b in res.boxes]
            found = match([b for b, _ in items], preds, a.match_iou)
            for (_, lvl), f in zip(items, found):
                s = stats.setdefault((batch, lvl), [0, 0])
                s[0] += int(f); s[1] += 1
        print(f"{batch}: {len(per_img) - missing} frames scored" + (f", {missing} not found on disk" if missing else ""))

    rows = []
    for batch in a.batches + ["ALL"]:
        for lvl in LEVELS:
            if batch == "ALL":
                k = sum(v[0] for (b, l), v in stats.items() if l == lvl)
                n = sum(v[1] for (b, l), v in stats.items() if l == lvl)
            else:
                k, n = stats.get((batch, lvl), [0, 0])
            lo, hi = wilson(k, n)
            rows.append({"batch": batch, "occlusion_level": lvl, "found": k, "total": n,
                         "recall": round(k / n, 3) if n else "",
                         "ci95_low": round(lo, 3) if n else "", "ci95_high": round(hi, 3) if n else ""})

    print(f"\nRecall by occlusion level (match IoU >= {a.match_iou}, conf {a.conf}, NMS IoU {a.iou})")
    print(f"{'batch':30s} {'level':7s} {'found':>6s} {'total':>6s} {'recall':>7s}  95% CI")
    for r in rows:
        if r["total"]:
            print(f"{r['batch']:30s} {r['occlusion_level']:7s} {r['found']:6d} {r['total']:6d} {r['recall']:7.3f}  [{r['ci95_low']:.3f}, {r['ci95_high']:.3f}]")

    Path(a.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {a.out_csv}")
    print("Note: frames from one batch are near-duplicates, so the 95% CI is optimistic (too narrow).")


if __name__ == "__main__":
    main()
