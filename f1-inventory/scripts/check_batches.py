"""
check_batches.py

Two jobs in one pass over every converted annotation batch:

1. INVENTORY - finds each annotated frame's real image on disk, works out
   which camera folder and which day it came from, and writes
   annotations/inventory.csv (one row per batch: day, camera folder,
   frames, boxes, low/medium/high counts, boxes per frame, duplicate-box
   pairs, problems).

2. VISUAL QA - draws the converted YOLO boxes onto a few frames per batch,
   coloured by occlusion level (green=low, orange=medium, red=high), so you
   can eyeball that boxes sit on the cartons and nothing is duplicated.

Annotated frames are identified purely by the filenames inside each batch's
occlusion_labels.csv. Frames you extracted but never annotated are simply
never touched, so you do NOT need to move or separate anything.

Expected folder layout (what you have now):
    <frames-root>/<dd-mm-yyyy>/<camera folder>/extracted_frames/<frame>.jpg

Usage (from the f1-inventory folder):
    python scripts\\check_batches.py --frames-root "D:\\Research\\research camera data"

Optional:
    --converted annotations\\converted     where the batch folders are
    --out annotations                      where inventory.csv is written
    --qa-per-batch 3                       overlay images per batch (0 = skip)

Requires: opencv-python (you already have it from frame extraction)
"""

import re
import sys
import csv
import argparse
from pathlib import Path
from collections import defaultdict

try:
    import cv2
except ImportError:
    print("This script needs opencv-python:  python -m pip install opencv-python")
    sys.exit(1)

COLORS = {            # BGR
    "low": (80, 200, 80),
    "medium": (0, 165, 255),
    "high": (60, 60, 230),
    "unknown": (200, 200, 200),
}
DAY_RE = re.compile(r"^\d{2}-\d{2}-\d{4}$")


def index_frames(frames_root: Path):
    """filename -> list of paths, for every .jpg inside an extracted_frames folder."""
    index = defaultdict(list)
    for p in frames_root.rglob("*"):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and p.parent.name == "extracted_frames":
            index[p.name].append(p)
    return index


def day_and_camera(path: Path):
    """Pull the dd-mm-yyyy day folder and the camera folder out of the path."""
    parts = path.parts
    day = next((x for x in parts if DAY_RE.match(x)), "unknown-day")
    camera = path.parent.parent.name  # .../<camera folder>/extracted_frames/<file>
    return day, camera


def read_batch(batch_dir: Path):
    rows = list(csv.DictReader(open(batch_dir / "occlusion_labels.csv", newline="")))
    by_image = defaultdict(list)
    for r in rows:
        by_image[r["image_name"]].append(r)
    for img in by_image:
        by_image[img].sort(key=lambda r: int(r["box_index"]))
    return by_image


def dup_pairs(box_rows, thresh=0.8):
    """Count box pairs in one image that overlap almost completely (IoU > thresh).
    Exact copies drawn on top of each other are invisible in the overlay images,
    so this catches propagation/copy-paste duplicates numerically."""
    b = [(float(r["xtl"]), float(r["ytl"]), float(r["xbr"]), float(r["ybr"])) for r in box_rows]
    n_dup = 0
    for i in range(len(b)):
        ax1, ay1, ax2, ay2 = b[i]
        a_area = max(ax2 - ax1, 0) * max(ay2 - ay1, 0)
        for j in range(i + 1, len(b)):
            bx1, by1, bx2, by2 = b[j]
            iw = min(ax2, bx2) - max(ax1, bx1)
            ih = min(ay2, by2) - max(ay1, by1)
            if iw <= 0 or ih <= 0:
                continue
            inter = iw * ih
            union = a_area + max(bx2 - bx1, 0) * max(by2 - by1, 0) - inter
            if union > 0 and inter / union > thresh:
                n_dup += 1
    return n_dup


def draw_overlay(img_path: Path, label_path: Path, box_rows, out_path: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        return False
    h, w = img.shape[:2]
    lines = [l.split() for l in label_path.read_text().splitlines() if l.strip()]
    for i, parts in enumerate(lines):
        _, xc, yc, bw, bh = map(float, parts[:5])
        x1, y1 = int((xc - bw / 2) * w), int((yc - bh / 2) * h)
        x2, y2 = int((xc + bw / 2) * w), int((yc + bh / 2) * h)
        level = box_rows[i]["occlusion_level"] if i < len(box_rows) else "unknown"
        cv2.rectangle(img, (x1, y1), (x2, y2), COLORS.get(level, COLORS["unknown"]), 1)
    cv2.putText(img, f"{img_path.name}  boxes={len(lines)}", (6, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, f"{img_path.name}  boxes={len(lines)}", (6, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-root", required=True)
    ap.add_argument("--converted", default=r"annotations\converted")
    ap.add_argument("--out", default="annotations")
    ap.add_argument("--qa-per-batch", type=int, default=3)
    args = ap.parse_args()

    frames_root = Path(args.frames_root)
    converted = Path(args.converted)
    out_dir = Path(args.out)
    if not frames_root.exists():
        sys.exit(f"Frames folder not found: {frames_root}")
    if not converted.exists():
        sys.exit(f"Converted folder not found: {converted}")

    print(f"Indexing frames under {frames_root} ...")
    index = index_frames(frames_root)
    print(f"  found {sum(len(v) for v in index.values())} frame files in extracted_frames folders\n")

    batches = sorted(d for d in converted.iterdir() if (d / "occlusion_labels.csv").exists())
    if not batches:
        sys.exit("No batch folders with occlusion_labels.csv found.")

    inventory = []
    for batch_dir in batches:
        by_image = read_batch(batch_dir)
        problems = []
        days, cameras = set(), set()
        counts = {"low": 0, "medium": 0, "high": 0}
        found_paths = {}
        dup_total, dup_images = 0, []

        for name, box_rows in by_image.items():
            d_ = dup_pairs(box_rows)
            if d_:
                dup_total += d_
                dup_images.append(name)
            for r in box_rows:
                counts[r["occlusion_level"]] = counts.get(r["occlusion_level"], 0) + 1
            paths = index.get(name, [])
            if not paths:
                problems.append(f"missing:{name}")
                continue
            if len(paths) > 1:
                problems.append(f"duplicate-name:{name}")
            p = paths[0]
            found_paths[name] = p
            d, c = day_and_camera(p)
            days.add(d)
            cameras.add(c)

            label = batch_dir / "labels" / (Path(name).stem + ".txt")
            n_lines = len([l for l in label.read_text().splitlines() if l.strip()]) if label.exists() else -1
            if n_lines != len(box_rows):
                problems.append(f"label-csv-mismatch:{name}({n_lines}vs{len(box_rows)})")

        if dup_total:
            problems.append(f"duplicate-boxes:{dup_total}pairs in {len(dup_images)} frames (e.g. {dup_images[0]})")
        n_frames = len(by_image)
        n_boxes = sum(len(v) for v in by_image.values())
        inventory.append({
            "batch": batch_dir.name,
            "day": "|".join(sorted(days)) or "?",
            "camera_folder": "|".join(sorted(cameras)) or "?",
            "frames": n_frames,
            "frames_found_on_disk": len(found_paths),
            "boxes": n_boxes,
            "low": counts.get("low", 0),
            "medium": counts.get("medium", 0),
            "high": counts.get("high", 0),
            "boxes_per_frame": round(n_boxes / n_frames, 1) if n_frames else 0,
            "dup_pairs": dup_total,
            "problems": ";".join(problems) if problems else "",
        })

        # visual QA overlays: evenly spaced frames
        if args.qa_per_batch > 0 and found_paths:
            names = sorted(found_paths)
            k = min(args.qa_per_batch, len(names))
            picks = [names[int(i * (len(names) - 1) / max(k - 1, 1))] for i in range(k)] if k > 1 else names[:1]
            for name in dict.fromkeys(picks):
                label = batch_dir / "labels" / (Path(name).stem + ".txt")
                if label.exists():
                    draw_overlay(found_paths[name], label, by_image[name],
                                 out_dir / "qa_overlays" / f"{batch_dir.name}__{Path(name).stem}.jpg")

    out_dir.mkdir(parents=True, exist_ok=True)
    inv_path = out_dir / "inventory.csv"
    fields = list(inventory[0].keys())
    with open(inv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(inventory)

    print(f"{'batch':34} {'day':11} {'frames':>6} {'boxes':>6} {'low':>5} {'med':>5} {'high':>5} {'box/fr':>6}  problems")
    for r in inventory:
        print(f"{r['batch'][:34]:34} {r['day'][:11]:11} {r['frames']:>6} {r['boxes']:>6} "
              f"{r['low']:>5} {r['medium']:>5} {r['high']:>5} {r['boxes_per_frame']:>6}  {r['problems'] or 'ok'}")
    tot = lambda k: sum(r[k] for r in inventory)
    print(f"{'TOTAL':34} {'':11} {tot('frames'):>6} {tot('boxes'):>6} {tot('low'):>5} {tot('medium'):>5} {tot('high'):>5}")
    print(f"\nInventory written to: {inv_path}")
    if args.qa_per_batch > 0:
        print(f"Overlay previews written to: {out_dir / 'qa_overlays'}  (open a few and check the boxes)")


if __name__ == "__main__":
    main()
