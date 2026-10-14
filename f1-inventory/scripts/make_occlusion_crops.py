r"""
F1 sub-function 2, step 2a: cut out every annotated carton as a small image, sorted by occlusion level,
to train the occlusion classifier (low / medium / high).

Each crop includes some area AROUND the carton (--pad), because occlusion is about what covers it.
Crops are padded to a square (grey) and resized to --size pixels.

Split by whole batch (never by frame), same idea as the detector:
  train = day-1 batches (except the val batch), val = one day-1 batch, test = the three day-2 batches.

Output layout (works directly with torchvision ImageFolder):
  <out>/train/low|medium|high/*.jpg
  <out>/val/low|medium|high/*.jpg
  <out>/test/low|medium|high/*.jpg
  <out>/crops_manifest.csv

Usage (from the f1-inventory folder):
  python scripts\make_occlusion_crops.py --frames-root "D:\Research\research camera data" ^
      --out "D:\Research\dataset\f1_occlusion_crops"
"""
import argparse, csv
from pathlib import Path
import cv2
import numpy as np

DAY1 = ["location_a", "location_camera_18_b1", "location_checkingPart_2_b1",
        "location_checkingPart_Full_b1", "location_G_H_1", "location_X_5_b1"]
DAY2 = ["location_G_H_b2", "location_X_5_b2", "location_checkingPart_2_b2"]
LEVELS = ["low", "medium", "high"]


def crop_square(img, x1, y1, x2, y2, pad, size):
    h, w = img.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(bw, bh) * (1 + 2 * pad)
    sx1, sy1 = int(round(cx - side / 2)), int(round(cy - side / 2))
    sx2, sy2 = int(round(cx + side / 2)), int(round(cy + side / 2))
    canvas = np.full((sy2 - sy1, sx2 - sx1, 3), 114, np.uint8)
    ix1, iy1, ix2, iy2 = max(sx1, 0), max(sy1, 0), min(sx2, w), min(sy2, h)
    if ix2 > ix1 and iy2 > iy1:
        canvas[iy1 - sy1:iy2 - sy1, ix1 - sx1:ix2 - sx1] = img[iy1:iy2, ix1:ix2]
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-root", required=True)
    ap.add_argument("--converted", default="annotations/converted")
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-batches", nargs="+", default=[b for b in DAY1 if b != "location_camera_18_b1"])
    ap.add_argument("--val-batches", nargs="+", default=["location_camera_18_b1"])
    ap.add_argument("--test-batches", nargs="+", default=DAY2)
    ap.add_argument("--pad", type=float, default=0.25, help="extra context around the box, as a fraction of its size")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--min-side", type=int, default=6, help="skip boxes smaller than this (pixels)")
    ap.add_argument("--dedupe", action="store_true",
                    help="train/val only: keep one crop per carton position in each batch (test is left unchanged so results stay comparable; propagated annotations repeat the "
                         "same box in many frames; repeats let the model memorise instead of learn)")
    ap.add_argument("--dedupe-grid", type=float, default=4.0, help="pixel grid used to decide 'same position'")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    out = Path(a.out)
    if out.exists() and any(out.iterdir()) and not a.overwrite:
        raise SystemExit(f"{out} is not empty. Use --overwrite or choose another --out.")
    for s in ("train", "val", "test"):
        for lvl in LEVELS:
            (out / s / lvl).mkdir(parents=True, exist_ok=True)

    index = {}
    for p in Path(a.frames_root).rglob("*"):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png") and p.parent.name == "extracted_frames":
            index.setdefault(p.name, p)

    plan = [(b, "train") for b in a.train_batches] + [(b, "val") for b in a.val_batches] + \
           [(b, "test") for b in a.test_batches]
    manifest, counts = [], {}
    for batch, split in plan:
        csv_path = Path(a.converted) / batch / "occlusion_labels.csv"
        if not csv_path.exists():
            print(f"!! missing {csv_path}, skipped"); continue
        rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
        cache, skipped, missing = {}, 0, set()
        seen, dups = set(), 0
        for r in rows:
            name = Path(r["image_name"]).name
            lvl = r["occlusion_level"].strip().lower()
            if lvl not in LEVELS:
                skipped += 1; continue
            if name not in cache:
                p = index.get(name)
                cache[name] = cv2.imread(str(p)) if p else None
                if p is None:
                    missing.add(name)
            img = cache[name]
            if img is None:
                continue
            x1, y1, x2, y2 = (float(r[k]) for k in ("xtl", "ytl", "xbr", "ybr"))
            if min(x2 - x1, y2 - y1) < a.min_side:
                skipped += 1; continue
            if a.dedupe and split != "test":
                g = a.dedupe_grid
                key = (round(x1 / g), round(y1 / g), round(x2 / g), round(y2 / g), lvl)
                if key in seen:
                    dups += 1; continue
                seen.add(key)
            crop = crop_square(img, x1, y1, x2, y2, a.pad, a.size)
            fn = f"{batch}__{Path(name).stem}__{r['box_index']}.jpg"
            cv2.imwrite(str(out / split / lvl / fn), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            manifest.append({"split": split, "batch": batch, "image": name, "box_index": r["box_index"],
                             "occlusion_level": lvl, "box_w": round(x2 - x1, 1), "box_h": round(y2 - y1, 1),
                             "file": f"{split}/{lvl}/{fn}"})
            counts[(split, lvl)] = counts.get((split, lvl), 0) + 1
        print(f"{batch:32s} -> {split:5s}  {len(rows)} boxes, {skipped} skipped"
              + (f", {dups} repeats removed" if a.dedupe and split != "test" else "")
              + (f", {len(missing)} frames not found" if missing else ""))

    with open(out / "crops_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        w.writeheader(); w.writerows(manifest)

    print(f"\n{'split':6s} {'low':>7s} {'medium':>7s} {'high':>7s} {'total':>7s}")
    for s in ("train", "val", "test"):
        c = [counts.get((s, l), 0) for l in LEVELS]
        print(f"{s:6s} {c[0]:7d} {c[1]:7d} {c[2]:7d} {sum(c):7d}")
    print(f"\nCrops written to {out}  (manifest: crops_manifest.csv)")


if __name__ == "__main__":
    main()
