r"""
SIMULATED carton removal using your annotations + inpainting (more realistic than a grey block).

For one location, it picks some annotated cartons (boxes whose centre is in the region),
erases them with OpenCV inpainting (fills from the surrounding texture), and writes:
  - the edited image   (name contains SIMULATED, stamped "SIMULATED" in the corner)
  - a matching label file with the REMAINING boxes, so count_stock.py --labels-dir can
    print the true count after removal
  - a small JSON log: how many boxes were in the region before, how many removed, how many left

Then run count_stock.py on the edited image and check: did the detected count drop by
about the number removed?

Usage (from the f1-inventory folder):
  python scripts\simulate_removal.py --config configs\locations.json --camera-id X_5 ^
      --frame "D:\...\cam12__frame_00009.jpg" ^
      --labels annotations\converted\location_X_5_b2\labels\cam12__frame_00009.txt ^
      --location X5-RIGHT-RACK --fraction 0.5 --seed 1 --out-dir outputs\simulated_removal

  --fraction 0.5  -> remove 50% of the annotated cartons in that location
  --remove 30     -> remove exactly 30 cartons instead (overrides --fraction)

Limits (say these honestly): inpainting can leave smudges, and where a removed carton
overlaps a kept carton, the kept carton is protected, so small remnants of the removed one
can stay. It is still a simulation, not a real empty shelf.
"""
import argparse, json, random
from pathlib import Path
import cv2
import numpy as np


def read_yolo(path, w, h):
    boxes = []
    for line in Path(path).read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        cx, cy, bw, bh = (float(v) for v in p[1:5])
        boxes.append({"line": line.strip(), "cx": cx, "cy": cy,
                      "x1": int((cx - bw / 2) * w), "y1": int((cy - bh / 2) * h),
                      "x2": int((cx + bw / 2) * w), "y2": int((cy + bh / 2) * h)})
    return boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--camera-id", required=True)
    ap.add_argument("--frame", required=True)
    ap.add_argument("--labels", required=True, help="YOLO .txt label file for this frame")
    ap.add_argument("--location", required=True)
    ap.add_argument("--fraction", type=float, default=0.5)
    ap.add_argument("--remove", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--pad", type=int, default=2, help="pixels added around each erased box")
    ap.add_argument("--out-dir", default="outputs/simulated_removal")
    a = ap.parse_args()

    cfg = json.load(open(a.config, encoding="utf-8"))
    locs = {l["location_id"]: l for l in cfg["cameras"][a.camera_id]["locations"]}
    if a.location not in locs:
        raise SystemExit(f"Unknown location '{a.location}'. Options: {', '.join(locs)}")
    rx1, ry1, rx2, ry2 = locs[a.location]["region"]

    img = cv2.imread(a.frame)
    if img is None:
        raise SystemExit(f"Cannot read image: {a.frame}")
    h, w = img.shape[:2]
    boxes = read_yolo(a.labels, w, h)
    inside = [b for b in boxes if rx1 <= b["cx"] <= rx2 and ry1 <= b["cy"] <= ry2]
    if not inside:
        raise SystemExit("No annotated boxes in that location.")

    n = a.remove if a.remove is not None else round(len(inside) * a.fraction)
    n = max(0, min(n, len(inside)))
    random.seed(a.seed)
    removed = random.sample(inside, n)
    removed_ids = {id(b) for b in removed}
    kept = [b for b in boxes if id(b) not in removed_ids]

    # mask = removed boxes, minus kept boxes (so cartons we keep are not erased)
    mask = np.zeros((h, w), np.uint8)
    for b in removed:
        cv2.rectangle(mask, (b["x1"] - a.pad, b["y1"] - a.pad), (b["x2"] + a.pad, b["y2"] + a.pad), 255, -1)
    for b in kept:
        cv2.rectangle(mask, (b["x1"], b["y1"]), (b["x2"], b["y2"]), 0, -1)
    out_img = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)

    pct = int(round(100 * n / len(inside)))
    txt = f"SIMULATED: {n} of {len(inside)} cartons removed from {a.location}"
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(out_img, (4, h - th - 14), (tw + 12, h - 4), (0, 0, 0), -1)
    cv2.putText(out_img, txt, (8, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    out = Path(a.out_dir)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    stem = f"{Path(a.frame).stem}__SIMULATED_REMOVE_{a.location}_{pct}"
    cv2.imwrite(str(out / f"{stem}.jpg"), out_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    (out / "labels" / f"{stem}.txt").write_text("\n".join(b["line"] for b in kept) + "\n")
    log = {"frame": Path(a.frame).name, "location": a.location, "seed": a.seed,
           "boxes_in_location_before": len(inside), "removed": n,
           "boxes_in_location_after": len(inside) - n, "image": f"{stem}.jpg"}
    json.dump(log, open(out / f"{stem}.json", "w"), indent=2)
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
