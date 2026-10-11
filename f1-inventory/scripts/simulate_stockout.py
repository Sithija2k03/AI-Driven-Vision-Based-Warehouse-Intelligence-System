r"""
Make a SIMULATED stock-removal copy of a frame, to demonstrate LOW / STOCK-OUT
(the real CCTV frames only show full stock).

It paints over part of one location's region with a flat colour (the region's average
colour), so the detector sees "no cartons" there. The output image is stamped
"SIMULATED" and the file name contains SIMULATED, so it can never be mistaken for real data.

Usage (from the f1-inventory folder):
  python scripts\simulate_stockout.py --config configs\locations.json --camera-id X_5 ^
      --frame "D:\...\cam12__frame_00009.jpg" --location X5-RIGHT-RACK --fraction 1.0 ^
      --out-dir outputs\simulated

--fraction 1.0  -> the whole region is cleared  (expected: STOCK-OUT)
--fraction 0.6  -> 60% of the region is cleared (expected: LOW)
The cleared part starts at the region's left edge and covers that share of its width.
"""
import argparse, json
from pathlib import Path
import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--camera-id", required=True)
    ap.add_argument("--frame", required=True)
    ap.add_argument("--location", required=True, help="location_id from the config")
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--out-dir", default="outputs/simulated")
    a = ap.parse_args()

    cfg = json.load(open(a.config, encoding="utf-8"))
    locs = {l["location_id"]: l for l in cfg["cameras"][a.camera_id]["locations"]}
    if a.location not in locs:
        raise SystemExit(f"Unknown location '{a.location}'. Options: {', '.join(locs)}")

    img = cv2.imread(a.frame)
    if img is None:
        raise SystemExit(f"Cannot read image: {a.frame}")
    h, w = img.shape[:2]
    x1, y1, x2, y2 = locs[a.location]["region"]
    px1, py1, px2, py2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
    cut = px1 + int((px2 - px1) * max(0.0, min(1.0, a.fraction)))

    mean = cv2.mean(img[py1:py2, px1:px2])[:3]
    img[py1:py2, px1:cut] = [int(c) for c in mean]

    txt = f"SIMULATED: {int(a.fraction * 100)}% of {a.location} cleared"
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (4, h - th - 14), (tw + 12, h - 4), (0, 0, 0), -1)
    cv2.putText(img, txt, (8, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = f"{Path(a.frame).stem}__SIMULATED_{a.location}_{int(a.fraction * 100)}.jpg"
    cv2.imwrite(str(out / name), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    print(f"Wrote {out / name}")


if __name__ == "__main__":
    main()
