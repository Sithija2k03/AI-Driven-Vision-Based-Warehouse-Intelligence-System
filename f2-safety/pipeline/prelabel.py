"""
Step 4. Draft labels with best.pt (forklift, load, worker). You MUST check
and correct them in Roboflow afterwards. Images with no detection get an
empty label file (they act as negatives only if you confirm they are empty).

Usage:  python prelabel.py FRAMES_DIR OUT_DIR [MODEL]
Output: OUT_DIR/images, OUT_DIR/labels, OUT_DIR/data.yaml
"""
import sys, shutil
from pathlib import Path
from ultralytics import YOLO

frames, out = Path(sys.argv[1]), Path(sys.argv[2])
model = YOLO(sys.argv[3] if len(sys.argv) > 3 else "best.pt")
(out / "images").mkdir(parents=True, exist_ok=True)
(out / "labels").mkdir(parents=True, exist_ok=True)

n_img = n_box = 0
for img in sorted(frames.glob("*.jpg")):
    r = model.predict(str(img), conf=0.25, imgsz=960, verbose=False)[0]
    lines = []
    for b in r.boxes:
        x, y, w, h = b.xywhn[0].tolist()
        lines.append(f"{int(b.cls[0])} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    shutil.copy(img, out / "images" / img.name)
    (out / "labels" / (img.stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
    n_img += 1; n_box += len(lines)

names = ["forklift", "load", "worker", "vehicle"]   # vehicle: draw by hand
(out / "data.yaml").write_text(
    f"path: {out.resolve()}\ntrain: images\nval: images\nnames:\n" +
    "".join(f"  {i}: {n}\n" for i, n in enumerate(names)))
print(f"{n_img} images, {n_box} draft boxes. Model classes: {model.names}")
print("Class 3 (vehicle) has no draft boxes: add them in Roboflow.")
