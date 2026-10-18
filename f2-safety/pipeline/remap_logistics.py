"""
Pick vehicle images from the Roboflow Logistics dataset and rewrite
their labels to the F2 class order:
  0 forklift, 1 load, 2 worker, 3 vehicle

Usage:
  python remap_logistics.py SRC_DIR OUT_DIR [MAX_IMAGES] [SEED]

SRC_DIR = the unzipped Roboflow export (YOLOv8 format, has data.yaml).
Class numbers are read from data.yaml by NAME, so the export order
does not matter.
"""
import sys, random, shutil
from pathlib import Path
import yaml

src = Path(sys.argv[1])
out = Path(sys.argv[2])
max_n = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42

NEW_NAMES = ["forklift", "load", "worker", "vehicle"]
NAME_MAP = {"forklift": 0, "person": 2, "car": 3, "truck": 3, "van": 3}
MUST_HAVE = {"car", "truck", "van"}   # keep images with at least one of these

cfg = yaml.safe_load(open(src / "data.yaml"))
names = cfg["names"]
if isinstance(names, dict):
    names = [names[k] for k in sorted(names)]
old = {i: n.strip().lower() for i, n in enumerate(names)}
missing = [n for n in NAME_MAP if n not in old.values()]
if missing:
    sys.exit(f"Class names not found in data.yaml: {missing}\nFound: {sorted(old.values())}")

# collect (image, label) pairs from every split
pairs = []
for lab in src.glob("*/labels/*.txt"):
    for ext in (".jpg", ".jpeg", ".png"):
        img = lab.parent.parent / "images" / (lab.stem + ext)
        if img.exists():
            pairs.append((img, lab)); break

def convert(lab):
    rows, has = [], set()
    for line in lab.read_text().splitlines():
        p = line.split()
        if len(p) != 5:            # skip polygons / bad lines
            continue
        n = old.get(int(float(p[0])))
        if n in NAME_MAP:
            rows.append(f"{NAME_MAP[n]} {' '.join(p[1:])}")
            has.add(n)
    return rows, has

cand = []
for img, lab in pairs:
    rows, has = convert(lab)
    if has & MUST_HAVE:
        cand.append((img, rows))

random.seed(seed)
random.shuffle(cand)
cand = cand[:max_n]

(out / "images").mkdir(parents=True, exist_ok=True)
(out / "labels").mkdir(parents=True, exist_ok=True)
counts = [0] * len(NEW_NAMES)
for img, rows in cand:
    stem = "log_" + img.stem
    shutil.copy(img, out / "images" / (stem + img.suffix))
    (out / "labels" / (stem + ".txt")).write_text("\n".join(rows) + "\n")
    for r in rows:
        counts[int(r.split()[0])] += 1

print(f"images found: {len(pairs)}, with car/truck/van: {len(cand)} (cap {max_n})")
for i, n in enumerate(NEW_NAMES):
    print(f"  class {i} {n}: {counts[i]} boxes")
print("Classes dropped on purpose: helmet, fire, smoke, traffic light, license plate,")
print("barcode, qr code, ladder, freight container, road sign, safety vest, wood pallet,")
print("cardboard box, gloves, traffic cone")
