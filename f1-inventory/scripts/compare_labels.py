r"""
After the agreement check is scored: show, for every relabelled carton, the original label, your new label
and the hybrid model's prediction, and make picture sheets of the cartons where you disagreed with yourself.

Usage (after agreement_check.py score):
  python scripts\compare_labels.py --out outputs\agreement ^
      --predictions outputs\occlusion\fusion\test_predictions.csv

Writes:
  <out>\comparison.csv            item_id, original, my_label, model, agree flags, note, image ...
  <out>\disagreements_01.jpg ...  12 cartons per sheet, caption: ORIGINAL / YOU NOW / MODEL
"""
import argparse, csv
from pathlib import Path

import cv2
import numpy as np

LEVELS = ["low", "medium", "high"]
SHORT = {"l": "low", "m": "medium", "h": "high", "low": "low", "medium": "medium", "med": "medium", "high": "high"}

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="outputs/agreement")
ap.add_argument("--predictions", default="outputs/occlusion/fusion/test_predictions.csv")
ap.add_argument("--cnn-share", type=float, default=0.5)
ap.add_argument("--per-sheet", type=int, default=12)
a = ap.parse_args()
out = Path(a.out)

key = {r["item_id"]: r for r in csv.DictReader(open(out / "key_DO_NOT_OPEN.csv", encoding="utf-8"))}
sheet = list(csv.DictReader(open(out / "relabel_sheet.csv", encoding="utf-8-sig")))

preds = {}
if a.predictions and Path(a.predictions).exists():
    for r in csv.DictReader(open(a.predictions, encoding="utf-8")):
        k = (r["batch"], Path(r["image"]).stem, str(r["box_index"]))
        cnn = np.array([float(r[f"cnn_{l}"]) for l in LEVELS])
        geo = np.array([float(r[f"geo_{l}"]) for l in LEVELS])
        preds[k] = LEVELS[int(np.argmax(a.cnn_share * cnn + (1 - a.cnn_share) * geo))]

rows = []
for s in sheet:
    it = s["item_id"]; k0 = key[it]
    mine = SHORT.get((s.get("my_label") or "").strip().lower(), "")
    model = preds.get((k0["batch"], Path(k0["image"]).stem, str(k0["box_index"])), "")
    rows.append({"item_id": it, "original": k0["original_label"], "my_label": mine, "model": model,
                 "you_agree_with_original": "yes" if mine == k0["original_label"] else "no",
                 "model_agrees_with_original": "yes" if model == k0["original_label"] else "no",
                 "model_agrees_with_you": "yes" if model == mine else "no",
                 "note": s.get("note", ""), "batch": k0["batch"], "image": k0["image"]})

with open(out / "comparison.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

order = {"low": 0, "medium": 1, "high": 2}
dis = [r for r in rows if r["you_agree_with_original"] == "no" and r["my_label"]]
dis.sort(key=lambda r: (order[r["original"]], order[r["my_label"]]))
print(f"{len(rows)} cartons; you changed {len(dis)}:")
for o in LEVELS:
    for m in LEVELS:
        n = sum(r["original"] == o and r["my_label"] == m for r in dis)
        if n:
            print(f"  original {o:6s} -> you now {m:6s}: {n}")


def tile(r, w=420):
    img = cv2.imread(str(out / "images" / f"{r['item_id']}.jpg"))
    if img is None:
        img = np.full((300, w, 3), 200, np.uint8)
    img = cv2.resize(img, (w, int(img.shape[0] * w / img.shape[1])))
    bar = np.full((54, w, 3), 255, np.uint8)
    cv2.putText(bar, r["item_id"], (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    txt = f"ORIGINAL {r['original'].upper()} | YOU NOW {r['my_label'].upper()} | MODEL {r['model'].upper() or '-'}"
    cv2.putText(bar, txt, (6, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 40, 20), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


for p in range(0, len(dis), a.per_sheet):
    tiles = [tile(r) for r in dis[p:p + a.per_sheet]]
    h = max(t.shape[0] for t in tiles)
    tiles = [np.vstack([t, np.full((h - t.shape[0], t.shape[1], 3), 255, np.uint8)]) for t in tiles]
    while len(tiles) % 3:
        tiles.append(np.full_like(tiles[0], 255))
    grid = np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)])
    name = out / f"disagreements_{p // a.per_sheet + 1:02d}.jpg"
    cv2.imwrite(str(name), grid, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    print(f"saved {name}")
print(f"saved {out / 'comparison.csv'}")
