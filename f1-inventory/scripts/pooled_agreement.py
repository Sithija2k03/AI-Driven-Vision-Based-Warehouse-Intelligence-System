r"""
Pool several agreement-check rounds and print accuracy, Cohen's kappa and QWK (with 95% bootstrap CI)
for: human relabel vs original, and hybrid model vs original.

Usage:
  python scripts\pooled_agreement.py --outs outputs\agreement outputs\agreement_round2 ^
      --predictions outputs\occlusion\fusion\test_predictions.csv
"""
import argparse, csv
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score

LEVELS = ["low", "medium", "high"]
SHORT = {"l": "low", "m": "medium", "h": "high", "low": "low", "medium": "medium", "med": "medium", "high": "high"}

ap = argparse.ArgumentParser()
ap.add_argument("--outs", nargs="+", required=True, help="agreement folders (one per round)")
ap.add_argument("--predictions", default="outputs/occlusion/fusion/test_predictions.csv")
ap.add_argument("--cnn-share", type=float, default=0.5)
ap.add_argument("--boot", type=int, default=2000)
a = ap.parse_args()

preds = {}
for r in csv.DictReader(open(a.predictions, encoding="utf-8")):
    k = (r["batch"], Path(r["image"]).stem, str(r["box_index"]))
    p = a.cnn_share * np.array([float(r[f"cnn_{l}"]) for l in LEVELS]) + \
        (1 - a.cnn_share) * np.array([float(r[f"geo_{l}"]) for l in LEVELS])
    preds[k] = int(np.argmax(p))

orig, human, model = [], [], []
for folder in a.outs:
    out = Path(folder)
    key = {r["item_id"]: r for r in csv.DictReader(open(out / "key_DO_NOT_OPEN.csv", encoding="utf-8"))}
    n = 0
    for s in csv.DictReader(open(out / "relabel_sheet.csv", encoding="utf-8-sig")):
        v = SHORT.get((s.get("my_label") or "").strip().lower())
        k0 = key[s["item_id"]]
        k = (k0["batch"], Path(k0["image"]).stem, str(k0["box_index"]))
        if v is None or k not in preds:
            continue
        orig.append(LEVELS.index(k0["original_label"])); human.append(LEVELS.index(v)); model.append(preds[k]); n += 1
    print(f"{folder}: {n} cartons")
orig, human, model = map(np.array, (orig, human, model))
rng = np.random.default_rng(0)


def stats(y1, y2):
    acc = float((y1 == y2).mean())
    k = cohen_kappa_score(y1, y2)
    q = cohen_kappa_score(y1, y2, weights="quadratic")
    ks, qs = [], []
    for _ in range(a.boot):
        i = rng.integers(0, len(y1), len(y1))
        ks.append(cohen_kappa_score(y1[i], y2[i])); qs.append(cohen_kappa_score(y1[i], y2[i], weights="quadratic"))
    return acc, k, np.percentile(ks, [2.5, 97.5]), q, np.percentile(qs, [2.5, 97.5])


print(f"\nPOOLED over {len(orig)} cartons")
print(f"{'comparison':22s} {'accuracy':>8s} {'kappa':>6s} {'kappa 95% CI':>14s} {'QWK':>6s} {'QWK 95% CI':>14s}")
for name, y in (("human vs original", human), ("model vs original", model)):
    acc, k, kci, q, qci = stats(orig, y)
    print(f"{name:22s} {acc:8.3f} {k:6.3f} {kci[0]:6.3f}-{kci[1]:.3f} {q:6.3f} {qci[0]:6.3f}-{qci[1]:.3f}")
