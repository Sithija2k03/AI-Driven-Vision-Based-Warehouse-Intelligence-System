r"""
F1 sub-function 2, step 4: label agreement check for the occlusion levels.

Question: how consistent are the occlusion labels themselves?
If a person relabelling the same cartons only agrees with the original labels ~65% of the time,
then a model scoring ~60% is close to the ceiling the labels allow.

Two modes:

1) sample  - pick N cartons from the day-2 test batches (balanced across low/medium/high, one per
             physical carton), save a review image for each (whole frame + zoom, box drawn in yellow),
             and write a blank sheet to fill in. The original labels go to a separate key file
             that you must NOT open until you finish.

   python scripts\agreement_check.py sample --frames-root "D:\Research\research camera data" ^
       --out outputs\agreement

2) score   - after filling the my_label column (low / medium / high, or l / m / h):

   python scripts\agreement_check.py score --out outputs\agreement ^
       --predictions outputs\occlusion\fusion\test_predictions.csv

   Prints human-vs-original agreement (accuracy, Cohen's kappa, quadratic-weighted kappa,
   within-one, low<->high swaps, confusion matrix) and, if --predictions is given, the hybrid
   model (0.5 CNN + 0.5 geometry) against both label sets on the same cartons.
   Writes <out>/agreement_metrics.json.
"""
import argparse, csv, json, random
from pathlib import Path

import numpy as np

LEVELS = ["low", "medium", "high"]
DAY2 = ["location_G_H_b2", "location_X_5_b2", "location_checkingPart_2_b2"]
SHORT = {"l": "low", "m": "medium", "h": "high", "low": "low", "medium": "medium", "med": "medium", "high": "high"}


# ---------------------------------------------------------------- sample
def review_image(img, x1, y1, x2, y2, panel=512):
    import cv2
    h, w = img.shape[:2]
    # left: whole frame with the box marked
    full = img.copy()
    t = max(1, round(max(h, w) / 320))
    cv2.rectangle(full, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), t)
    full = cv2.resize(full, (panel, panel), interpolation=cv2.INTER_AREA)
    # right: zoom around the box (box + 1x its size on each side)
    bw, bh = x2 - x1, y2 - y1
    side = max(bw, bh) * 3
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    sx1, sy1 = int(round(cx - side / 2)), int(round(cy - side / 2))
    sx2, sy2 = sx1 + int(round(side)), sy1 + int(round(side))
    canvas = np.full((sy2 - sy1, sx2 - sx1, 3), 114, np.uint8)
    ix1, iy1, ix2, iy2 = max(sx1, 0), max(sy1, 0), min(sx2, w), min(sy2, h)
    canvas[iy1 - sy1:iy2 - sy1, ix1 - sx1:ix2 - sx1] = img[iy1:iy2, ix1:ix2]
    s = panel / canvas.shape[0]
    zoom = cv2.resize(canvas, (panel, panel), interpolation=cv2.INTER_CUBIC)
    cv2.rectangle(zoom, (int((x1 - sx1) * s), int((y1 - sy1) * s)),
                  (int((x2 - sx1) * s), int((y2 - sy1) * s)), (0, 255, 255), 2)
    sep = np.full((panel, 6, 3), 255, np.uint8)
    return np.hstack([full, sep, zoom])


def sample(a):
    import cv2
    out = Path(a.out)
    img_dir = out / "images"
    if (out / "relabel_sheet.csv").exists() and not a.overwrite:
        raise SystemExit(f"{out / 'relabel_sheet.csv'} already exists. Use --overwrite to start again "
                         "(this deletes your filled labels).")
    img_dir.mkdir(parents=True, exist_ok=True)

    index = {}
    for p in Path(a.frames_root).rglob("*"):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png") and p.parent.name == "extracted_frames":
            index.setdefault(p.name, p)

    # one candidate per physical carton (propagated labels repeat the same box in many frames)
    pools = {l: [] for l in LEVELS}
    seen = set()
    rng = random.Random(a.seed)
    for batch in a.batches:
        csv_path = Path(a.converted) / batch / "occlusion_labels.csv"
        if not csv_path.exists():
            print(f"!! missing {csv_path}, skipped"); continue
        rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
        rng.shuffle(rows)
        for r in rows:
            lvl = r["occlusion_level"].strip().lower()
            name = Path(r["image_name"]).name
            if lvl not in LEVELS or name not in index:
                continue
            x1, y1, x2, y2 = (float(r[k]) for k in ("xtl", "ytl", "xbr", "ybr"))
            if min(x2 - x1, y2 - y1) < a.min_side:
                continue
            g = a.dedupe_grid
            key = (batch, round(x1 / g), round(y1 / g), round(x2 / g), round(y2 / g))
            if key in seen:
                continue
            seen.add(key)
            pools[lvl].append({"batch": batch, "image": name, "box_index": r["box_index"],
                               "original_label": lvl, "xtl": x1, "ytl": y1, "xbr": x2, "ybr": y2})

    per = a.n // 3
    chosen = []
    for l in LEVELS:
        if len(pools[l]) < per:
            print(f"!! only {len(pools[l])} unique '{l}' cartons available")
        chosen += rng.sample(pools[l], min(per, len(pools[l])))
    rng.shuffle(chosen)  # mixed order so the level can't be guessed from position

    cache = {}
    for i, c in enumerate(chosen, 1):
        c["item_id"] = f"item_{i:03d}"
        if c["image"] not in cache:
            cache[c["image"]] = cv2.imread(str(index[c["image"]]))
        view = review_image(cache[c["image"]], c["xtl"], c["ytl"], c["xbr"], c["ybr"])
        cv2.imwrite(str(img_dir / f"{c['item_id']}.jpg"), view, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    with open(out / "relabel_sheet.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "my_label", "note"])
        for c in chosen:
            w.writerow([c["item_id"], "", ""])
    with open(out / "key_DO_NOT_OPEN.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["item_id", "batch", "image", "box_index", "original_label",
                                          "xtl", "ytl", "xbr", "ybr"])
        w.writeheader(); w.writerows(chosen)

    counts = {l: sum(c["original_label"] == l for c in chosen) for l in LEVELS}
    print(f"Unique cartons available: " + ", ".join(f"{l} {len(pools[l])}" for l in LEVELS))
    print(f"Sampled {len(chosen)} cartons {counts} from {', '.join(a.batches)}")
    print(f"Review images : {img_dir}  (left = whole frame, right = zoom; the carton is in the yellow box)")
    print(f"Fill in       : {out / 'relabel_sheet.csv'}  column my_label = low / medium / high (or l / m / h)")
    print(f"Do NOT open   : {out / 'key_DO_NOT_OPEN.csv'}")


# ---------------------------------------------------------------- score
def metrics(y, p):
    from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
    y, p = np.asarray(y), np.asarray(p)
    return {
        "n": int(len(y)),
        "accuracy": round(float(accuracy_score(y, p)), 3),
        "macro_f1": round(float(f1_score(y, p, average="macro", labels=[0, 1, 2], zero_division=0)), 3),
        "cohen_kappa": round(float(cohen_kappa_score(y, p, labels=[0, 1, 2])), 3),
        "quadratic_weighted_kappa": round(float(cohen_kappa_score(y, p, labels=[0, 1, 2], weights="quadratic")), 3),
        "within_one_accuracy": round(float(np.mean(np.abs(y - p) <= 1)), 3),
        "low_vs_high_swaps": int(np.sum(np.abs(y - p) == 2)),
        "confusion_matrix_rows_first_cols_second": confusion_matrix(y, p, labels=[0, 1, 2]).tolist(),
    }


def bootstrap_kappa(y, p, n=2000, seed=0):
    from sklearn.metrics import cohen_kappa_score
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    ks = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        ks.append(cohen_kappa_score(y[i], p[i], labels=[0, 1, 2]))
    return [round(float(np.percentile(ks, 2.5)), 3), round(float(np.percentile(ks, 97.5)), 3)]


def show(title, m, ci=None, row_name="first"):
    print(f"\n{title}  (n={m['n']})")
    print(f"  accuracy {m['accuracy']:.3f} | macro-F1 {m['macro_f1']:.3f} | kappa {m['cohen_kappa']:.3f}"
          + (f" [95% CI {ci[0]:.3f}-{ci[1]:.3f}]" if ci else "")
          + f" | QWK {m['quadratic_weighted_kappa']:.3f} | within-one {m['within_one_accuracy']:.3f}"
          f" | low<->high {m['low_vs_high_swaps']}")
    print(f"  confusion (rows = {row_name}, cols = second; order low/medium/high):")
    for r in m["confusion_matrix_rows_first_cols_second"]:
        print("   ", r)


def score(a):
    out = Path(a.out)
    key = {r["item_id"]: r for r in csv.DictReader(open(out / "key_DO_NOT_OPEN.csv", encoding="utf-8"))}
    sheet = list(csv.DictReader(open(out / "relabel_sheet.csv", encoding="utf-8-sig")))
    items, orig, mine, bad = [], [], [], []
    for r in sheet:
        v = (r.get("my_label") or "").strip().lower()
        if v not in SHORT:
            bad.append(r["item_id"]); continue
        items.append(r["item_id"])
        orig.append(LEVELS.index(key[r["item_id"]]["original_label"]))
        mine.append(LEVELS.index(SHORT[v]))
    if bad:
        print(f"!! {len(bad)} rows without a valid label were left out: {', '.join(bad[:10])}"
              + (" ..." if len(bad) > 10 else ""))
    if len(items) < 10:
        raise SystemExit("Fewer than 10 labelled rows. Fill in my_label first.")

    res = {"human_vs_original": metrics(orig, mine)}
    res["human_vs_original"]["kappa_95ci"] = bootstrap_kappa(orig, mine)
    show("HUMAN relabel vs ORIGINAL labels", res["human_vs_original"],
         res["human_vs_original"]["kappa_95ci"], "original")

    if a.predictions:
        preds = {}
        for r in csv.DictReader(open(a.predictions, encoding="utf-8")):
            k = (r["batch"], Path(r["image"]).stem, str(r["box_index"]))
            cnn = np.array([float(r[f"cnn_{l}"]) for l in LEVELS])
            geo = np.array([float(r[f"geo_{l}"]) for l in LEVELS])
            preds[k] = int(np.argmax(a.cnn_share * cnn + (1 - a.cnn_share) * geo))
        o2, m2, h2 = [], [], []
        for it, yo, ym in zip(items, orig, mine):
            k0 = key[it]
            k = (k0["batch"], Path(k0["image"]).stem, str(k0["box_index"]))
            if k in preds:
                o2.append(yo); m2.append(ym); h2.append(preds[k])
        print(f"\nHybrid predictions found for {len(h2)} of {len(items)} items")
        if len(h2) >= 10:
            res["model_vs_original"] = metrics(o2, h2)
            res["model_vs_original"]["kappa_95ci"] = bootstrap_kappa(o2, h2)
            res["model_vs_human"] = metrics(m2, h2)
            res["model_vs_human"]["kappa_95ci"] = bootstrap_kappa(m2, h2)
            show("HYBRID MODEL vs ORIGINAL labels", res["model_vs_original"],
                 res["model_vs_original"]["kappa_95ci"], "original")
            show("HYBRID MODEL vs HUMAN relabel", res["model_vs_human"],
                 res["model_vs_human"]["kappa_95ci"], "human relabel")

    k = res["human_vs_original"]["cohen_kappa"]
    band = ("slight" if k < 0.21 else "fair" if k < 0.41 else "moderate" if k < 0.61
            else "substantial" if k < 0.81 else "almost perfect")
    print(f"\nHuman agreement kappa {k:.3f} = '{band}' (Landis & Koch). "
          "Compare the model's kappa with this number: it is the practical ceiling.")
    json.dump(res, open(out / "agreement_metrics.json", "w"), indent=2)
    print(f"Saved {out / 'agreement_metrics.json'}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--frames-root", required=True)
    s.add_argument("--converted", default="annotations/converted")
    s.add_argument("--out", default="outputs/agreement")
    s.add_argument("--batches", nargs="+", default=DAY2)
    s.add_argument("--n", type=int, default=102, help="total cartons (split equally across the 3 levels)")
    s.add_argument("--min-side", type=int, default=6)
    s.add_argument("--dedupe-grid", type=float, default=8.0)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--overwrite", action="store_true")
    c = sub.add_parser("score")
    c.add_argument("--out", default="outputs/agreement")
    c.add_argument("--predictions", help="outputs/occlusion/fusion/test_predictions.csv")
    c.add_argument("--cnn-share", type=float, default=0.5)
    a = ap.parse_args()
    sample(a) if a.mode == "sample" else score(a)


if __name__ == "__main__":
    main()
