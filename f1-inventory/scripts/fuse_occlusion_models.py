r"""
F1 sub-function 2, step 2d: combine the CNN (appearance) and the random forest (geometry) occlusion models.

For every val and test carton it gets class probabilities from both models, then averages them:
    p = w * p_cnn + (1 - w) * p_geometry
The weight w is chosen on the VAL set only (by macro-F1); the TEST set is scored once with that w.
It also reports each model alone (re-computed here) and a plain 50/50 average, so the comparison is fair.

Runs on your PC (CPU is fine: it only predicts, it does not train the CNN). Needs:
  - the crops folder from make_occlusion_crops.py (val/ and test/)
  - the trained CNN weights (best.pt from the run-B folder on Drive), e.g. weights\occlusion_r18.pt
  - annotations\converted\<batch>\occlusion_labels.csv (for the geometry model)

Usage (from the f1-inventory folder):
  python scripts\fuse_occlusion_models.py --crops "D:\Research\dataset\f1_occlusion_crops" ^
      --cnn-weights weights\occlusion_r18.pt --out outputs\occlusion\fusion
"""
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix, classification_report

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geometry_occlusion_rf import frame_features, LEVELS, DAY1, VAL, DAY2  # noqa: E402


def parse_crop_name(fn):
    """'<batch>__<image stem>__<box_index>.jpg' -> (batch, image stem, box_index). Image stems may contain '__'."""
    parts = Path(fn).stem.split("__")
    return parts[0], "__".join(parts[1:-1]), parts[-1]


def geometry_table(converted, batches, img_w, img_h, min_side):
    """Returns dict key -> (features, label). key = (batch, image stem, box_index)."""
    out = {}
    for batch in batches:
        p = Path(converted) / batch / "occlusion_labels.csv"
        per_img = {}
        for r in csv.DictReader(open(p, encoding="utf-8")):
            per_img.setdefault(r["image_name"], []).append(r)
        for name, rows in per_img.items():
            boxes = np.array([[float(r[k]) for k in ("xtl", "ytl", "xbr", "ybr")] for r in rows])
            F = frame_features(boxes, img_w, img_h)
            for r, f, bx in zip(rows, F, boxes):
                lvl = r["occlusion_level"].strip().lower()
                if lvl not in LEVELS or min(bx[2] - bx[0], bx[3] - bx[1]) < min_side:
                    continue
                out[(batch, Path(name).stem, str(r["box_index"]))] = (f, LEVELS.index(lvl))
    return out


def cnn_probs(crop_dir, weights, size, bs):
    """Returns dict key -> (prob[3], label) for every crop in crop_dir/<level>/*.jpg."""
    import torch, torch.nn as nn
    from torchvision import models, transforms
    from PIL import Image
    tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 3)
    m.load_state_dict(torch.load(weights, map_location="cpu"))
    m.eval()
    items = [(p, LEVELS.index(p.parent.name)) for lvl in LEVELS for p in sorted((Path(crop_dir) / lvl).glob("*.jpg"))]
    out = {}
    with torch.no_grad():
        for i in range(0, len(items), bs):
            chunk = items[i:i + bs]
            x = torch.stack([tf(Image.open(p).convert("RGB")) for p, _ in chunk])
            pr = torch.softmax(m(x), 1).numpy()
            for (p, y), q in zip(chunk, pr):
                out[parse_crop_name(p.name)] = (q, y)
            print(f"\r  {Path(crop_dir).name}: {min(i + bs, len(items))}/{len(items)} crops", end="")
    print()
    return out


def metrics(y, p):
    return {"accuracy": round(float(accuracy_score(y, p)), 4),
            "macro_f1": round(float(f1_score(y, p, average="macro")), 4),
            "cohen_kappa": round(float(cohen_kappa_score(y, p)), 4),
            "quadratic_weighted_kappa": round(float(cohen_kappa_score(y, p, weights="quadratic")), 4),
            "within_one_accuracy": round(float(np.mean(np.abs(y - p) <= 1)), 4),
            "low_vs_high_confusions": int(np.sum(np.abs(y - p) == 2)),
            "confusion_matrix_rows_true_cols_pred": confusion_matrix(y, p, labels=[0, 1, 2]).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", required=True)
    ap.add_argument("--cnn-weights", required=True)
    ap.add_argument("--converted", default="annotations/converted")
    ap.add_argument("--out", default="outputs/occlusion/fusion")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--img-w", type=int, default=640)
    ap.add_argument("--img-h", type=int, default=640)
    ap.add_argument("--min-side", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    print("Geometry features ...")
    g_tr = geometry_table(a.converted, DAY1, a.img_w, a.img_h, a.min_side)
    g_va = geometry_table(a.converted, VAL, a.img_w, a.img_h, a.min_side)
    g_te = geometry_table(a.converted, DAY2, a.img_w, a.img_h, a.min_side)
    rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=5, class_weight="balanced",
                                n_jobs=-1, random_state=a.seed)
    rf.fit(np.array([v[0] for v in g_tr.values()]), np.array([v[1] for v in g_tr.values()]))

    print("CNN predictions (CPU) ...")
    c_va = cnn_probs(Path(a.crops) / "val", a.cnn_weights, a.size, a.bs)
    c_te = cnn_probs(Path(a.crops) / "test", a.cnn_weights, a.size, a.bs)

    def aligned(g, c):
        keys = [k for k in c if k in g]
        if len(keys) < len(c):
            print(f"  note: {len(c) - len(keys)} crops had no matching geometry row and were left out")
        pc = np.array([c[k][0] for k in keys])
        pg = rf.predict_proba(np.array([g[k][0] for k in keys]))
        y = np.array([c[k][1] for k in keys])
        return keys, y, pc, pg

    kv, yv, pcv, pgv = aligned(g_va, c_va)
    kt, yt, pct, pgt = aligned(g_te, c_te)

    # choose the weight on VAL only
    grid = [round(w, 1) for w in np.arange(0, 1.01, 0.1)]
    val_f1 = {w: f1_score(yv, (w * pcv + (1 - w) * pgv).argmax(1), average="macro") for w in grid}
    w_best = max(val_f1, key=val_f1.get)

    rows = {"CNN only (ResNet18)": pct.argmax(1),
            "Geometry only (random forest)": pgt.argmax(1),
            "Fusion 50/50": (0.5 * pct + 0.5 * pgt).argmax(1),
            f"Fusion w={w_best} (chosen on val)": (w_best * pct + (1 - w_best) * pgt).argmax(1)}
    res = {"n_val": len(yv), "n_test": len(yt), "val_macro_f1_by_weight": {str(k): round(float(v), 4) for k, v in val_f1.items()},
           "chosen_weight_cnn": w_best, "test": {name: metrics(yt, p) for name, p in rows.items()}}

    print(f"\nval macro-F1 by CNN weight w: " + "  ".join(f"{w}:{val_f1[w]:.3f}" for w in grid))
    print(f"chosen w (CNN share) = {w_best}\n")
    print(f"=== TEST (day-2, {len(yt)} cartons) ===")
    print(f"{'model':36s} {'acc':>6s} {'mF1':>6s} {'kappa':>6s} {'QWK':>6s} {'within1':>8s} {'low<->high':>10s}")
    for name, m in res["test"].items():
        print(f"{name:36s} {m['accuracy']:6.3f} {m['macro_f1']:6.3f} {m['cohen_kappa']:6.3f} "
              f"{m['quadratic_weighted_kappa']:6.3f} {m['within_one_accuracy']:8.3f} {m['low_vs_high_confusions']:10d}")
    best_name = f"Fusion w={w_best} (chosen on val)"
    print(f"\n{best_name}:")
    print(classification_report(yt, rows[best_name], target_names=LEVELS, digits=3))
    print("confusion matrix (rows = true low/medium/high, cols = predicted):")
    for r in res["test"][best_name]["confusion_matrix_rows_true_cols_pred"]:
        print("   ", r)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / "metrics.json", "w"), indent=2)
    with open(out / "test_predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["batch", "image", "box_index", "true", "cnn_low", "cnn_medium", "cnn_high",
                    "geo_low", "geo_medium", "geo_high", "fused_pred"])
        fused = (w_best * pct + (1 - w_best) * pgt)
        for k, y, c, g, fp in zip(kt, yt, pct, pgt, fused.argmax(1)):
            w.writerow([*k, LEVELS[y], *[round(float(v), 4) for v in c], *[round(float(v), 4) for v in g], LEVELS[fp]])
    print(f"\nSaved {out / 'metrics.json'} and {out / 'test_predictions.csv'}")


if __name__ == "__main__":
    main()
