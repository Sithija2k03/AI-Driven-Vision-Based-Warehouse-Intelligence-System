r"""
Independent realism tests for synthetic carton images (second line of evidence, next to the detector realism check).

All tests compare small carton crops (same crop style as the occlusion classifier: 25% context, square, 128 px),
balanced over low / medium / high occlusion, described by ImageNet ResNet18 features (512 numbers per crop).

  1. Real-vs-fake classifier test (C2ST, Lopez-Paz & Oquab, ICLR 2017)
       A simple classifier (logistic regression, 5-fold CV, frames kept together) tries to tell the two sets apart.
       ~0.50 accuracy = cannot tell them apart (good); ~1.00 = obviously different.
  2. KID (Binkowski et al., ICLR 2018): kernel distance between the two feature sets, unbiased for small samples.
       0 = same distribution; reported x1000. Lower is better.
  3. Precision / recall for generated data (Kynkaenniemi et al., NeurIPS 2019), k-NN manifolds (k=3):
       precision = share of synthetic crops that fall inside the real-crop manifold (realism),
       recall    = share of real crops covered by the synthetic manifold (variety).

Each test is also run REAL day-1 vs REAL day-2 as a baseline: that shows what "as similar as real data gets"
looks like (different day, same cameras), so the synthetic numbers can be read against it.

Usage (f1-inventory environment):
  python scripts\synthetic_validation.py --real-crops D:\Research\dataset\f1_occlusion_crops ^
      --synth D:\Research\synth3d_trial\synth_500 --out outputs\synthetic_validation
  Several synthetic sets (e.g. before / after CUT) can be compared:  --synth A_dir B_dir --names raw cut
"""
import argparse, csv, json, random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

LEVELS = ["low", "medium", "high"]


def crop_square(img, x1, y1, x2, y2, pad=0.25, size=128):
    h, w = img.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(bw, bh) * (1 + 2 * pad)
    sx1, sy1 = int(round(cx - side / 2)), int(round(cy - side / 2))
    sx2, sy2 = int(round(cx + side / 2)), int(round(cy + side / 2))
    canvas = np.full((max(sy2 - sy1, 1), max(sx2 - sx1, 1), 3), 114, np.uint8)
    ix1, iy1, ix2, iy2 = max(sx1, 0), max(sy1, 0), min(sx2, w), min(sy2, h)
    if ix2 > ix1 and iy2 > iy1:
        canvas[iy1 - sy1:iy2 - sy1, ix1 - sx1:ix2 - sx1] = img[iy1:iy2, ix1:ix2]
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)


def real_set(root, split, n, rng):
    """n crops per level from <root>/<split>/<level>; group = source frame (from crops_manifest.csv if present)."""
    root = Path(root)
    groups = {}
    man = root / "crops_manifest.csv"
    if man.exists():
        for r in csv.DictReader(open(man, encoding="utf-8")):
            groups[Path(r["file"]).name] = f"{r.get('batch', '')}/{r.get('image', '')}"
    crops, grp, lvl = [], [], []
    for l in LEVELS:
        files = sorted((root / split / l).glob("*.jpg"))
        rng.shuffle(files)
        for f in files[:n]:
            im = cv2.imread(str(f))
            if im is None:
                continue
            crops.append(cv2.resize(im, (128, 128))); grp.append(groups.get(f.name, f.stem.split("_")[0])); lvl.append(l)
        if len(files) < n:
            print(f"  !! real {split}/{l}: only {len(files)} crops")
    return crops, grp, lvl


def synth_set(folder, n, rng, min_side=6):
    folder = Path(folder)
    rows = [r for r in csv.DictReader(open(folder / "occlusion_labels.csv", encoding="utf-8"))
            if str(r.get("labelled", "1")) == "1" and r["occlusion_level"] in LEVELS]
    by = defaultdict(list)
    for r in rows:
        if min(float(r["xbr"]) - float(r["xtl"]), float(r["ybr"]) - float(r["ytl"])) >= min_side:
            by[r["occlusion_level"]].append(r)
    cache, crops, grp, lvl = {}, [], [], []
    for l in LEVELS:
        pick = rng.sample(by[l], min(n, len(by[l])))
        for r in pick:
            name = r["image_name"]
            if name not in cache:
                cache[name] = cv2.imread(str(folder / "images" / name))
            im = cache[name]
            if im is None:
                continue
            crops.append(crop_square(im, *(float(r[k]) for k in ("xtl", "ytl", "xbr", "ybr"))))
            grp.append(name); lvl.append(l)
        if len(by[l]) < n:
            print(f"  !! synthetic {l}: only {len(by[l])} cartons")
    return crops, grp, lvl


def features(crops, pretrained=True, batch=64):
    import torch, torchvision
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = torchvision.models.resnet18(weights=w); m.fc = torch.nn.Identity(); m.eval().to(dev)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(dev)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(dev)
    out = []
    with torch.no_grad():
        for i in range(0, len(crops), batch):
            x = np.stack([cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops[i:i + batch]]).astype(np.float32) / 255
            x = torch.from_numpy(x).permute(0, 3, 1, 2).to(dev)
            x = torch.nn.functional.interpolate(x, size=224, mode="bilinear", align_corners=False)
            out.append(m((x - mean) / std).cpu().numpy())
    return np.concatenate(out)


def c2st(fa, fb, ga, gb, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X = np.vstack([fa, fb]); y = np.r_[np.zeros(len(fa)), np.ones(len(fb))]
    g = np.array([f"A:{x}" for x in ga] + [f"B:{x}" for x in gb])
    pred = np.zeros(len(y)); prob = np.zeros(len(y))
    n_groups = len(set(g))
    for tr, te in GroupKFold(n_splits=min(5, n_groups)).split(X, y, g):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
        clf.fit(X[tr], y[tr])
        prob[te] = clf.predict_proba(X[te])[:, 1]; pred[te] = prob[te] >= 0.5
    acc = float((pred == y).mean())
    se = np.sqrt(acc * (1 - acc) / len(y))
    return {"accuracy": round(acc, 3), "accuracy_95ci": [round(acc - 1.96 * se, 3), round(acc + 1.96 * se, 3)],
            "auc": round(float(roc_auc_score(y, prob)), 3), "n": int(len(y))}


def kid(fa, fb, subsets=20, size=500, seed=0):
    rng = np.random.default_rng(seed)
    d = fa.shape[1]; m = min(size, len(fa), len(fb)); vals = []
    for _ in range(subsets):
        x = fa[rng.choice(len(fa), m, replace=False)]; y = fb[rng.choice(len(fb), m, replace=False)]
        kxx = (x @ x.T / d + 1) ** 3; kyy = (y @ y.T / d + 1) ** 3; kxy = (x @ y.T / d + 1) ** 3
        mmd = (kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1)) - 2 * kxy.mean()
        vals.append(mmd)
    return {"kid_x1000": round(float(np.mean(vals)) * 1000, 2), "kid_std_x1000": round(float(np.std(vals)) * 1000, 2)}


def prec_rec(real, fake, k=3):
    from sklearn.neighbors import NearestNeighbors
    def radii(z):
        nn = NearestNeighbors(n_neighbors=k + 1).fit(z)
        return nn.kneighbors(z)[0][:, -1]
    def inside(points, manifold, r):
        d = NearestNeighbors(n_neighbors=len(manifold) if len(manifold) < 2000 else 2000).fit(manifold)
        dist, idx = d.kneighbors(points)
        return float(np.mean([(dd <= r[ii]).any() for dd, ii in zip(dist, idx)]))
    rr, rf = radii(real), radii(fake)
    return {"precision": round(inside(fake, real, rr), 3), "recall": round(inside(real, fake, rf), 3)}


def compare(name, fa, fb, ga, gb, lvl_b=None):
    res = {"c2st": c2st(fa, fb, ga, gb), **kid(fa, fb), **prec_rec(fa, fb)}
    print(f"  {name:34s} C2ST acc {res['c2st']['accuracy']:.3f} (AUC {res['c2st']['auc']:.3f}) | "
          f"KID x1000 {res['kid_x1000']:7.2f} | precision {res['precision']:.2f} recall {res['recall']:.2f}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-crops", required=True, help="f1_occlusion_crops folder (train = day-1, test = day-2)")
    ap.add_argument("--synth", nargs="+", required=True, help="one or more synthetic set folders")
    ap.add_argument("--names", nargs="+", default=None)
    ap.add_argument("--n-per-level", type=int, default=400)
    ap.add_argument("--out", default="outputs/synthetic_validation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-pretrained", action="store_true", help="testing only")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    names = a.names or [Path(s).name for s in a.synth]

    print("Loading crops ...")
    ra, ga, _ = real_set(a.real_crops, "train", a.n_per_level, rng)
    rb, gb, _ = real_set(a.real_crops, "test", a.n_per_level, rng)
    sets = [synth_set(s, a.n_per_level, rng) for s in a.synth]
    print(f"  real day-1 {len(ra)}, real day-2 {len(rb)}, " + ", ".join(f"{n} {len(s[0])}" for n, s in zip(names, sets)))

    print("Computing ResNet18 features ...")
    fa, fb = features(ra, not a.no_pretrained), features(rb, not a.no_pretrained)
    fs = [features(s[0], not a.no_pretrained) for s in sets]

    print("\nResults (reference set = real day-1 crops):")
    res = {"baseline_real_day1_vs_real_day2": compare("BASELINE real day-1 vs real day-2", fa, fb, ga, gb)}
    for n, s, f in zip(names, sets, fs):
        res[f"real_day1_vs_{n}"] = compare(f"real day-1 vs synthetic '{n}'", fa, f, ga, s[1])
    print("\nHow to read: C2ST accuracy near the baseline (and far from 1.0), KID near the baseline, and precision /"
          " recall near the baseline mean the synthetic crops look like real ones.")
    json.dump({"n_per_level": a.n_per_level, "results": res}, open(out / "synthetic_validation.json", "w"), indent=2)

    # example sheet: 8 real day-1, 8 real day-2, 8 per synthetic set
    rows = [("real day-1", ra), ("real day-2", rb)] + [(f"synthetic {n}", s[0]) for n, s in zip(names, sets)]
    lines = []
    for title, crops in rows:
        pick = [crops[i] for i in random.Random(1).sample(range(len(crops)), min(8, len(crops)))]
        strip = np.hstack([cv2.resize(c, (96, 96)) for c in pick])
        bar = np.full((22, strip.shape[1], 3), 30, np.uint8)
        cv2.putText(bar, title, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        lines.append(np.vstack([bar, strip]))
    wmax = max(l.shape[1] for l in lines)
    lines = [np.hstack([l, np.full((l.shape[0], wmax - l.shape[1], 3), 30, np.uint8)]) for l in lines]
    cv2.imwrite(str(out / "crop_examples.jpg"), np.vstack(lines))
    print(f"Saved {out / 'synthetic_validation.json'} and {out / 'crop_examples.jpg'}")


if __name__ == "__main__":
    main()
