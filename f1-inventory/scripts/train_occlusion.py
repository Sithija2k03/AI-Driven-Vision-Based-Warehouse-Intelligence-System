r"""
F1 sub-function 2, step 2b: train the occlusion classifier (low / medium / high) on carton crops.
Run on Colab with a GPU (Runtime -> Change runtime type -> T4 GPU).

Model (--arch): resnet18 (default), efficientnet_b0 or mobilenet_v3_small, pre-trained on ImageNet,
last layer replaced by 3 outputs.
Loss: cross-entropy with class weights (classes are imbalanced).
Best epoch chosen on VAL macro-F1; final numbers reported once on TEST (day-2 crops).
Also reports a majority-class baseline, so the result can be judged fairly.

Colab usage:
  !python train_occlusion.py --data /content/f1_occlusion_crops --out /content/drive/MyDrive/f1_runs/occlusion_r18
  !python train_occlusion.py --data /content/f1_occlusion_crops --arch efficientnet_b0 --out /content/drive/MyDrive/f1_runs/occlusion_effb0
"""
import argparse, json, time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, cohen_kappa_score

CLASSES = ["low", "medium", "high"]  # ordinal order


def loaders(root, size, bs, workers):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    train_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.2, 0.02),
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ToTensor(), transforms.Normalize(mean, std)])
    eval_tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(),
                                  transforms.Normalize(mean, std)])
    ds = {}
    for split, tf in (("train", train_tf), ("val", eval_tf), ("test", eval_tf)):
        d = datasets.ImageFolder(Path(root) / split, transform=tf)
        # force class index order low=0, medium=1, high=2 (ImageFolder sorts alphabetically)
        remap = {d.class_to_idx[c]: i for i, c in enumerate(CLASSES)}
        d.samples = [(p, remap[t]) for p, t in d.samples]
        d.targets = [t for _, t in d.samples]
        d.class_to_idx = {c: i for i, c in enumerate(CLASSES)}
        d.classes = CLASSES
        ds[split] = d
    dl = {s: DataLoader(ds[s], batch_size=bs, shuffle=(s == "train"), num_workers=workers, pin_memory=True)
          for s in ds}
    return ds, dl


def build_model(arch):
    if arch == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, 3)
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 3)
    elif arch == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 3)
    else:
        raise SystemExit(f"unknown --arch {arch}")
    return m


def predict(model, dl, device):
    model.eval()
    ys, ps, probs = [], [], []
    with torch.no_grad():
        for x, y in dl:
            out = torch.softmax(model(x.to(device)), 1).cpu()
            probs.append(out); ps.append(out.argmax(1)); ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy(), torch.cat(probs).numpy()


def summary(y, p, prob):
    exp_level = prob @ np.array([0.0, 1.0, 2.0])  # expected ordinal level (0..2)
    return {
        "accuracy": round(float(accuracy_score(y, p)), 4),
        "cohen_kappa": round(float(cohen_kappa_score(y, p)), 4),
        "quadratic_weighted_kappa": round(float(cohen_kappa_score(y, p, weights="quadratic")), 4),
        "macro_f1": round(float(f1_score(y, p, average="macro")), 4),
        "within_one_accuracy": round(float(np.mean(np.abs(y - p) <= 1)), 4),
        "low_vs_high_confusions": int(np.sum(np.abs(y - p) == 2)),
        "mae_expected_level": round(float(np.mean(np.abs(exp_level - y))), 4),
        "confusion_matrix_rows_true_cols_pred": confusion_matrix(y, p, labels=[0, 1, 2]).tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with train/ val/ test/ from make_occlusion_crops.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", default="resnet18", choices=["resnet18", "efficientnet_b0", "mobilenet_v3_small"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label-smoothing", type=float, default=0.0,
                    help="e.g. 0.1: softer targets, helps with noisy (subjective) labels")
    ap.add_argument("--freeze-epochs", type=int, default=0,
                    help="train only the last layer for the first N epochs, then the whole network")
    ap.add_argument("--patience", type=int, default=0,
                    help="stop if val macro-F1 has not improved for N epochs (0 = never stop early)")
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ds, dl = loaders(a.data, a.size, a.bs, a.workers)
    for s in ds:
        c = Counter(ds[s].targets)
        print(f"{s:5s}: " + ", ".join(f"{CLASSES[i]} {c.get(i, 0)}" for i in range(3)) + f"  (total {len(ds[s])})")

    counts = Counter(ds["train"].targets)
    weights = torch.tensor([len(ds["train"]) / (3 * counts[i]) for i in range(3)], dtype=torch.float)
    print("class weights:", [round(float(w), 3) for w in weights])

    model = build_model(a.arch).to(device)
    print(f"model: {a.arch}  ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters)  device: {device}")
    crit = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=a.label_smoothing)
    head = model.fc if a.arch == "resnet18" else model.classifier
    head_ids = {id(p) for p in head.parameters()}

    def set_backbone(trainable):
        for p in model.parameters():
            if id(p) not in head_ids:
                p.requires_grad = trainable
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    best_f1, history, since_best = -1.0, [], 0
    for ep in range(1, a.epochs + 1):
        set_backbone(ep > a.freeze_epochs)
        model.train(); t0 = time.time(); tot, n = 0.0, 0
        for x, y in dl["train"]:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward(); opt.step()
            tot += loss.item() * len(y); n += len(y)
        sched.step()
        yv, pv, prv = predict(model, dl["val"], device)
        f1v = f1_score(yv, pv, average="macro"); accv = accuracy_score(yv, pv)
        history.append({"epoch": ep, "train_loss": round(tot / n, 4), "val_acc": round(accv, 4), "val_macro_f1": round(f1v, 4)})
        flag = ""
        if f1v > best_f1:
            best_f1 = f1v; since_best = 0; torch.save(model.state_dict(), out / "best.pt"); flag = "  <- best"
        else:
            since_best += 1
        frozen = "  [backbone frozen]" if ep <= a.freeze_epochs else ""
        print(f"epoch {ep:2d}  loss {tot / n:.4f}  val acc {accv:.3f}  val macro-F1 {f1v:.3f}  ({time.time() - t0:.0f}s){flag}{frozen}")
        if a.patience and since_best >= a.patience:
            print(f"early stop: no val improvement for {a.patience} epochs"); break

    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    yt, pt, prt = predict(model, dl["test"], device)
    yv, pv, prv = predict(model, dl["val"], device)

    majority = Counter(ds["train"].targets).most_common(1)[0][0]
    base_pred = np.full_like(yt, majority)
    results = {
        "classes": CLASSES,
        "best_val_macro_f1": round(float(best_f1), 4),
        "val": summary(yv, pv, prv),
        "test": summary(yt, pt, prt),
        "test_majority_baseline": {"predicts": CLASSES[majority],
                                   "accuracy": round(float(accuracy_score(yt, base_pred)), 4),
                                   "macro_f1": round(float(f1_score(yt, base_pred, average="macro")), 4)},
        "history": history,
        "settings": vars(a),
    }
    # per-batch test accuracy (batch name is the file-name prefix)
    per_batch = {}
    for (path, _), yy, pp in zip(ds["test"].samples, yt, pt):
        b = Path(path).name.split("__")[0]
        s = per_batch.setdefault(b, [0, 0]); s[0] += int(yy == pp); s[1] += 1
    results["test_accuracy_per_batch"] = {b: round(k / n, 4) for b, (k, n) in per_batch.items()}

    json.dump(results, open(out / "metrics.json", "w"), indent=2)
    print("\n=== TEST (day-2 crops) ===")
    print(classification_report(yt, pt, target_names=CLASSES, digits=3))
    t = results["test"]
    print(f"accuracy {t['accuracy']}  macro-F1 {t['macro_f1']}  kappa {t['cohen_kappa']}  QWK {t['quadratic_weighted_kappa']}  "
          f"within-one {t['within_one_accuracy']}  "
          f"low<->high confusions {t['low_vs_high_confusions']}")
    print("confusion matrix (rows = true low/medium/high, cols = predicted):")
    for row in t["confusion_matrix_rows_true_cols_pred"]:
        print("   ", row)
    mb = results["test_majority_baseline"]
    print(f"majority baseline (always '{mb['predicts']}'): accuracy {mb['accuracy']}  macro-F1 {mb['macro_f1']}")
    print("per-batch accuracy:", results["test_accuracy_per_batch"])
    # speed: average milliseconds per crop on this device
    model.eval(); x = next(iter(dl["test"]))[0][:64].to(device)
    with torch.no_grad():
        for _ in range(3): model(x)
        if device == "cuda": torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(10): model(x)
        if device == "cuda": torch.cuda.synchronize()
    ms = (time.time() - t0) / (10 * len(x)) * 1000
    results["ms_per_crop_" + device] = round(ms, 3)
    json.dump(results, open(out / "metrics.json", "w"), indent=2)
    print(f"speed: {ms:.2f} ms per crop on {device}")
    print(f"\nSaved {out / 'best.pt'} and {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
