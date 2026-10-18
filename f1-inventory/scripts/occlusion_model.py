r"""
Shared occlusion code for F1 sub-function 2 (used by count_stock.py and evaluate_trust.py).

OcclusionEstimator gives, for every detected box in a frame, probabilities for low / medium / high occlusion:
  mode "cnn"    : ResNet18 on the box crop (appearance)
  mode "hybrid" : average of the CNN and the geometry random forest (layout of all boxes in the frame)

Trust (how far a count can be believed) is built from the measured detector recall per occlusion level
(step 1: recall low 0.659, medium 0.572, high 0.414 at IoU 0.5). Relative to low:
  low 1.00, medium 0.87, high 0.63
Each box gets an expected weight  sum_k p(k) * w(k); a location's trust is the mean over its boxes.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_occlusion_crops import crop_square                     # noqa: E402
from geometry_occlusion_rf import frame_features, load as load_geometry, DAY1, LEVELS  # noqa: E402

RECALL = {"low": 0.659, "medium": 0.572, "high": 0.414}
TRUST_WEIGHTS = np.array([RECALL[l] / RECALL["low"] for l in LEVELS])   # [1.00, 0.87, 0.63]


def train_or_load_rf(path, converted="annotations/converted", img_w=640, img_h=640, min_side=6, seed=0):
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    path = Path(path)
    if path.exists():
        return joblib.load(path)
    print(f"Training geometry random forest on day-1 annotations (saved to {path}) ...")
    X, y, _ = load_geometry(converted, DAY1, img_w, img_h, min_side)
    rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=5, class_weight="balanced",
                                n_jobs=-1, random_state=seed).fit(X, y)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf, path)
    return rf


class OcclusionEstimator:
    def __init__(self, cnn_weights, mode="hybrid", rf_path="weights/occlusion_geometry_rf.joblib",
                 converted="annotations/converted", pad=0.25, size=128, cnn_share=0.5, bs=64):
        import torch, torch.nn as nn
        from torchvision import models, transforms
        self.torch = torch
        self.mode, self.pad, self.size, self.w, self.bs = mode, pad, size, cnn_share, bs
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, 3)
        m.load_state_dict(torch.load(cnn_weights, map_location="cpu"))
        self.cnn = m.eval()
        self.tf = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.rf = train_or_load_rf(rf_path, converted) if mode == "hybrid" else None

    def cnn_probs(self, img_bgr, boxes):
        import cv2
        torch = self.torch
        out = []
        with torch.no_grad():
            for i in range(0, len(boxes), self.bs):
                crops = [crop_square(img_bgr, *b[:4], self.pad, self.size) for b in boxes[i:i + self.bs]]
                x = torch.stack([self.tf(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops])
                out.append(torch.softmax(self.cnn(x), 1).numpy())
        return np.concatenate(out) if out else np.zeros((0, 3))

    def predict(self, img_bgr, boxes):
        """boxes: list/array of (x1, y1, x2, y2[, ...]) in pixels. Returns (n, 3) probabilities."""
        boxes = np.asarray([b[:4] for b in boxes], dtype=float)
        if len(boxes) == 0:
            return np.zeros((0, 3))
        p = self.cnn_probs(img_bgr, boxes)
        if self.mode == "hybrid":
            h, w = img_bgr.shape[:2]
            pg = self.rf.predict_proba(frame_features(boxes, w, h))
            p = self.w * p + (1 - self.w) * pg
        return p


def box_trust(probs):
    """Expected trust weight per box (0.63 .. 1.0)."""
    return probs @ TRUST_WEIGHTS if len(probs) else np.zeros(0)


def location_summary(probs, trust_threshold):
    """Occlusion mix, trust and verify flag for the boxes of one location."""
    if len(probs) == 0:
        return {"occlusion_mix": {l: 0 for l in LEVELS}, "trust": None, "verify_manually": None}
    lvl = probs.argmax(1)
    t = float(box_trust(probs).mean())
    return {"occlusion_mix": {l: int((lvl == i).sum()) for i, l in enumerate(LEVELS)},
            "trust": round(t, 3), "verify_manually": bool(t < trust_threshold)}
