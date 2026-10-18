# predict_demographic.py
# F3 Pipeline — Demographic Inference Module (Novelty 3)
# Warehouse Intelligence System | J26-DS-352 | SLIIT
#
# Loads the trained DemographicMLP and predicts age band + gender
# for a single crop image.
#
# Usage (standalone test):
#   python predict_demographic.py --image D:\path\to\crop.jpg
#
# Usage (from pipeline / dashboard):
#   from predict_demographic import predict_demographic, load_demographic_model
#   model, meta = load_demographic_model()
#   result = predict_demographic(model, meta, img_path)
#   # result = {"age": "36-50+", "gender": "Male",
#   #           "age_conf": 0.87, "gender_conf": 0.91, "status": "ok"}

import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

MODEL_PATH = r"D:\F3-Project\models\demographic_mlp.pt"
META_PATH  = r"D:\F3-Project\models\demographic_meta.json"

# ── Model definition (must match train_demographic.py exactly) ────────────────
class DemographicMLP(nn.Module):
    def __init__(self, n_age_classes=2, n_gender_classes=2):
        super().__init__()
        backbone = models.resnet18(weights=None)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.age_head = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_age_classes)
        )
        self.gender_head = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_gender_classes)
        )

    def forward(self, x):
        feat = self.backbone(x)
        return self.age_head(feat), self.gender_head(feat)

# ── Transform (must match val_tf in training) ─────────────────────────────────
INFER_TF = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── Loader (call once, cache the result) ─────────────────────────────────────
_cached_model = None
_cached_meta  = None

def load_demographic_model(model_path: str = MODEL_PATH, meta_path: str = META_PATH):
    """
    Load the trained model and class meta.
    Returns (model, meta) or (None, None) if model file doesn't exist yet.
    """
    global _cached_model, _cached_meta

    if _cached_model is not None:
        return _cached_model, _cached_meta

    if not Path(model_path).exists():
        return None, None

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = DemographicMLP(
        n_age_classes    = checkpoint["n_age_classes"],
        n_gender_classes = checkpoint["n_gender_classes"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    meta = {
        "age_classes":      checkpoint["age_classes"],
        "gender_classes":   checkpoint["gender_classes"],
        "age_to_idx":       checkpoint["age_to_idx"],
        "gender_to_idx":    checkpoint["gender_to_idx"],
    }

    # Also pull note fields from meta JSON if it exists
    if Path(meta_path).exists():
        with open(meta_path) as f:
            json_meta = json.load(f)
        meta.update(json_meta)

    _cached_model = model
    _cached_meta  = meta
    return model, meta


def predict_demographic(model, meta: dict, img_path: str) -> dict:
    """
    Run age + gender inference on a single image.

    Returns:
        {
          "age":         "18-35" | "36-50+",
          "gender":      "Male"  | "Female",
          "age_conf":    float (0–1),
          "gender_conf": float (0–1),
          "status":      "ok" | "error: <reason>"
        }
    """
    if model is None:
        return {
            "age": "Pending", "gender": "Pending",
            "age_conf": 0.0, "gender_conf": 0.0,
            "status": "model not trained yet"
        }

    try:
        img   = Image.open(img_path).convert("RGB")
        x     = INFER_TF(img).unsqueeze(0)   # [1, 3, 128, 128]

        with torch.no_grad():
            age_logits, gen_logits = model(x)

        age_probs = torch.softmax(age_logits, dim=1)[0]
        gen_probs = torch.softmax(gen_logits, dim=1)[0]

        age_idx  = int(age_probs.argmax())
        gen_idx  = int(gen_probs.argmax())

        age_label  = meta["age_classes"][age_idx]
        gen_label  = meta["gender_classes"][gen_idx]
        age_conf   = float(age_probs[age_idx])
        gen_conf   = float(gen_probs[gen_idx])

        return {
            "age":         age_label,
            "gender":      gen_label,
            "age_conf":    round(age_conf, 3),
            "gender_conf": round(gen_conf, 3),
            "status":      "ok",
        }

    except Exception as e:
        return {
            "age": "Error", "gender": "Error",
            "age_conf": 0.0, "gender_conf": 0.0,
            "status": f"error: {e}"
        }


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",  required=True, help="Path to crop image")
    parser.add_argument("--model",  default=MODEL_PATH)
    parser.add_argument("--meta",   default=META_PATH)
    args = parser.parse_args()

    model, meta = load_demographic_model(args.model, args.meta)
    if model is None:
        print("❌ Model not found. Run train_demographic.py first.")
    else:
        result = predict_demographic(model, meta, args.image)
        print(f"\nImage : {args.image}")
        print(f"Age   : {result['age']}  (conf {result['age_conf']:.2f})")
        print(f"Gender: {result['gender']}  (conf {result['gender_conf']:.2f})")
        print(f"Status: {result['status']}")