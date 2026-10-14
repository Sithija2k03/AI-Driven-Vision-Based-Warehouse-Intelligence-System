# F1 sub-function 2: occlusion-aware compliance (PP1 results)

## Step 1. Detector recall by occlusion level

Question: does the detector miss more cartons when they are more hidden?
Method: `scripts/recall_by_occlusion.py`. Detector `day2_s1024` (conf 0.25, NMS IoU 0.5, imgsz 1024).
Each annotated carton counts as found if one predicted box overlaps it with IoU >= 0.5 (one-to-one, greedy).
Data: the three day-2 batches, never used for training (37 frames, 8,846 cartons).

| Batch | Low | Medium | High |
|---|---|---|---|
| G_H day 2 (test) | 0.705 (340/482) | 0.545 (515/945) | 0.338 (350/1034) |
| X_5 day 2 (test) | 0.667 (340/510) | 0.542 (906/1673) | 0.412 (475/1152) |
| checkingPart_2 day 2 (val) | 0.572 (170/297) | 0.625 (899/1438) | 0.475 (624/1315) |
| **All** | **0.659** (850/1289) | **0.572** (2320/4056) | **0.414** (1449/3501) |

Findings
- Overall recall falls from 0.66 (low) to 0.57 (medium) to 0.41 (high): about 59% of highly occluded cartons are missed,
  against about 34% of low-occlusion cartons.
- High occlusion has the lowest recall in every batch. Low > medium > high holds in G_H and X_5;
  in checkingPart_2 low is slightly below medium (only 297 low boxes; the intervals overlap).
- Conclusion: carton counts are least reliable where occlusion is high, which supports estimating occlusion per box
  and using it to judge how far a count can be trusted.

Limits
- 95% intervals (Wilson) treat cartons as independent; frames in a batch are near-duplicates, so true uncertainty is larger.
- Occlusion levels come from one annotator (agreement check pending).
- checkingPart_2 day 2 was the validation split.

### Robustness check: looser match (IoU >= 0.3)

| Batch | Low | Medium | High |
|---|---|---|---|
| G_H day 2 | 0.772 | 0.619 | 0.473 |
| X_5 day 2 | 0.745 | 0.735 | 0.569 |
| checkingPart_2 day 2 | 0.818 | 0.778 | 0.703 |
| **All** | **0.772** (995/1289) | **0.723** (2933/4056) | **0.591** (2069/3501) |

- With a looser match, low > medium > high now holds in all three batches.
- The low-to-high gap shrinks from 0.245 (IoU 0.5) to 0.181 (IoU 0.3): part of the drop is loose boxing of
  partly hidden cartons, but most of it remains, so highly occluded cartons are genuinely missed more often.
- About 41% of highly occluded cartons are not found even with the looser match (against about 23% of low ones).
- Caveat: at IoU 0.3 in dense scenes a prediction can occasionally match a neighbouring carton, so these values are
  an upper bound on recall.

## Step 2. Occlusion classifier (low / medium / high per carton crop)

Method: `scripts/make_occlusion_crops.py` cuts every annotated carton (+25% surrounding context, padded to a square,
128x128 px; boxes under 6 px skipped). Split by whole batch: train = 5 day-1 batches (12,603 crops),
val = camera_18 day 1 (2,120), test = 3 day-2 batches (7,950). `scripts/train_occlusion.py` fine-tunes an
ImageNet-pretrained CNN with class-weighted cross-entropy; best epoch chosen on val macro-F1; test reported once.

| Run | Model | Training data | Settings | Test acc | Macro-F1 | Kappa | QWK | Within-one | Low<->high errors |
|---|---|---|---|---|---|---|---|---|---|
| Baseline | always "medium" | - | - | 0.456 | 0.209 | 0 | - | - | - |
| v1 | ResNet18 | all crops (12,603) | none | 0.542 | 0.529 | 0.29 | 0.39 | 0.929 | 564 |
| A | ResNet18 | de-duplicated (1,074) | LS 0.1, frozen 2 ep, early stop | 0.529 | 0.514 | 0.25 | 0.38 | 0.946 | 429 |
| **B (selected)** | **ResNet18** | all crops (12,603) | LS 0.1, frozen 2 ep, early stop | **0.575** | **0.561** | **0.34** | **0.46** | **0.946** | **431** |
| C | EfficientNet-B0 | all crops (12,603) | same as B | 0.555 | 0.538 | 0.31 | 0.43 | 0.935 | 521 |

LS = label smoothing; QWK = quadratic-weighted kappa (penalises low<->high more than one-level errors).

Selected model: ResNet18, run B. Per-class F1: low 0.49, medium 0.56, high 0.64.
Per-camera accuracy: G_H 0.63, X_5 0.61, checkingPart_2 0.50. Speed 0.37 ms per crop (T4 GPU).

Findings
- All CNN runs clearly beat the majority baseline (macro-F1 0.51-0.56 vs 0.21); about 95% of predictions are within one level.
- v1 over-fitted: training loss fell to ~0.005 while validation peaked at epoch 3. Cause: propagated annotations repeat
  the same carton in 10-15 frames (de-duplication showed only 1,074 unique cartons in training).
- Regularisation (label smoothing, frozen-backbone warm-up, early stopping) fixed this and improved every metric (run B).
- Keeping the repeated crops (B) beat de-duplicating (A): the repeats differ slightly (lighting, people), and A had too little data.
- EfficientNet-B0 under identical conditions was slightly worse and slower than ResNet18, so ResNet18 was kept.
- Medium is the hardest class (often confused with both neighbours), as expected for an in-between, subjective level.

Limits
- One training run (seed) per setting; differences of 1-2 points may be chance.
- Labels come from one annotator; the classifier cannot be more consistent than the labels (agreement check pending).
- Classifier is trained on annotated boxes; on live frames it will see detector boxes, which are less precise.

### Step 2c. Geometry-based alternative (random forest on box layout)

`scripts/geometry_occlusion_rf.py`: each box is described only by its geometry relative to the other boxes in the same
frame (size, aspect, position, size relative to neighbours and to the frame median, overlaps, cover by boxes in front,
crowding). Random forest (400 trees, balanced class weights). Identical split, filtering and test set (7,950 boxes).

| Model | Input | Test acc | Macro-F1 | Kappa | QWK | Within-one | Low<->high errors | G_H | X_5 | checkingPart_2 |
|---|---|---|---|---|---|---|---|---|---|---|
| ResNet18 (run B) | crop pixels | 0.575 | 0.561 | 0.34 | 0.46 | 0.946 | 431 | 0.63 | 0.61 | 0.50 |
| Random forest | box geometry | 0.575 | 0.559 | 0.32 | 0.46 | **0.959** | **325** | 0.57 | 0.59 | **0.56** |

Findings
- Same overall accuracy (0.575) from completely different information: appearance vs layout.
- The geometry model makes fewer extreme errors (low<->high 325 vs 431) and is much better on checkingPart_2 (0.56 vs 0.50);
  the CNN is better on G_H. The two models make different mistakes, so combining them is a promising next step.
- Most useful features: box size relative to its neighbours and to the frame median, horizontal position, aspect ratio,
  crowding. Occluded cartons tend to appear smaller and more oddly shaped than their neighbours (the annotations mark the
  visible part). Horizontal position being important suggests some camera-layout learning, which may not transfer to new cameras.

Important caveat for deployment
- Here the geometry features were computed from the annotated boxes. On live frames they must come from detector boxes,
  and the detector misses about 40-60% of medium/high cartons, so "neighbours" and relative sizes will look different.
  The CNN only needs each box's own crop, so it is less affected. Geometry features must be re-tested on detector output.

### Step 2d. Combining the CNN and the geometry model (probability averaging)

`scripts/fuse_occlusion_models.py`: p = w * p_CNN + (1 - w) * p_geometry. Two variants were defined before testing:
an untuned equal-weight average (w = 0.5) and w chosen on the validation set.

| Model | Test acc | Macro-F1 | Kappa | QWK | Within-one | Low<->high errors |
|---|---|---|---|---|---|---|
| CNN only (ResNet18) | 0.575 | 0.561 | 0.336 | 0.460 | 0.946 | 431 |
| Geometry only (random forest) | 0.575 | 0.559 | 0.319 | 0.462 | 0.959 | 325 |
| **Equal-weight fusion (w = 0.5)** | **0.605** | **0.589** | **0.376** | **0.506** | 0.957 | 339 |
| Fusion, w chosen on val (w = 1.0 = CNN only) | 0.575 | 0.561 | 0.336 | 0.460 | 0.946 | 431 |

Findings
- Equal-weight fusion beats both single models on every agreement measure (+3 points accuracy and macro-F1,
  kappa 0.34 -> 0.38), and keeps the geometry model's low rate of low<->high errors.
- Choosing w on the validation set did not work: val macro-F1 was flat and noisy across all weights (0.49-0.52),
  and the same val set had already been used to pick the CNN's best epoch, which biases it towards w = 1.0.
- Decision: use the untuned equal-weight fusion. Because the 0.5 weight is a standard default and was not tuned on test,
  the test result remains a fair estimate; a better weight should later be chosen with leave-one-batch-out validation.
- Caveat (same as 2c): geometry features here come from annotated boxes; on live frames they come from detector boxes.
  The end-to-end check on detector boxes is done in step 3.

## Step 3 – Trust score and "verify manually" flag

**Method.** Each detected box gets occlusion probabilities from the hybrid model (0.5·CNN + 0.5·geometry RF).
Box trust = Σ p(level) · w(level), where the weights come from measured recall by occlusion level (IoU 0.5):
low 1.00, medium 0.868 (0.572/0.659), high 0.628 (0.414/0.659).
Location trust = mean box trust. If location trust < threshold, the location is flagged "verify manually".

**Evaluation data:** day-2 test frames (G_H_b2, X_5_b2) and val (checkingPart_2_b2): 37 frames, 3 cameras.
Each frame was split into a 3×3 grid; cells with ≥5 ground-truth boxes were kept (304 cells).

### A) Occlusion accuracy on detector boxes (n = 4,619 matched boxes)
| Model  | Acc   | Macro-F1 | Kappa | Low↔High errors |
|--------|-------|----------|-------|-----------------|
| CNN    | 0.558 | 0.560    | 0.311 | 133 |
| Hybrid | 0.564 | 0.565    | 0.318 | 120 |

### B) Does trust predict detector recall? (304 cells)
| Trust source        | Spearman ρ with cell recall |
|---------------------|------------------------------|
| Hybrid (used)       | +0.351 (p = 3.2e-10) |
| CNN only            | +0.296 |
| True labels         | +0.311 |

Mean recall by trust quartile: 0.462 → 0.483 → 0.691 → 0.613.
Flagged at 0.80: 108/304 cells, mean recall 0.443 vs 0.628 for unflagged cells.

### C) Threshold selection
"Unreliable" cell = recall < 0.50 (124/304 cells). ROC AUC = 0.669. Youden's J optimum = 0.806.

| Held-out camera | Threshold (from other cams) | Unreliable caught | False alarms | Flagged |
|-----------------|-----------------------------|-------------------|--------------|---------|
| G_H             | 0.804 | 60%  | 49% | 55% |
| X_5             | 0.800 | 48%  | 19% | 33% |
| CP2             | 0.806 | 100% | 1%  | 26% |

At the chosen threshold 0.80 (all cells): 62% of unreliable cells caught, 17% false alarms, 36% of cells flagged.

**Decision:** keep the hybrid and a threshold of 0.80. It matches Youden's optimum and is stable across cameras.

**Limitations:** moderate AUC (0.67); the top trust quartile is not monotonic; high false alarms on the darker G_H camera;
only 37 frames from 3 cameras. Retest after the new cameras (cam16–21) are annotated.
