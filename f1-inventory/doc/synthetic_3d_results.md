# F1 synthetic data: 3D rendering attempt and validation (PP1)

## Why
Real annotated CCTV data is limited (two recorded days). The proposal planned synthetic warehouse scenes to add
training data, with a three-stage sim-to-real validation: (1) measure the gap, (2) reduce it / mixed training,
(3) end-to-end check against human ground truth. FID was not used (unreliable for small, domain-specific data).

## How the scenes were made
- Tool: BlenderProc 2.8 (Blender 4.2). Scripts: `render_synthetic_3d.py`, `postprocess_synthetic_3d.py`,
  `prepare_textures.py`, `realism_check_3d.py` (trial folder, outside the main dataset).
- Two racks per scene (3-5 levels), front/back carton rows, stacking, floor cartons; carton faces textured with
  real day-1 carton crops (no day-2 data used).
- Camera matched to the real CCTV by overlay (fSpy calibration was unreliable on this footage): height ~3-3.7 m,
  tilt ~40-60 deg, FOV ~75-85 deg, randomised per scene (domain randomisation). Lighting randomised.
- Rendered at 736 x 832 (real camera aspect 1472 x 1664), then fisheye bend, CCTV look (blur, haze, noise,
  JPEG 70) and resize to 640 x 640 like the real frames.
- Exact labels: full outline = convex hull of the 8 projected box corners; visible = instance mask;
  hidden share = 1 - visible / full; low < 0.30 <= medium < 0.70 <= high. Cartons > 85% hidden or < 150 visible
  pixels are not boxed (a person could not label them either).

## Dataset (500 scenes)
| | Synthetic (500 images) | Real day-2 test/val (37 frames) |
|---|---|---|
| Labelled cartons | 35,275 (70.6 per image) | 8,846 |
| Low / medium / high | 17% / 49% / 34% | 15% / 46% / 40% |

The occlusion mix of the synthetic scenes is close to the real one.

## Realism check (Stage 1: measure the gap)
The real-trained detector (`day2_s1024`, YOLOv8s @1024, conf 0.25, IoU 0.5 one-to-one matching) was run on the
synthetic images. If synthetic cartons looked real, recall per occlusion level would be close to the real recall.
Pass rule fixed before the test: synthetic medium recall >= 0.40.

| Run | Cartons | Low | Medium | High | Result |
|---|---|---|---|---|---|
| Real day-2 frames | 8,846 | 0.66 | 0.57 | 0.41 | reference |
| Pilot 1: plain boxes, 5 scenes | 419 | 0.38 | 0.21 | 0.03 | FAIL |
| Pilot 2: real textures + fisheye + haze, 5 scenes | 413 | 0.29 | 0.09 | 0.05 | FAIL |
| **Full set: 500 scenes** | **35,275** | **0.31** | **0.09** | **0.03** | **FAIL** |

Per-image recall on the 500 scenes: median 0.10 (10th-90th percentile 0.06-0.16).

## Findings
- The gap is large and stable: the 500-scene result repeats the 5-scene pilot (0.29/0.09/0.05), so it is not
  sample noise.
- The occlusion mix matches the real data, but the appearance does not: the real-trained detector finds only
  about half as many low-occlusion cartons and almost none of the medium/high ones.
- Decision (pre-set rule): the 3D synthetic data is NOT used to train the detector. The detector stays trained
  on real data only. Synthetic data is used only where it was validated: inpainting-based stock removal /
  empty-shelf simulation for testing the stock-status logic (STOCK-OUT detection 58% -> 90% after calibration).

## Limits and next steps (PP2)
- Likely causes: carton surfaces, edges and lighting differ from real cardboard under CCTV; rack geometry is simplified;
  box definition of heavily hidden cartons may differ from the human annotation style.
- Stage 2 (mixed training, real day-1 + synthetic, tested on real day-2) was not run for PP1; planned for PP2
  together with copy-paste augmentation (Dwibedi et al. 2017; Ghiasi et al. 2021) as a lower-gap alternative.

### Second, independent check: distribution tests on carton crops (`synthetic_validation.py`)
1,200 crops per set (400 per occlusion level), ImageNet ResNet18 features; baseline = real day-1 vs real day-2.

| Test | Real day-1 vs day-2 (baseline) | Real day-1 vs synthetic |
|---|---|---|
| Real-vs-fake classifier accuracy (C2ST; 0.5 = indistinguishable) | 0.726 (AUC 0.79) | 0.998 (AUC 1.00) |
| KID x1000 (lower = closer) | 41.3 | 798.5 |
| Precision (synthetic inside real manifold) | 0.70 | 0.05 |
| Recall (real covered by synthetic) | 0.49 | 0.01 |

All three tests agree with the detector realism check: the synthetic cartons are clearly distinguishable from real ones.
Diagnosis (100 scenes): lowering confidence to 0.05 or the match to IoU 0.3 raised medium recall only to 0.15,
and at conf 0.25 the detector made ~22 boxes per image for ~71 cartons, so the main gap is appearance; small (far) cartons are missed more (median height 48 vs 71 px).