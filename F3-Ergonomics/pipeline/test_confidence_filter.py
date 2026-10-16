
import sys
sys.path.insert(0, r"D:\F3-Project\pipeline")
from confidence_filter import check_confidence, reset_worker

PASS = 0
FAIL = 0

def run_test(name, kp, worker_id, expect_freeze, expect_zones):
    global PASS, FAIL
    reset_worker(worker_id)
    result = check_confidence(kp, worker_id)

    freeze  = result["freeze_state"]          # dict: {upper, core, lower}
    zones   = result["frozen_zones"]          # list of frozen zone names
    reason  = result["reason"]

    # freeze_state can be a dict {upper:bool, core:bool, lower:lower} OR a single bool
    if isinstance(freeze, dict):
        any_frozen = any(freeze.values())
    else:
        any_frozen = bool(freeze)

    ok_freeze = (any_frozen == expect_freeze)
    ok_zones  = all(z in zones for z in expect_zones)
    passed    = ok_freeze and ok_zones

    status = "✅ PASS" if passed else "❌ FAIL"
    if passed:
        PASS += 1
    else:
        FAIL += 1

    print(f"{status}  {name}")
    print(f"       freeze_state={freeze}  frozen_zones={zones}")
    print(f"       reason: {reason}")
    if not ok_freeze:
        print(f"       → Expected any_frozen={expect_freeze}, got {any_frozen}")
    if not ok_zones:
        print(f"       → Expected zones to contain {expect_zones}, got {zones}")
    print()

# ── Test 1: All keypoints fully visible → no freeze ───────────────────────────
kp1 = [[100 + i*10, 200 + i*5, 0.9] for i in range(17)]
run_test(
    "Test 1 — All visible (conf=0.9)",
    kp1, "worker_t1",
    expect_freeze=False,
    expect_zones=[]
)

# ── Test 2: Lower body occluded (hips/knees/ankles low conf) → freeze lower ───
kp2 = [[100 + i*10, 200 + i*5, 0.9] for i in range(17)]
for i in [11, 12, 13, 14, 15, 16]:   # l_hip, r_hip, l_knee, r_knee, l_ankle, r_ankle
    kp2[i][2] = 0.05
run_test(
    "Test 2 — Lower body occluded (hips/knees/ankles conf=0.05)",
    kp2, "worker_t2",
    expect_freeze=True,
    expect_zones=["lower"]
)

# ── Test 3: Upper body occluded (shoulders/elbows/wrists low conf) → freeze upper
kp3 = [[100 + i*10, 200 + i*5, 0.9] for i in range(17)]
for i in [5, 6, 7, 8, 9, 10]:        # l_shl, r_shl, l_elb, r_elb, l_wrist, r_wrist
    kp3[i][2] = 0.05
run_test(
    "Test 3 — Upper body occluded (shoulders/elbows/wrists conf=0.05)",
    kp3, "worker_t3",
    expect_freeze=True,
    expect_zones=["upper"]
)

# ── Test 4: Core occluded (hips only low conf) → freeze core ─────────────────
kp4 = [[100 + i*10, 200 + i*5, 0.9] for i in range(17)]
for i in [11, 12]:                    # l_hip, r_hip
    kp4[i][2] = 0.05
run_test(
    "Test 4 — Core occluded (hips conf=0.05)",
    kp4, "worker_t4",
    expect_freeze=True,
    expect_zones=["core"]
)

# ── Test 5: Borderline confidence (exactly at 0.3 threshold) ─────────────────
kp5 = [[100 + i*10, 200 + i*5, 0.30] for i in range(17)]
run_test(
    "Test 5 — All keypoints at borderline conf=0.30",
    kp5, "worker_t5",
    expect_freeze=False,   # 0.30 is the acceptance threshold — should pass
    expect_zones=[]
)

# ── Test 6: Completely invisible worker (all conf=0.0) → full freeze ──────────
kp6 = [[0, 0, 0.0] for _ in range(17)]
run_test(
    "Test 6 — All keypoints missing (conf=0.0)",
    kp6, "worker_t6",
    expect_freeze=True,
    expect_zones=[]        # zones list may vary — just checking any freeze fires
)

# ── Summary ────────────────────────────────────────────────────────────────────
print("=" * 55)
print(f"  Results: {PASS} passed, {FAIL} failed out of {PASS+FAIL} tests")
if FAIL == 0:
    print("  ✅ Novelty 1 — Confidence Filter verified correctly.")
else:
    print("  ⚠  Some tests failed — check frozen_zones logic above.")
print("=" * 55)