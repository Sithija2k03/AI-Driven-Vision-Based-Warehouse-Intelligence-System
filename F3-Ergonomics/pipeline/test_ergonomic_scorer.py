
import sys
sys.path.insert(0, r"D:\F3-Project\pipeline")
from ergonomic_scorer import compute_risk

PASS = 0
FAIL = 0

def run_test(name, kp, expect_color, expect_min_composite=None, expect_max_composite=None):
    global PASS, FAIL

    result = compute_risk(kp)
    color     = result["posture_color"]
    composite = result["composite_score"]
    angles    = result["angles"]
    joints    = result["joint_scores"]

    ok_color = (color == expect_color)
    ok_range = True
    if expect_min_composite is not None:
        ok_range = ok_range and (composite >= expect_min_composite)
    if expect_max_composite is not None:
        ok_range = ok_range and (composite <= expect_max_composite)

    passed = ok_color and ok_range
    status = "✅ PASS" if passed else "❌ FAIL"
    if passed: PASS += 1
    else:       FAIL += 1

    print(f"{status}  {name}")
    print(f"       color={color}  composite={composite}/10")
    print(f"       trunk={angles.get('trunk')}°  shoulder={angles.get('shoulder_elevation')}°  "
          f"elbow={angles.get('elbow_flexion')}°  knee={angles.get('knee_flexion')}°")
    trunk_s = round(joints.get('trunk_flexion', 0) * 10, 1)
    shl_s   = round(joints.get('shoulder_elevation', 0) * 10, 1)
    elb_s   = round(joints.get('elbow_flexion', 0) * 10, 1)
    kne_s   = round(joints.get('knee_flexion', 0) * 10, 1)
    print(f"       joint scores → trunk={trunk_s}  shoulder={shl_s}  elbow={elb_s}  knee={kne_s}")
    if not ok_color:
        print(f"       → Expected color={expect_color}, got {color}")
    if not ok_range:
        print(f"       → Expected composite between {expect_min_composite} and {expect_max_composite}, got {composite}")
    print()


# ── Helper: build 17-keypoint array ──────────────────────────────────────────
# All at high confidence (0.9). We only change the key joint positions.
# Default: person standing perfectly upright (all joints stacked vertically)
def make_kp(
    shoulder_y=100, hip_y=250, knee_y=400, ankle_y=550,
    elbow_y=200,    wrist_y=300,
    shoulder_x=200, hip_x=200, knee_x=200, ankle_x=200,
    elbow_x=250,    wrist_x=300,
    conf=0.9
):
    """Build a 17-keypoint list with the key joints placed at given coords."""
    kp = [[200, 50 + i*30, conf] for i in range(17)]   # default positions
    # Overwrite the joints we care about (left side only for simplicity)
    kp[5]  = [shoulder_x, shoulder_y, conf]   # L_shoulder
    kp[6]  = [shoulder_x, shoulder_y, conf]   # R_shoulder (mirror)
    kp[7]  = [elbow_x,    elbow_y,    conf]   # L_elbow
    kp[8]  = [elbow_x,    elbow_y,    conf]   # R_elbow
    kp[9]  = [wrist_x,    wrist_y,    conf]   # L_wrist
    kp[10] = [wrist_x,    wrist_y,    conf]   # R_wrist
    kp[11] = [hip_x,      hip_y,      conf]   # L_hip
    kp[12] = [hip_x,      hip_y,      conf]   # R_hip
    kp[13] = [knee_x,     knee_y,     conf]   # L_knee
    kp[14] = [knee_x,     knee_y,     conf]   # R_knee
    kp[15] = [ankle_x,    ankle_y,    conf]   # L_ankle
    kp[16] = [ankle_x,    ankle_y,    conf]   # R_ankle
    return kp


# ── Test 1: Perfect upright standing posture → GREEN ─────────────────────────
# Shoulder directly above hip, hip above knee, knee above ankle — straight line
kp1 = make_kp(
    shoulder_x=200, shoulder_y=100,
    hip_x=200,      hip_y=250,
    knee_x=200,     knee_y=400,
    ankle_x=200,    ankle_y=550,
    elbow_x=250,    elbow_y=200,   # arms slightly out, comfortable
    wrist_x=280,    wrist_y=310,
)
run_test("Test 1 — Perfect upright standing", kp1,
         expect_color="GREEN", expect_max_composite=3.0)


# ── Test 2: Moderate forward bend (picking from mid shelf) → AMBER ────────────
# Hip shifts forward (x increases), trunk angle drops to ~130°
kp2 = make_kp(
    shoulder_x=200, shoulder_y=150,
    hip_x=230,      hip_y=270,     # hip shifted forward
    knee_x=200,     knee_y=420,
    ankle_x=200,    ankle_y=560,
    elbow_x=170,    elbow_y=230,   # arms reaching forward
    wrist_x=140,    wrist_y=300,
)
run_test("Test 2 — Moderate forward bend (AMBER expected)", kp2,
         expect_color="AMBER")


# ── Test 3: Severe bend + raised arms (floor-level picking) → RED ─────────────
# Trunk heavily bent, shoulder much lower, arms reaching down
kp3 = make_kp(
    shoulder_x=200, shoulder_y=250,  # shoulder dropped low (heavy forward lean)
    hip_x=240,      hip_y=290,       # hip forward
    knee_x=200,     knee_y=420,
    ankle_x=200,    ankle_y=560,
    elbow_x=160,    elbow_y=320,     # arms reaching far forward/down
    wrist_x=120,    wrist_y=390,
)
run_test("Test 3 — Severe forward bend (RED expected)", kp3,
         expect_color="RED", expect_min_composite=5.5)


# ── Test 4: Deep squat posture → RED ─────────────────────────────────────────
# Knee angle collapses (knee moves forward of ankle), trunk also bends
kp4 = make_kp(
    shoulder_x=200, shoulder_y=200,
    hip_x=210,      hip_y=310,
    knee_x=240,     knee_y=400,      # knee pushed forward → deep bend
    ankle_x=200,    ankle_y=450,     # ankle close to knee → deep squat
    elbow_x=230,    elbow_y=300,
    wrist_x=250,    wrist_y=380,
)
run_test("Test 4 — Deep squat posture (RED expected)", kp4,
         expect_color="RED", expect_min_composite=5.5)


# ── Test 5: Arms raised overhead (overhead shelf stacking) → AMBER/RED ────────
# Shoulder elevation angle high — arms above head
kp5 = make_kp(
    shoulder_x=200, shoulder_y=150,
    hip_x=200,      hip_y=290,
    knee_x=200,     knee_y=430,
    ankle_x=200,    ankle_y=560,
    elbow_x=200,    elbow_y=60,      # elbow above shoulder (overhead)
    wrist_x=200,    wrist_y=-30,     # wrist even higher
)
run_test("Test 5 — Arms raised overhead (AMBER or RED expected)", kp5,
         expect_color=None)   # just print, no color assertion


# ── Test 6: Low-confidence keypoints → falls back to mid risk ─────────────────
# All joints have conf=0.1 → angles return None → scores default to 5
kp6 = [[200, 100 + i*30, 0.1] for i in range(17)]
run_test("Test 6 — All low-confidence keypoints (mid risk expected)", kp6,
         expect_color="AMBER", expect_min_composite=3.0, expect_max_composite=6.0)


# ── Summary ────────────────────────────────────────────────────────────────────
print("=" * 60)
print(f"  Results: {PASS} passed, {FAIL} failed out of {PASS+FAIL} tests")
if FAIL == 0:
    print("  ✅ Novelty 2 — Ergonomic Scorer verified correctly.")
else:
    print("  ⚠  Some tests failed — review angles and thresholds above.")
print("=" * 60)