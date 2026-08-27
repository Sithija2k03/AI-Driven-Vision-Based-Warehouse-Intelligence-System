"""
f4-routing/mock-publishers/mock_f3_publisher.py

Simulates F3 ergonomics and fall risk output on the message bus.
Used for F4 development while the real F3 function is being built.

Publishes to channel: f3.ergonomics every 10 seconds.

IMPORTANT FOR F4 INTEGRATION:
- ergonomic_risk_score (1-10) is used as edge weight modifier ONLY
  It never overrides the primary efficiency objective
- gait_stability_score (1-10) triggers fall-risk aisle avoidance
  when score exceeds threshold of 7
- task_tier gates node eligibility (Heavy/Moderate/Light)
  incompatible nodes are masked from action space
- freeze_state = True means DO NOT update edge weights this cycle
  F4 must hold previous valid scores unchanged

Scores are on a 1-10 integer scale.
No fatigue time progression — risk is posture × demographic weight only.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import time
import random
from datetime import datetime
from shared.message_bus import publish, CHANNEL_F3_ERGONOMICS

# ── Simulated pickers ──────────────────────────────────────────────────────
PICKERS = [
    {"picker_id": "P-001", "demographic_band": "18-35", "vulnerability_weight": 1.0},
    {"picker_id": "P-002", "demographic_band": "36-50", "vulnerability_weight": 1.3},
    {"picker_id": "P-003", "demographic_band": "51+",   "vulnerability_weight": 1.6},
]

# ── Configuration ──────────────────────────────────────────────────────────
PUBLISH_INTERVAL_SECONDS = 10
FREEZE_STATE_PROBABILITY = 0.05   # 5% of cycles skeleton is occluded

# ── Per-picker state ───────────────────────────────────────────────────────
picker_state = {
    p["picker_id"]: {
        "base_risk":          random.uniform(0.1, 0.3),
        "demographic_weight": p["vulnerability_weight"],
        "demographic_band":   p["demographic_band"],
        "last_ergo_score":    3,    # held during freeze states
        "last_gait_score":    2,
    }
    for p in PICKERS
}


# ── Helpers ────────────────────────────────────────────────────────────────

def calculate_task_tier(score: int) -> str:
    """
    1-3  → Heavy   (picker is low-risk, can handle all task weights)
    4-7  → Moderate (elevated risk, avoid heaviest tasks)
    8-10 → Light   (high risk, light duties only)

    NOTE: Heavy tier means the PICKER is low-risk and can do Heavy tasks.
    Light tier means the PICKER is high-risk and should only do Light tasks.
    This is counter-intuitive but correct — tier describes picker capability,
    not task weight.
    """
    if score <= 3:
        return "Heavy"
    elif score <= 7:
        return "Moderate"
    else:
        return "Light"


def build_reason(ergo: int, gait: int, tier: str,
                 band: str, frozen: bool) -> str:
    if frozen:
        return ("Skeleton confidence below threshold — freeze state. "
                "Last valid scores held. No route change triggered.")
    parts = []
    if ergo >= 8:
        parts.append(f"Ergonomic risk high ({ergo}/10): "
                     "spine flexion above safe threshold.")
    elif ergo >= 4:
        parts.append(f"Ergonomic risk elevated ({ergo}/10): "
                     "posture approaching unsafe range.")
    else:
        parts.append(f"Ergonomic risk normal ({ergo}/10).")
    if gait >= 7:
        parts.append(f"Gait instability high ({gait}/10): "
                     "fall-risk aisle avoidance activated in F4.")
    elif gait >= 4:
        parts.append(f"Gait moderate ({gait}/10).")
    else:
        parts.append(f"Gait stable ({gait}/10).")
    if band == "51+":
        parts.append("Senior band — 1.6x vulnerability weight.")
    elif band == "36-50":
        parts.append("Mid-age band — 1.3x vulnerability weight.")
    parts.append(f"Tier assigned: {tier}.")
    return " ".join(parts)


def generate_message(picker_id: str) -> dict:
    state  = picker_state[picker_id]
    frozen = random.random() < FREEZE_STATE_PROBABILITY

    if frozen:
        # Hold last valid scores — do not update base_risk
        ergo_score = state["last_ergo_score"]
        gait_score = state["last_gait_score"]
    else:
        # Posture risk × demographic weight + small noise
        raw = (state["base_risk"] * state["demographic_weight"]
               + random.uniform(-0.02, 0.05))
        raw = max(0.0, min(raw, 1.0))

        # Drift base_risk slightly each cycle to simulate posture changes
        state["base_risk"] = max(
            0.05, min(state["base_risk"] + random.uniform(-0.01, 0.02), 0.95)
        )

        ergo_score = round(1 + raw * 9)
        gait_raw   = raw * 0.6 + random.uniform(0.0, 0.1)
        gait_score = round(1 + min(gait_raw, 1.0) * 9)

        # Store for freeze holdback
        state["last_ergo_score"] = ergo_score
        state["last_gait_score"] = gait_score

    tier   = calculate_task_tier(ergo_score)
    reason = build_reason(
        ergo_score, gait_score, tier,
        state["demographic_band"], frozen
    )

    return {
        "picker_id":            picker_id,
        "timestamp":            datetime.now().isoformat(),
        "ergonomic_risk_score": ergo_score,         # int 1-10
        "gait_stability_score": gait_score,         # int 1-10
        "demographic_weight":   state["demographic_weight"],
        "task_tier":            tier,               # Heavy|Moderate|Light
        "reason":               reason,
        "freeze_state":         frozen,             # bool
    }


# ── Main ───────────────────────────────────────────────────────────────────

def run():
    print("=" * 65)
    print("Mock F3 Publisher")
    print(f"Channel         : {CHANNEL_F3_ERGONOMICS}")
    print(f"Pickers         : {len(PICKERS)}")
    print(f"Interval        : {PUBLISH_INTERVAL_SECONDS}s")
    print(f"Freeze rate     : {int(FREEZE_STATE_PROBABILITY * 100)}% per cycle")
    print("=" * 65)
    print()

    while True:
        for picker in PICKERS:
            msg = generate_message(picker["picker_id"])
            publish(CHANNEL_F3_ERGONOMICS, msg)

            freeze_tag = " [FROZEN — scores held]" if msg["freeze_state"] else ""
            print(
                f"[F3] {datetime.now().strftime('%H:%M:%S')} | "
                f"{msg['picker_id']} | "
                f"band={picker['demographic_band']:<5} | "
                f"ergo={msg['ergonomic_risk_score']:>2}/10 | "
                f"gait={msg['gait_stability_score']:>2}/10 | "
                f"tier={msg['task_tier']:<8} | "
                f"freeze={str(msg['freeze_state']):<5}"
                f"{freeze_tag}"
            )

        print(f"\n--- F3 cycle complete | sleeping {PUBLISH_INTERVAL_SECONDS}s ---\n")
        time.sleep(PUBLISH_INTERVAL_SECONDS)


if __name__ == "__main__":
    run()