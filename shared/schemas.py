from pydantic import BaseModel
from typing import Optional
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────

class InventoryMessage(BaseModel):
    item_code: str                    # TODO: confirm format e.g. "SKU-1042"
    shelf_location: str               # TODO: confirm format e.g. "A4-B7-S2"
    compliance_score: float           # TODO: confirm range 0-1 or 0-100
    exception_flag: bool              # TODO: confirm what triggers True
    confidence: float                 # TODO: confirm range 0-1
    timestamp: datetime


# ─────────────────────────────────────────────────────────────────────────────

class ErgonomicsMessage(BaseModel):
    picker_id: str                    # TODO: confirm format e.g. "P-001"
    ergonomic_risk_score: float       # TODO: confirm range and meaning
    fall_risk_score: float            # TODO: confirm if included here or separate
    demographic_weight: float         # TODO: confirm how this is calculated
    task_suitability: str             # TODO: confirm exact string values
    timestamp: datetime


# ─────────────────────────────────────────────────────────────────────────────

class RouteMessage(BaseModel):
    picker_id: str
    next_node: str
    path_sequence: list[str]
    estimated_time: float
    replan_reason: Optional[str]      # "stockout" | "congestion" | "fatigue" | None
    timestamp: datetime

