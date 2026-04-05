# env/state.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Step record (for graders / trajectory logging)
# ---------------------------------------------------------------------------


@dataclass
class StepRecord:
    day: int
    demand: Dict[str, Dict[str, int]]  # store_id -> sku_id -> units demanded
    sales: Dict[str, Dict[str, int]]  # store_id -> sku_id -> units sold
    stockouts: Dict[str, Dict[str, int]]  # store_id -> sku_id -> unmet demand
    waste: Dict[str, Dict[str, int]]  # store_id -> sku_id -> units expired
    emergency_units: int  # total emergency order units this step
    reward: float


@dataclass
class Trajectory:
    task_id: str
    steps: List[StepRecord] = field(default_factory=list)
    final_score: Optional[float] = None

    # Aggregate helpers used by graders
    def total_demand(self) -> int:
        return sum(
            v for s in self.steps for sv in s.demand.values() for v in sv.values()
        )

    def total_sales(self) -> int:
        return sum(
            v for s in self.steps for sv in s.sales.values() for v in sv.values()
        )

    def total_waste(self) -> int:
        return sum(
            v for s in self.steps for sv in s.waste.values() for v in sv.values()
        )

    def total_emergency_units(self) -> int:
        return sum(s.emergency_units for s in self.steps)

    def fill_rate(self, days: Optional[List[int]] = None) -> float:
        steps = [s for s in self.steps if days is None or s.day in days]
        demand = sum(v for s in steps for sv in s.demand.values() for v in sv.values())
        sales = sum(v for s in steps for sv in s.sales.values() for v in sv.values())
        return sales / demand if demand > 0 else 1.0

    def waste_rate(self, total_received: int) -> float:
        return self.total_waste() / total_received if total_received > 0 else 0.0
