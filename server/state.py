# env/state.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Domain entities
# ---------------------------------------------------------------------------


@dataclass
class SKU:
    sku_id: str
    name: str
    shelf_life_days: int  # -1 = non-perishable
    unit_cost: float
    unit_revenue: float
    base_demand_lambda: float  # Poisson mean units/day (single store baseline)

    @property
    def is_perishable(self) -> bool:
        return self.shelf_life_days > 0


@dataclass
class Store:
    store_id: str
    sku_shelf_capacity: Dict[str, int]  # sku_id -> max units on shelf
    truck_capacity: int  # max total units deliverable per day


@dataclass
class Supplier:
    supplier_id: str
    sku_ids: List[str]  # SKUs this supplier provides
    lead_time_days: int
    is_operational: bool = True


# ---------------------------------------------------------------------------
# Inventory tracking
# ---------------------------------------------------------------------------


@dataclass
class ExpiryBatch:
    """A batch of perishable units with a specific expiry day."""

    units: int
    expires_on_day: int


@dataclass
class DeliveryOrder:
    """In-transit replenishment from DC to a store."""

    store_id: str
    sku_id: str
    units: int
    arrives_on_day: int
    is_emergency: bool = False


@dataclass
class SupplierOrder:
    """Pending reorder from supplier to DC."""

    sku_id: str
    units: int
    arrives_on_day: int
    is_emergency: bool = False


# ---------------------------------------------------------------------------
# Core environment state
# ---------------------------------------------------------------------------


@dataclass
class StoreInventory:
    """Per-store inventory, tracking batches for perishables."""

    units: int = 0
    batches: List[ExpiryBatch] = field(default_factory=list)

    def add_units(
        self, units: int, expires_on_day: int = -1, is_perishable: bool = False
    ) -> None:
        self.units += units
        if is_perishable and expires_on_day > 0:
            self.batches.append(ExpiryBatch(units=units, expires_on_day=expires_on_day))

    def consume_units(self, units: int) -> int:
        """Consume units (FIFO from oldest batch). Returns actual units consumed."""
        consumed = min(units, self.units)
        self.units -= consumed
        remaining = consumed
        for batch in sorted(self.batches, key=lambda b: b.expires_on_day):
            if remaining <= 0:
                break
            take = min(batch.units, remaining)
            batch.units -= take
            remaining -= take
        self.batches = [b for b in self.batches if b.units > 0]
        return consumed

    def expire_units(self, current_day: int) -> int:
        """Remove expired batches. Returns number of wasted units."""
        expired = sum(b.units for b in self.batches if b.expires_on_day <= current_day)
        self.batches = [b for b in self.batches if b.expires_on_day > current_day]
        self.units = max(0, self.units - expired)
        return expired

    def days_until_next_expiry(self, current_day: int) -> int:
        """Returns days until oldest batch expires, or -1 if no perishables."""
        valid = [b for b in self.batches if b.units > 0]
        if not valid:
            return -1
        return min(b.expires_on_day for b in valid) - current_day


@dataclass
class EnvState:
    day: int
    # store_id -> sku_id -> StoreInventory
    store_inventory: Dict[str, Dict[str, StoreInventory]]
    # sku_id -> units at DC
    dc_inventory: Dict[str, int]
    in_transit: List[DeliveryOrder]
    pending_dc_orders: List[SupplierOrder]
    supplier_status: Dict[str, bool]  # supplier_id -> operational
    # Demand event overrides: day -> store_id -> sku_id -> multiplier
    demand_overrides: Dict[int, Dict[str, Dict[str, float]]] = field(
        default_factory=dict
    )


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
