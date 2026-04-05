from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

from openenv.core import State
from openenv.core.env_server import Action, Observation


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
class RetailReplenishState(State):
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

    episode_id: Optional[str] = None
    step_count: int


@dataclass
class RetailReplenishAction(Action):
    replenish: Dict[str, Dict[str, int]]  # store_id -> sku_id -> units
    emergency_order: dict  # sku_id -> units
    dc_reorder: Dict[str, int]  # sku_id -> units
    inter_store_transfer: Dict[
        str, Dict[str, Dict[str, int]]
    ]  # src -> dst -> sku -> units


@dataclass
class RetailReplenishObservation(Observation):
    store_inventory: List[List[int]]
    dc_inventory: List[int]
    expiry_countdown: List[List[int]]
    in_transit: List[List[int]]
    truck_capacity: List[int]
    supplier_status: List[bool]
    demand_forecast: List[List[List[float]]]
    current_day: int
    day_of_week: int

    reward_breakdown: Optional[Dict[str, Any]]
