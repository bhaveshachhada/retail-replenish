# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Retail Replenish Environment.

The retail_replenish environment is a simple test environment that echoes back messages.
"""

from dataclasses import field
from typing import Dict, List, Optional, Any

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Domain entities
# ---------------------------------------------------------------------------


class SKU(BaseModel):
    sku_id: str = Field(...)
    name: str = Field(...)
    shelf_life_days: int = Field(...)  # -1 = non-perishable
    unit_cost: float = Field(...)
    unit_revenue: float = Field(...)
    base_demand_lambda: float = Field(
        ...
    )  # Poisson mean units/day (single store baseline)

    @property
    def is_perishable(self) -> bool:
        return self.shelf_life_days > 0


class Store(BaseModel):
    store_id: str = Field(...)
    sku_shelf_capacity: Dict[str, int] = Field(
        default_factory=dict
    )  # sku_id -> max units on shelf
    truck_capacity: int = Field(...)  # max total units deliverable per day


class Supplier(BaseModel):
    supplier_id: str = Field(...)
    sku_ids: List[str] = Field(default_factory=list)  # SKUs this supplier provides
    lead_time_days: int = Field(...)
    is_operational: bool = Field(default=True)


# ---------------------------------------------------------------------------
# Inventory tracking
# ---------------------------------------------------------------------------


class ExpiryBatch(BaseModel):
    """A batch of perishable units with a specific expiry day."""

    units: int = Field(...)
    expires_on_day: int = Field(...)


class DeliveryOrder(BaseModel):
    """In-transit replenishment from DC to a store."""

    store_id: str = Field(...)
    sku_id: str = Field(...)
    units: int = Field(...)
    arrives_on_day: int = Field(...)
    is_emergency: bool = Field(default=False)


class SupplierOrder(BaseModel):
    """Pending reorder from supplier to DC."""

    sku_id: str = Field(...)
    units: int = Field(...)
    arrives_on_day: int = Field(...)
    is_emergency: bool = Field(default=False)


# ---------------------------------------------------------------------------
# Core environment state
# ---------------------------------------------------------------------------


class StoreInventory(BaseModel):
    """Per-store inventory, tracking batches for perishables."""

    units: int = Field(default=0)
    batches: List[ExpiryBatch] = Field(default_factory=list)

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


class RetailReplenishState(State):
    day: int = Field(...)
    # store_id -> sku_id -> StoreInventory
    store_inventory: Dict[str, Dict[str, StoreInventory]] = Field(default_factory=dict)
    # sku_id -> units at DC
    dc_inventory: Dict[str, int] = Field(default_factory=dict)
    in_transit: List[DeliveryOrder] = Field(default_factory=list)
    pending_dc_orders: List[SupplierOrder] = Field(default_factory=list)
    supplier_status: Dict[str, bool] = Field(
        default_factory=dict
    )  # supplier_id -> operational
    # Demand event overrides: day -> store_id -> sku_id -> multiplier
    demand_overrides: Dict[int, Dict[str, Dict[str, float]]] = field(
        default_factory=dict
    )

    episode_id: Optional[str] = Field(default=None)
    step_count: int = Field(...)


class RetailReplenishAction(Action):
    replenish: Dict[str, Dict[str, int]] = Field(
        default_factory=dict
    )  # store_id -> sku_id -> units
    emergency_order: Dict[str, int] = Field(default_factory=dict)  # sku_id -> units
    dc_reorder: Dict[str, int] = Field(default_factory=dict)  # sku_id -> units
    inter_store_transfer: Dict[str, Dict[str, Dict[str, int]]] = Field(
        default_factory=dict
    )  # src -> dst -> sku -> units


class RetailReplenishObservation(Observation):
    store_inventory: List[List[int]] = Field(default_factory=list)
    dc_inventory: List[int] = Field(default_factory=list)
    expiry_countdown: List[List[int]] = Field(default_factory=list)
    in_transit: List[List[int]] = Field(default_factory=list)
    truck_capacity: List[int] = Field(default_factory=list)
    supplier_status: List[bool] = Field(default_factory=dict)
    demand_forecast: List[List[List[float]]] = Field(default_factory=list)
    current_day: int = Field(...)
    day_of_week: int = Field(...)

    reward_breakdown: Optional[Dict[str, Any]] = Field(default_factory=dict)
