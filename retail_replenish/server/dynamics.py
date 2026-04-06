from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

from retail_replenish.models import (
    SKU,
    Store,
    RetailReplenishState,
    DeliveryOrder,
    SupplierOrder,
)


class DemandSimulator:
    """Generates stochastic daily demand per store/SKU."""

    DOW_MULTIPLIERS = [1.0, 1.0, 1.0, 1.0, 1.2, 1.5, 1.4]  # Mon–Sun

    def __init__(self, skus: List[SKU], stores: List[Store], rng: np.random.Generator):
        self.skus = {s.sku_id: s for s in skus}
        self.stores = {s.store_id: s for s in stores}
        self.rng = rng

    def sample(
        self,
        day: int,
        overrides: Dict[str, Dict[str, float]] | None = None,
    ) -> Dict[str, Dict[str, int]]:
        """
        Returns demand[store_id][sku_id] = units demanded.
        overrides: store_id -> sku_id -> demand multiplier (for surge events).
        """
        dow = day % 7
        dow_mult = self.DOW_MULTIPLIERS[dow]
        demand: Dict[str, Dict[str, int]] = {}

        for store in self.stores.values():
            demand[store.store_id] = {}
            for sku in self.skus.values():
                base = sku.base_demand_lambda * dow_mult
                # Apply event override if present
                if overrides and store.store_id in overrides:
                    base *= overrides[store.store_id].get(sku.sku_id, 1.0)
                units = int(self.rng.poisson(base))
                demand[store.store_id][sku.sku_id] = units

        return demand

    def forecast(
        self,
        day: int,
        horizon: int,
        overrides: Dict[int, Dict[str, Dict[str, float]]],
        noise_sigma: float = 0.15,
    ) -> Dict[str, Dict[str, List[float]]]:
        """
        Returns noisy forecast[store_id][sku_id] = [day+1 ... day+horizon] expected demand.
        """
        forecast: Dict[str, Dict[str, List[float]]] = {
            s.store_id: {sku.sku_id: [] for sku in self.skus.values()}
            for s in self.stores.values()
        }
        for offset in range(1, horizon + 1):
            future_day = day + offset
            dow = future_day % 7
            dow_mult = self.DOW_MULTIPLIERS[dow]
            day_overrides = overrides.get(future_day, {})
            for store in self.stores.values():
                for sku in self.skus.values():
                    base = sku.base_demand_lambda * dow_mult
                    if store.store_id in day_overrides:
                        base *= day_overrides[store.store_id].get(sku.sku_id, 1.0)
                    noise = self.rng.normal(1.0, noise_sigma)
                    forecast[store.store_id][sku.sku_id].append(max(0.0, base * noise))
        return forecast


class TransitionEngine:
    """Applies one day's transitions to RetailReplenishState, returns a StepRecord."""

    def __init__(self, skus: List[SKU], stores: List[Store]):
        self.skus = {s.sku_id: s for s in skus}
        self.stores = {s.store_id: s for s in stores}

    # ------------------------------------------------------------------
    # Phase 1: Resolve deliveries arriving today
    # ------------------------------------------------------------------
    def deliver_in_transit(self, state: RetailReplenishState) -> None:
        arriving = [o for o in state.in_transit if o.arrives_on_day == state.day]
        for order in arriving:
            inv = state.store_inventory[order.store_id][order.sku_id]
            sku = self.skus[order.sku_id]
            expires = state.day + sku.shelf_life_days if sku.is_perishable else -1
            inv.add_units(
                order.units, expires_on_day=expires, is_perishable=sku.is_perishable
            )
        state.in_transit = [
            o for o in state.in_transit if o.arrives_on_day != state.day
        ]

    # ------------------------------------------------------------------
    # Phase 2: Resolve DC reorders arriving today
    # ------------------------------------------------------------------
    def receive_dc_orders(self, state: RetailReplenishState) -> None:
        arriving = [o for o in state.pending_dc_orders if o.arrives_on_day == state.day]
        for order in arriving:
            state.dc_inventory[order.sku_id] = (
                state.dc_inventory.get(order.sku_id, 0) + order.units
            )
        state.pending_dc_orders = [
            o for o in state.pending_dc_orders if o.arrives_on_day != state.day
        ]

    # ------------------------------------------------------------------
    # Phase 3: Expire perishable units at stores
    # ------------------------------------------------------------------
    def expire_perishables(
        self, state: RetailReplenishState
    ) -> Dict[str, Dict[str, int]]:
        waste: Dict[str, Dict[str, int]] = {}
        for store_id, sku_inv in state.store_inventory.items():
            waste[store_id] = {}
            for sku_id, inv in sku_inv.items():
                sku = self.skus[sku_id]
                if sku.is_perishable:
                    expired = inv.expire_units(state.day)
                    waste[store_id][sku_id] = expired
                else:
                    waste[store_id][sku_id] = 0
        return waste

    # ------------------------------------------------------------------
    # Phase 4: Validate & apply agent actions
    # ------------------------------------------------------------------
    def apply_replenishment(
        self,
        state: RetailReplenishState,
        replenish: Dict[str, Dict[str, int]],  # store_id -> sku_id -> units
        emergency_reorder: Dict[str, int],  # sku_id -> units
        dc_reorder: Dict[str, int],  # sku_id -> units
        inter_store: Dict[str, Dict[str, Dict[str, int]]],  # src -> dst -> sku -> units
    ) -> Tuple[int, int]:
        """
        Dispatches deliveries, places reorders.
        Returns (total_emergency_units, total_transferred_units).
        """
        emergency_total = 0
        transferred_total = 0

        # 1. Standard DC replenishment to stores
        for store_id, sku_map in replenish.items():
            store = self.stores[store_id]
            truck_used = 0
            for sku_id, units in sku_map.items():
                if units <= 0:
                    continue
                available = state.dc_inventory.get(sku_id, 0)
                cap_left = store.truck_capacity - truck_used
                actual = min(units, available, cap_left)
                if actual > 0:
                    state.dc_inventory[sku_id] -= actual
                    truck_used += actual
                    state.in_transit.append(
                        DeliveryOrder(
                            store_id=store_id,
                            sku_id=sku_id,
                            units=actual,
                            arrives_on_day=state.day + 1,
                            is_emergency=False,
                        )
                    )

        # 2. Standard DC reorder from supplier
        for sku_id, units in dc_reorder.items():
            if units <= 0:
                continue
            supplier = self._find_supplier_for_sku(state, sku_id)
            if supplier is None:
                continue
            lead = self._get_lead_time(state, sku_id)
            state.pending_dc_orders.append(
                SupplierOrder(
                    sku_id=sku_id,
                    units=units,
                    arrives_on_day=state.day + lead,
                    is_emergency=False,
                )
            )

        # 3. Emergency DC reorder (1-day lead, 2x cost captured in reward)
        for sku_id, units in emergency_reorder.items():
            if units <= 0:
                continue
            emergency_total += units
            state.pending_dc_orders.append(
                SupplierOrder(
                    sku_id=sku_id,
                    units=units,
                    arrives_on_day=state.day + 1,
                    is_emergency=True,
                )
            )

        # 4. Inter-store transfers (arrive same day + 1)
        for src_id, dst_map in inter_store.items():
            for dst_id, sku_map in dst_map.items():
                for sku_id, units in sku_map.items():
                    if units <= 0:
                        continue
                    src_inv = state.store_inventory[src_id][sku_id]
                    actual = min(units, src_inv.units)
                    if actual > 0:
                        src_inv.consume_units(actual)
                        transferred_total += actual
                        state.in_transit.append(
                            DeliveryOrder(
                                store_id=dst_id,
                                sku_id=sku_id,
                                units=actual,
                                arrives_on_day=state.day + 1,
                                is_emergency=False,
                            )
                        )

        return emergency_total, transferred_total

    # ------------------------------------------------------------------
    # Phase 5: Simulate demand & compute sales / stockouts
    # ------------------------------------------------------------------
    def simulate_sales(
        self,
        state: RetailReplenishState,
        demand: Dict[str, Dict[str, int]],
    ) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, int]]]:
        """
        Returns (sales, stockouts) both shaped store_id -> sku_id -> units.
        """
        sales: Dict[str, Dict[str, int]] = {}
        stockouts: Dict[str, Dict[str, int]] = {}

        for store_id, sku_demand in demand.items():
            sales[store_id] = {}
            stockouts[store_id] = {}
            for sku_id, demanded in sku_demand.items():
                inv = state.store_inventory[store_id][sku_id]
                sold = inv.consume_units(demanded)
                sales[store_id][sku_id] = sold
                stockouts[store_id][sku_id] = demanded - sold

        return sales, stockouts

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _find_supplier_for_sku(self, state: RetailReplenishState, sku_id: str):
        # Returns first operational supplier that handles this SKU
        # Supplier metadata is held externally; here we just check status dict keys
        # For simplicity, if supplier is disrupted, return None
        for sup_id, operational in state.supplier_status.items():
            if operational:
                return sup_id
        return None

    def _get_lead_time(self, state: RetailReplenishState, sku_id: str) -> int:
        # Default lead time 2 days; overridable via task config
        return 2
