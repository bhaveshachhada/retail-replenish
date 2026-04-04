from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
from .state import SKU, Store


@dataclass
class RewardWeights:
    sales_revenue: float = 1.0
    stockout_penalty: float = 2.0
    waste_penalty: float = 1.5
    overstock_penalty: float = 0.1
    emergency_penalty: float = 0.5
    transfer_cost: float = 0.3
    fill_rate_bonus: float = 5.0
    fill_rate_threshold: float = 0.95  # per-store fill rate to earn daily bonus


class RewardFunction:
    """
    Dense reward computed at every step.
    Provides signal over the full trajectory — not just episode end.
    """

    def __init__(
        self,
        skus: list[SKU],
        stores: list[Store],
        weights: RewardWeights | None = None,
    ):
        self.skus = {s.sku_id: s for s in skus}
        self.stores = {s.store_id: s for s in stores}
        self.w = weights or RewardWeights()

    def compute(
        self,
        demand: Dict[str, Dict[str, int]],
        sales: Dict[str, Dict[str, int]],
        stockouts: Dict[str, Dict[str, int]],
        waste: Dict[str, Dict[str, int]],
        store_inventory: Dict[str, Dict[str, int]],  # units after sales
        emergency_units: int,
        transferred_units: int,
    ) -> tuple[float, dict]:
        """
        Returns (total_reward, breakdown_dict).
        breakdown_dict is useful for logging and debugging.
        """
        rev = 0.0
        stockout_pen = 0.0
        waste_pen = 0.0
        overstock_pen = 0.0
        fill_bonus = 0.0

        for store_id in self.stores:
            store = self.stores[store_id]
            store_demand = sum(demand[store_id].values())
            store_sales = sum(sales[store_id].values())

            # Revenue
            for sku_id, sold in sales[store_id].items():
                rev += sold * self.skus[sku_id].unit_revenue

            # Stockout penalty (per unmet demand unit)
            for sku_id, unmet in stockouts[store_id].items():
                stockout_pen += unmet * self.w.stockout_penalty

            # Waste penalty (per expired perishable unit)
            for sku_id, wasted in waste[store_id].items():
                waste_pen += wasted * self.w.waste_penalty

            # Overstock penalty (units beyond shelf capacity)
            for sku_id, units in store_inventory[store_id].items():
                cap = store.sku_shelf_capacity.get(sku_id, float("inf"))
                overstock = max(0, units - cap)
                overstock_pen += overstock * self.w.overstock_penalty

            # Per-store fill rate bonus
            if store_demand > 0:
                store_fill = store_sales / store_demand
                if store_fill >= self.w.fill_rate_threshold:
                    fill_bonus += self.w.fill_rate_bonus

        emergency_pen = emergency_units * self.w.emergency_penalty
        transfer_pen = transferred_units * self.w.transfer_cost

        total = (
            rev
            - stockout_pen
            - waste_pen
            - overstock_pen
            - emergency_pen
            - transfer_pen
            + fill_bonus
        )

        breakdown = {
            "revenue": round(rev, 4),
            "stockout_penalty": round(-stockout_pen, 4),
            "waste_penalty": round(-waste_pen, 4),
            "overstock_penalty": round(-overstock_pen, 4),
            "emergency_penalty": round(-emergency_pen, 4),
            "transfer_cost": round(-transfer_pen, 4),
            "fill_rate_bonus": round(fill_bonus, 4),
            "total": round(total, 4),
        }

        return total, breakdown
