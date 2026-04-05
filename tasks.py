# tasks/task1_single_store.py
"""
Task 1 — Single Store, Stable Demand (Easy)
============================================
One store, 5 non-perishable SKUs, predictable demand, unlimited truck capacity.
The agent simply needs to learn when and how much to reorder to maintain availability.

Task 2 — Multi-Store, Mixed Perishable Goods (Medium)
======================================================
4 stores, 15 SKUs (8 non-perishable + 7 perishable with 3-5 day shelf life).
Weekend demand spikes active. Truck capacity constrained at 200 units/store/day.
Agent must balance fill rate AND minimise perishable waste simultaneously.

Task 3 — Disruption Response (Hard)
=====================================
6 stores, 20 SKUs (10 perishable), 28-day episode.

Two simultaneous crises hit on day 10:
  - SUP-1 (primary supplier) goes offline for 5 days (days 10-14)
  - Stores 1, 3, 5 experience an 80% demand surge for 4 days (days 10-13)

The agent must pre-position stock before the crisis, triage during it,
use emergency orders judiciously, and recover gracefully afterwards.
"""

from __future__ import annotations

from retail_replenish.models import SKU, Store, Supplier
from retail_replenish.server.retail_env import TaskConfig


def make_task1_config() -> TaskConfig:
    skus = [
        SKU(
            sku_id="SKU-A",
            name="Bottled Water 1L",
            shelf_life_days=-1,
            unit_cost=0.3,
            unit_revenue=0.8,
            base_demand_lambda=15.0,
        ),
        SKU(
            sku_id="SKU-B",
            name="White Rice 1kg",
            shelf_life_days=-1,
            unit_cost=0.5,
            unit_revenue=1.2,
            base_demand_lambda=12.0,
        ),
        SKU(
            sku_id="SKU-C",
            name="Canned Tomatoes",
            shelf_life_days=-1,
            unit_cost=0.4,
            unit_revenue=1.0,
            base_demand_lambda=10.0,
        ),
        SKU(
            sku_id="SKU-D",
            name="Pasta 500g",
            shelf_life_days=-1,
            unit_cost=0.6,
            unit_revenue=1.4,
            base_demand_lambda=18.0,
        ),
        SKU(
            sku_id="SKU-E",
            name="Laundry Detergent",
            shelf_life_days=-1,
            unit_cost=1.2,
            unit_revenue=3.0,
            base_demand_lambda=8.0,
        ),
    ]

    stores = [
        Store(
            store_id="STORE-1",
            sku_shelf_capacity={s.sku_id: 200 for s in skus},
            truck_capacity=9999,  # unlimited for Task 1
        )
    ]

    suppliers = [
        Supplier(
            supplier_id="SUP-1",
            sku_ids=[s.sku_id for s in skus],
            lead_time_days=2,
            is_operational=True,
        )
    ]

    # DC starts with 7 days worth of average demand per SKU
    initial_dc = {s.sku_id: int(s.base_demand_lambda * 7) for s in skus}

    # Stores start with 2 days of stock
    initial_store = {"STORE-1": {s.sku_id: int(s.base_demand_lambda * 2) for s in skus}}

    return TaskConfig(
        task_id="task1_single_store_stable",
        skus=skus,
        stores=stores,
        suppliers=suppliers,
        episode_days=7,
        initial_dc_inventory=initial_dc,
        initial_store_inventory=initial_store,
        demand_overrides={},  # no demand events
        supplier_disruptions={},  # no disruptions
        forecast_horizon=3,
    )


def make_task2_config() -> TaskConfig:
    skus = [
        # --- Non-perishables ---
        SKU(
            sku_id="SKU-A",
            name="Bottled Water 1L",
            shelf_life_days=-1,
            unit_cost=0.3,
            unit_revenue=0.8,
            base_demand_lambda=15.0,
        ),
        SKU(
            sku_id="SKU-B",
            name="White Rice 1kg",
            shelf_life_days=-1,
            unit_cost=0.5,
            unit_revenue=1.2,
            base_demand_lambda=12.0,
        ),
        SKU(
            sku_id="SKU-C",
            name="Canned Tomatoes",
            shelf_life_days=-1,
            unit_cost=0.4,
            unit_revenue=1.0,
            base_demand_lambda=10.0,
        ),
        SKU(
            sku_id="SKU-D",
            name="Pasta 500g",
            shelf_life_days=-1,
            unit_cost=0.6,
            unit_revenue=1.4,
            base_demand_lambda=18.0,
        ),
        SKU(
            sku_id="SKU-E",
            name="Laundry Detergent",
            shelf_life_days=-1,
            unit_cost=1.2,
            unit_revenue=3.0,
            base_demand_lambda=8.0,
        ),
        SKU(
            sku_id="SKU-F",
            name="Paper Towels",
            shelf_life_days=-1,
            unit_cost=0.8,
            unit_revenue=2.0,
            base_demand_lambda=9.0,
        ),
        SKU(
            sku_id="SKU-G",
            name="Cooking Oil 1L",
            shelf_life_days=-1,
            unit_cost=1.0,
            unit_revenue=2.5,
            base_demand_lambda=7.0,
        ),
        SKU(
            sku_id="SKU-H",
            name="Coffee 250g",
            shelf_life_days=-1,
            unit_cost=2.0,
            unit_revenue=5.0,
            base_demand_lambda=6.0,
        ),
        # --- Perishables ---
        SKU(
            sku_id="SKU-P1",
            name="Whole Milk 1L",
            shelf_life_days=3,
            unit_cost=0.7,
            unit_revenue=1.5,
            base_demand_lambda=20.0,
        ),
        SKU(
            sku_id="SKU-P2",
            name="Sliced Bread",
            shelf_life_days=3,
            unit_cost=0.6,
            unit_revenue=1.3,
            base_demand_lambda=18.0,
        ),
        SKU(
            sku_id="SKU-P3",
            name="Greek Yogurt",
            shelf_life_days=4,
            unit_cost=0.9,
            unit_revenue=2.0,
            base_demand_lambda=12.0,
        ),
        SKU(
            sku_id="SKU-P4",
            name="Chicken Breast 1kg",
            shelf_life_days=4,
            unit_cost=3.5,
            unit_revenue=7.0,
            base_demand_lambda=8.0,
        ),
        SKU(
            sku_id="SKU-P5",
            name="Baby Spinach 200g",
            shelf_life_days=3,
            unit_cost=0.8,
            unit_revenue=2.2,
            base_demand_lambda=10.0,
        ),
        SKU(
            sku_id="SKU-P6",
            name="Orange Juice 1L",
            shelf_life_days=5,
            unit_cost=1.0,
            unit_revenue=2.5,
            base_demand_lambda=14.0,
        ),
        SKU(
            sku_id="SKU-P7",
            name="Cheddar Cheese 400g",
            shelf_life_days=5,
            unit_cost=2.0,
            unit_revenue=4.5,
            base_demand_lambda=9.0,
        ),
    ]

    stores = [
        Store(
            store_id="STORE-1",
            sku_shelf_capacity={s.sku_id: 150 for s in skus},
            truck_capacity=200,
        ),
        Store(
            store_id="STORE-2",
            sku_shelf_capacity={s.sku_id: 150 for s in skus},
            truck_capacity=200,
        ),
        Store(
            store_id="STORE-3",
            sku_shelf_capacity={s.sku_id: 150 for s in skus},
            truck_capacity=200,
        ),
        Store(
            store_id="STORE-4",
            sku_shelf_capacity={s.sku_id: 150 for s in skus},
            truck_capacity=200,
        ),
    ]

    suppliers = [
        Supplier(
            supplier_id="SUP-1",
            sku_ids=[s.sku_id for s in skus],
            lead_time_days=2,
            is_operational=True,
        )
    ]

    # DC starts with 10 days of average demand across all SKUs
    initial_dc = {s.sku_id: int(s.base_demand_lambda * 10 * len(stores)) for s in skus}

    # Stores start with 2 days of stock (perishables start with 1 day to create tension)
    initial_store = {}
    for store in stores:
        initial_store[store.store_id] = {}
        for s in skus:
            days = 1 if s.is_perishable else 2
            initial_store[store.store_id][s.sku_id] = int(s.base_demand_lambda * days)

    # Weekend spikes: days 5 (Sat) and 6 (Sun) have 40% demand increase on all perishables
    perishable_ids = [s.sku_id for s in skus if s.is_perishable]
    demand_overrides = {}
    for spike_day in [5, 6, 12, 13]:  # two weekends in 14-day episode
        demand_overrides[spike_day] = {
            store.store_id: {sku_id: 1.4 for sku_id in perishable_ids}
            for store in stores
        }

    return TaskConfig(
        task_id="task2_multi_store_perishables",
        skus=skus,
        stores=stores,
        suppliers=suppliers,
        episode_days=14,
        initial_dc_inventory=initial_dc,
        initial_store_inventory=initial_store,
        demand_overrides=demand_overrides,
        supplier_disruptions={},
        forecast_horizon=3,
    )


def make_task3_config() -> TaskConfig:
    skus = [
        # Non-perishables (10)
        SKU(
            sku_id="SKU-A",
            name="Bottled Water 1L",
            shelf_life_days=-1,
            unit_cost=0.3,
            unit_revenue=0.8,
            base_demand_lambda=15.0,
        ),
        SKU(
            sku_id="SKU-B",
            name="White Rice 1kg",
            shelf_life_days=-1,
            unit_cost=0.5,
            unit_revenue=1.2,
            base_demand_lambda=12.0,
        ),
        SKU(
            sku_id="SKU-C",
            name="Canned Tomatoes",
            shelf_life_days=-1,
            unit_cost=0.4,
            unit_revenue=1.0,
            base_demand_lambda=10.0,
        ),
        SKU(
            sku_id="SKU-D",
            name="Pasta 500g",
            shelf_life_days=-1,
            unit_cost=0.6,
            unit_revenue=1.4,
            base_demand_lambda=18.0,
        ),
        SKU(
            sku_id="SKU-E",
            name="Laundry Detergent",
            shelf_life_days=-1,
            unit_cost=1.2,
            unit_revenue=3.0,
            base_demand_lambda=8.0,
        ),
        SKU(
            sku_id="SKU-F",
            name="Paper Towels",
            shelf_life_days=-1,
            unit_cost=0.8,
            unit_revenue=2.0,
            base_demand_lambda=9.0,
        ),
        SKU(
            sku_id="SKU-G",
            name="Cooking Oil 1L",
            shelf_life_days=-1,
            unit_cost=1.0,
            unit_revenue=2.5,
            base_demand_lambda=7.0,
        ),
        SKU(
            sku_id="SKU-H",
            name="Coffee 250g",
            shelf_life_days=-1,
            unit_cost=2.0,
            unit_revenue=5.0,
            base_demand_lambda=6.0,
        ),
        SKU(
            sku_id="SKU-I",
            name="Toilet Paper 4pk",
            shelf_life_days=-1,
            unit_cost=1.0,
            unit_revenue=2.8,
            base_demand_lambda=11.0,
        ),
        SKU(
            sku_id="SKU-J",
            name="Dish Soap 500ml",
            shelf_life_days=-1,
            unit_cost=0.9,
            unit_revenue=2.2,
            base_demand_lambda=7.0,
        ),
        # Perishables (10)
        SKU(
            sku_id="SKU-P1",
            name="Whole Milk 1L",
            shelf_life_days=3,
            unit_cost=0.7,
            unit_revenue=1.5,
            base_demand_lambda=20.0,
        ),
        SKU(
            sku_id="SKU-P2",
            name="Sliced Bread",
            shelf_life_days=3,
            unit_cost=0.6,
            unit_revenue=1.3,
            base_demand_lambda=18.0,
        ),
        SKU(
            sku_id="SKU-P3",
            name="Greek Yogurt",
            shelf_life_days=4,
            unit_cost=0.9,
            unit_revenue=2.0,
            base_demand_lambda=12.0,
        ),
        SKU(
            sku_id="SKU-P4",
            name="Chicken Breast 1kg",
            shelf_life_days=4,
            unit_cost=3.5,
            unit_revenue=7.0,
            base_demand_lambda=8.0,
        ),
        SKU(
            sku_id="SKU-P5",
            name="Baby Spinach 200g",
            shelf_life_days=3,
            unit_cost=0.8,
            unit_revenue=2.2,
            base_demand_lambda=10.0,
        ),
        SKU(
            sku_id="SKU-P6",
            name="Orange Juice 1L",
            shelf_life_days=5,
            unit_cost=1.0,
            unit_revenue=2.5,
            base_demand_lambda=14.0,
        ),
        SKU(
            sku_id="SKU-P7",
            name="Cheddar Cheese 400g",
            shelf_life_days=5,
            unit_cost=2.0,
            unit_revenue=4.5,
            base_demand_lambda=9.0,
        ),
        SKU(
            sku_id="SKU-P8",
            name="Eggs 12pk",
            shelf_life_days=5,
            unit_cost=1.5,
            unit_revenue=3.5,
            base_demand_lambda=16.0,
        ),
        SKU(
            sku_id="SKU-P9",
            name="Butter 250g",
            shelf_life_days=5,
            unit_cost=1.8,
            unit_revenue=4.0,
            base_demand_lambda=10.0,
        ),
        SKU(
            sku_id="SKU-P10",
            name="Cream Cheese 200g",
            shelf_life_days=4,
            unit_cost=1.2,
            unit_revenue=3.0,
            base_demand_lambda=7.0,
        ),
    ]

    stores = [
        Store(
            store_id=f"STORE-{i}",
            sku_shelf_capacity={s.sku_id: 200 for s in skus},
            truck_capacity=250,
        )
        for i in range(1, 7)
    ]

    suppliers = [
        Supplier(
            supplier_id="SUP-1",
            sku_ids=[s.sku_id for s in skus],
            lead_time_days=2,
            is_operational=True,
        ),
        # Backup supplier: higher cost (captured in reward weights), longer lead, limited SKUs
        Supplier(
            supplier_id="SUP-2",
            sku_ids=[
                s.sku_id for s in skus if not s.is_perishable
            ],  # non-perishables only
            lead_time_days=4,
            is_operational=True,
        ),
    ]

    # DC starts with 14 days of average demand (agent needs buffer to pre-position)
    initial_dc = {s.sku_id: int(s.base_demand_lambda * 14 * len(stores)) for s in skus}

    # Stores start with 3 days stock (gives agent room to manage pre-disruption)
    initial_store = {}
    for store in stores:
        initial_store[store.store_id] = {
            s.sku_id: int(s.base_demand_lambda * (1 if s.is_perishable else 3))
            for s in skus
        }

    # --- Demand overrides ---
    # Days 5-6 and 12-13: regular weekend spikes (1.4x, all stores, all perishables)
    perishable_ids = [s.sku_id for s in skus if s.is_perishable]
    demand_overrides: dict = {}

    for spike_day in [5, 6, 12, 13, 19, 20, 26, 27]:  # 4 weekends
        demand_overrides[spike_day] = {
            store.store_id: {sku_id: 1.4 for sku_id in perishable_ids}
            for store in stores
        }

    # Days 10-13: SURGE on stores 1, 3, 5 (1.8x ALL SKUs — crisis event)
    surge_stores = ["STORE-1", "STORE-3", "STORE-5"]
    all_sku_ids = [s.sku_id for s in skus]
    for surge_day in [10, 11, 12, 13]:
        if surge_day not in demand_overrides:
            demand_overrides[surge_day] = {}
        for sid in surge_stores:
            # Merge with any existing weekend multiplier
            existing = demand_overrides[surge_day].get(sid, {})
            for sku_id in all_sku_ids:
                base_mult = existing.get(sku_id, 1.0)
                demand_overrides[surge_day].setdefault(sid, {})[sku_id] = (
                    base_mult * 1.8
                )

    # --- Supplier disruptions ---
    # Day 10: SUP-1 goes offline. Day 15: SUP-1 comes back online.
    supplier_disruptions = {
        10: {"SUP-1": False},
        15: {"SUP-1": True},
    }

    return TaskConfig(
        task_id="task3_disruption_response",
        skus=skus,
        stores=stores,
        suppliers=suppliers,
        episode_days=28,
        initial_dc_inventory=initial_dc,
        initial_store_inventory=initial_store,
        demand_overrides=demand_overrides,
        supplier_disruptions=supplier_disruptions,
        forecast_horizon=5,  # wider forecast window to help agent anticipate
    )
