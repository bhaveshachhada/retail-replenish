from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .state import (
    EnvState,
    SKU,
    Store,
    Supplier,
    StoreInventory,
    StepRecord,
    Trajectory,
)
from .dynamics import DemandSimulator, TransitionEngine
from .reward import RewardFunction, RewardWeights


# ---------------------------------------------------------------------------
# Task configuration
# ---------------------------------------------------------------------------


class TaskConfig:
    def __init__(
        self,
        task_id: str,
        skus: List[SKU],
        stores: List[Store],
        suppliers: List[Supplier],
        episode_days: int,
        initial_dc_inventory: Dict[str, int],
        initial_store_inventory: Dict[str, Dict[str, int]],
        demand_overrides: Dict[
            int, Dict[str, Dict[str, float]]
        ],  # day->store->sku->mult
        supplier_disruptions: Dict[int, Dict[str, bool]],  # day->supplier->status
        forecast_horizon: int = 3,
        reward_weights: Optional[RewardWeights] = None,
    ):
        self.task_id = task_id
        self.skus = skus
        self.stores = stores
        self.suppliers = suppliers
        self.episode_days = episode_days
        self.initial_dc_inventory = initial_dc_inventory
        self.initial_store_inventory = initial_store_inventory
        self.demand_overrides = demand_overrides
        self.supplier_disruptions = supplier_disruptions
        self.forecast_horizon = forecast_horizon
        self.reward_weights = reward_weights


# ---------------------------------------------------------------------------
# Observation & Action types
# ---------------------------------------------------------------------------

Observation = Dict[str, Any]
Action = Dict[str, Any]
Info = Dict[str, Any]


# ---------------------------------------------------------------------------
# Main environment
# ---------------------------------------------------------------------------


class RetailReplenishEnv:
    """
    OpenEnv-compliant retail inventory replenishment environment.

    Follows the standard gym-style interface:
        obs, info = env.reset(seed=42)
        obs, reward, terminated, truncated, info = env.step(action)
    """

    metadata = {"render_modes": ["text", "dict"], "version": "1.0.0"}

    def __init__(self, config: TaskConfig):
        self.config = config
        self._skus = {s.sku_id: s for s in config.skus}
        self._stores = {s.store_id: s for s in config.stores}

        self._state: Optional[EnvState] = None
        self._trajectory: Optional[Trajectory] = None
        self._rng: Optional[np.random.Generator] = None
        self._total_received: int = (
            0  # tracks units received at stores (for waste rate)
        )

        self._demand_sim: Optional[DemandSimulator] = None
        self._transition: TransitionEngine = TransitionEngine(
            config.skus, config.stores
        )
        self._reward_fn: RewardFunction = RewardFunction(
            config.skus, config.stores, config.reward_weights
        )

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> Tuple[Observation, Info]:
        self._rng = np.random.default_rng(seed)
        self._demand_sim = DemandSimulator(
            self.config.skus, self.config.stores, self._rng
        )
        self._total_received = 0

        # Build initial store inventory
        store_inv: Dict[str, Dict[str, StoreInventory]] = {}
        for store in self.config.stores:
            store_inv[store.store_id] = {}
            for sku in self.config.skus:
                init_units = self.config.initial_store_inventory.get(
                    store.store_id, {}
                ).get(sku.sku_id, 0)
                si = StoreInventory(units=init_units)
                if sku.is_perishable and init_units > 0:
                    si.batches.append(
                        type(
                            "ExpiryBatch",
                            (),
                            {
                                "units": init_units,
                                "expires_on_day": sku.shelf_life_days,
                            },
                        )()
                    )
                store_inv[store.store_id][sku.sku_id] = si
                self._total_received += init_units

        self._state = EnvState(
            day=0,
            store_inventory=store_inv,
            dc_inventory=dict(self.config.initial_dc_inventory),
            in_transit=[],
            pending_dc_orders=[],
            supplier_status={
                s.supplier_id: s.is_operational for s in self.config.suppliers
            },
            demand_overrides=self.config.demand_overrides,
        )

        self._trajectory = Trajectory(task_id=self.config.task_id)
        obs = self._build_observation()
        return obs, {"day": 0, "task_id": self.config.task_id}

    def step(self, action: Action) -> Tuple[Observation, float, bool, bool, Info]:
        assert self._state is not None, "Call reset() before step()."

        state = self._state

        # Apply any supplier disruption events for today
        disruptions = self.config.supplier_disruptions.get(state.day, {})
        for sup_id, status in disruptions.items():
            state.supplier_status[sup_id] = status

        # --- Transition phases ---
        self._transition.deliver_in_transit(state)
        self._transition.receive_dc_orders(state)
        waste = self._transition.expire_perishables(state)

        replenish = action.get("replenish", {})
        dc_reorder = action.get("dc_reorder", {})
        emergency_order = action.get("emergency_reorder", {})
        inter_store = action.get("inter_store_transfer", {})

        emerg_units, transfer_units = self._transition.apply_replenishment(
            state, replenish, emergency_order, dc_reorder, inter_store
        )

        # Track received for waste rate computation
        self._total_received += sum(
            o.units for o in state.in_transit if o.arrives_on_day == state.day + 1
        )

        # Demand & sales
        day_overrides = state.demand_overrides.get(state.day, {})
        demand = self._demand_sim.sample(state.day, overrides=day_overrides)
        sales, stockouts = self._transition.simulate_sales(state, demand)

        # Current inventory levels (post-sales) for overstock check
        curr_inv = {
            sid: {skuid: inv.units for skuid, inv in sku_map.items()}
            for sid, sku_map in state.store_inventory.items()
        }

        reward, breakdown = self._reward_fn.compute(
            demand, sales, stockouts, waste, curr_inv, emerg_units, transfer_units
        )

        record = StepRecord(
            day=state.day,
            demand=demand,
            sales=sales,
            stockouts=stockouts,
            waste=waste,
            emergency_units=emerg_units,
            reward=reward,
        )
        self._trajectory.steps.append(record)

        state.day += 1
        terminated = state.day >= self.config.episode_days
        obs = self._build_observation()

        info: Info = {
            "day": state.day,
            "reward_breakdown": breakdown,
            "fill_rate": self._trajectory.fill_rate(),
            "total_waste": self._trajectory.total_waste(),
        }

        return obs, reward, terminated, False, info

    def render(self, mode: str = "text") -> str | dict:
        if self._state is None:
            return "Environment not initialized. Call reset()."
        s = self._state
        if mode == "dict":
            return {
                "day": s.day,
                "dc_inventory": s.dc_inventory,
                "store_inventory": {
                    sid: {k: v.units for k, v in skus.items()}
                    for sid, skus in s.store_inventory.items()
                },
                "supplier_status": s.supplier_status,
                "in_transit_count": len(s.in_transit),
            }
        # text mode
        lines = [f"=== Day {s.day} / {self.config.episode_days} ==="]
        lines.append(f"DC Inventory: { {k: v for k, v in s.dc_inventory.items()} }")
        for sid, skus in s.store_inventory.items():
            inv_str = ", ".join(f"{k}:{v.units}" for k, v in skus.items())
            lines.append(f"  Store {sid}: [{inv_str}]")
        lines.append(f"Supplier status: {s.supplier_status}")
        lines.append(f"In-transit orders: {len(s.in_transit)}")
        return "\n".join(lines)

    def get_task_info(self) -> dict:
        return {
            "task_id": self.config.task_id,
            "episode_days": self.config.episode_days,
            "n_stores": len(self.config.stores),
            "n_skus": len(self.config.skus),
            "n_suppliers": len(self.config.suppliers),
            "forecast_horizon": self.config.forecast_horizon,
        }

    @property
    def trajectory(self) -> Optional[Trajectory]:
        return self._trajectory

    @property
    def total_received(self) -> int:
        return self._total_received

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_observation(self) -> Observation:
        s = self._state
        store_ids = [st.store_id for st in self.config.stores]
        sku_ids = [sk.sku_id for sk in self.config.skus]

        store_inv = np.array(
            [
                [s.store_inventory[sid][skuid].units for skuid in sku_ids]
                for sid in store_ids
            ],
            dtype=np.float32,
        )

        dc_inv = np.array(
            [s.dc_inventory.get(skuid, 0) for skuid in sku_ids], dtype=np.float32
        )

        expiry = np.array(
            [
                [
                    s.store_inventory[sid][skuid].days_until_next_expiry(s.day)
                    for skuid in sku_ids
                ]
                for sid in store_ids
            ],
            dtype=np.float32,
        )

        in_transit = np.zeros((len(store_ids), len(sku_ids)), dtype=np.float32)
        for order in s.in_transit:
            si = store_ids.index(order.store_id)
            ski = sku_ids.index(order.sku_id)
            in_transit[si, ski] += order.units

        truck_cap = np.array(
            [self._stores[sid].truck_capacity for sid in store_ids], dtype=np.float32
        )

        supplier_status = np.array(
            [float(v) for v in s.supplier_status.values()], dtype=np.float32
        )

        forecast = self._demand_sim.forecast(
            s.day, self.config.forecast_horizon, s.demand_overrides
        )
        forecast_arr = np.array(
            [
                [
                    [
                        forecast[sid][skuid][t]
                        for t in range(self.config.forecast_horizon)
                    ]
                    for skuid in sku_ids
                ]
                for sid in store_ids
            ],
            dtype=np.float32,
        )

        return {
            "store_inventory": store_inv,  # (n_stores, n_skus)
            "dc_inventory": dc_inv,  # (n_skus,)
            "expiry_countdown": expiry,  # (n_stores, n_skus)
            "in_transit": in_transit,  # (n_stores, n_skus)
            "truck_capacity": truck_cap,  # (n_stores,)
            "supplier_status": supplier_status,  # (n_suppliers,)
            "demand_forecast": forecast_arr,  # (n_stores, n_skus, horizon)
            "current_day": s.day,
            "day_of_week": s.day % 7,
        }
