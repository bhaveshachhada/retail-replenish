# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Retail Replenish Environment Implementation.

A simple test environment that echoes back messages sent to it.
Perfect for testing HTTP server infrastructure.
"""

import uuid
from typing import Optional, Dict, Any, List
from uuid import uuid4

import numpy as np
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from retail_replenish import RetailReplenishAction, RetailReplenishObservation
from retail_replenish.models import (
    RetailReplenishState,
    StoreInventory,
    SKU,
    Store,
    Supplier,
    ExpiryBatch,
)
from retail_replenish.server.dynamics import DemandSimulator, TransitionEngine
from retail_replenish.server.reward import RewardWeights, RewardFunction
from retail_replenish.server.state import Trajectory, StepRecord
from retail_replenish.server.tasks import make_task3_config


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


class RetailReplenishEnvironment(
    Environment[RetailReplenishAction, RetailReplenishObservation, RetailReplenishState]
):
    """
    A simple echo environment that echoes back messages.

    This environment is designed for testing the HTTP server infrastructure.
    It maintains minimal state and simply echoes back whatever message it receives.

    Example:
        >>> env = RetailReplenishEnvironment()
        >>> obs = env.reset()
        >>> print(obs.echoed_message)  # "Retail Replenish environment ready!"
        >>>
        >>> obs = env.step(RetailReplenishAction(message="Hello"))
        >>> print(obs.echoed_message)  # "Hello"
        >>> print(obs.message_length)  # 5
    """

    # Enable concurrent WebSocket sessions.
    # Set to True if your environment isolates state between instances.
    # When True, multiple WebSocket clients can connect simultaneously, each
    # getting their own environment instance (when using factory mode in app.py).
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the retail_replenish environment."""
        super().__init__()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0

        self.config = make_task3_config()

        self._skus = {s.sku_id: s for s in self.config.skus}
        self._stores = {s.store_id: s for s in self.config.stores}

        self._state: Optional[RetailReplenishState] = None
        self._trajectory: Optional[Trajectory] = None
        self._rng: Optional[np.random.Generator] = None
        self._total_received: int = (
            0  # tracks units received at stores (for waste rate)
        )

        self._demand_sim: Optional[DemandSimulator] = None
        self._transition: TransitionEngine = TransitionEngine(
            self.config.skus, self.config.stores
        )
        self._reward_fn: RewardFunction = RewardFunction(
            self.config.skus, self.config.stores, self.config.reward_weights
        )

    def reset(
        self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs
    ) -> RetailReplenishObservation:
        """
        Reset the environment.

        Returns:
            RetailReplenishObservation with a ready message
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1

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
                        ExpiryBatch(
                            units=init_units, expires_on_day=sku.shelf_life_days
                        )
                    )
                store_inv[store.store_id][sku.sku_id] = si
                self._total_received += init_units

        self._state = RetailReplenishState(
            day=0,
            store_inventory=store_inv,
            dc_inventory=dict(self.config.initial_dc_inventory),
            in_transit=[],
            pending_dc_orders=[],
            supplier_status={
                s.supplier_id: s.is_operational for s in self.config.suppliers
            },
            demand_overrides=self.config.demand_overrides,
            episode_id=str(uuid.uuid4()),
            step_count=0,
        )

        self._trajectory = Trajectory(task_id=self.config.task_id)
        obs = self._build_observation(reward=0)
        return obs

    def step(
        self,
        action: RetailReplenishAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> RetailReplenishObservation:
        """
        Execute a step in the environment by echoing the message.

        Args:
            action: RetailReplenishAction containing the message to echo
            timeout_s: Timeout in seconds to wait for a response
            **kwargs: Additional arguments to pass to the action
        Returns:
            RetailReplenishObservation with the echoed message and its length
        """
        self._state.step_count += 1

        assert self._state is not None, "Call reset() before step()."

        self._state.step_count += 1
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
        obs = self._build_observation(
            reward=reward, reward_breakdown=breakdown, terminated=terminated
        )

        return obs

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_observation(
        self,
        reward: float,
        reward_breakdown: Optional[Dict[str, Any]] = None,
        terminated: bool = False,
    ) -> RetailReplenishObservation:
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

        return RetailReplenishObservation(
            **{
                "store_inventory": store_inv,  # (n_stores, n_skus)
                "dc_inventory": dc_inv,  # (n_skus,)
                "expiry_countdown": expiry,  # (n_stores, n_skus)
                "in_transit": in_transit,  # (n_stores, n_skus)
                "truck_capacity": truck_cap,  # (n_stores,)
                "supplier_status": supplier_status,  # (n_suppliers,)
                "demand_forecast": forecast_arr,  # (n_stores, n_skus, horizon)
                "current_day": s.day,
                "day_of_week": s.day % 7,
                "done": terminated,
                "reward": reward,
                "reward_breakdown": reward_breakdown,
            }
        )
