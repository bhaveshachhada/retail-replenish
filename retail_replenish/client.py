# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Retail Replenish Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import (
    RetailReplenishAction,
    RetailReplenishObservation,
    RetailReplenishState,
)


class RetailReplenishEnv(
    EnvClient[RetailReplenishAction, RetailReplenishObservation, State]
):
    """
    Client for the Retail Replenish Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with RetailReplenishEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.echoed_message)
        ...
        ...     result = client.step(RetailReplenishAction(message="Hello!"))
        ...     print(result.observation.echoed_message)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = RetailReplenishEnv.from_docker_image("retail_replenish-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(RetailReplenishAction(message="Test"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: RetailReplenishAction) -> Dict:
        """
        Convert RetailReplenishAction to JSON payload for step message.

        Args:
            action: RetailReplenishAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "replenish": action.replenish,
            "emergency_order": action.emergency_order,
            "dc_order": action.dc_order,
            "inter_store_transfer": action.inter_store_transfer,
        }

    def _parse_result(self, payload: Dict) -> StepResult[RetailReplenishObservation]:
        """
        Parse server response into StepResult[RetailReplenishObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with RetailReplenishObservation
        """
        obs_data = payload.get("observation", {})
        observation = RetailReplenishObservation(
            store_inventory=obs_data.get("store_inventory", []),
            dc_inventory=obs_data.get("dc_inventory", []),
            expiry_countdown=obs_data.get("expiry_countdown", []),
            in_transit=obs_data.get("in_transit", []),
            supplier_status=obs_data.get("supplier_status", []),
            truck_capacity=obs_data.get("truck_capacity", []),
            demand_forecast=obs_data.get("demand_forecast", []),
            current_day=obs_data.get("current_day"),
            day_of_week=obs_data.get("day_of_week"),
            reward_breakdown=obs_data.get("reward_breakdown", None),
            done=obs_data.get("done", False),
            reward=obs_data.get("reward", 0),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=observation.reward,
            done=observation.done,
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return RetailReplenishState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            day=payload.get("day"),
            store_inventory=payload.get("store_inventory", {}),
            dc_inventory=payload.get("dc_inventory", {}),
            in_transit=payload.get("in_transit", []),
            pending_dc_orders=payload.get("pending_dc_orders", []),
            supplier_status=payload.get("supplier_status", {}),
            demand_overrides=payload.get("demand_overrides", {}),
        )
