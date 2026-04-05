from typing import Dict, Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from models import (
    RetailReplenishAction,
    RetailReplenishObservation,
    RetailReplenishState,
)


class RetailReplenishClient(
    EnvClient[RetailReplenishAction, RetailReplenishObservation, RetailReplenishState]
):
    def _step_payload(self, action: RetailReplenishAction) -> Dict[str, Any]:
        return {
            "replenish": action.replenish,
            "emergency_order": action.emergency_order,
            "dc_order": action.dc_order,
            "inter_store_transfer": action.inter_store_transfer,
        }

    def _parse_result(
        self, payload: Dict[str, Any]
    ) -> StepResult[RetailReplenishObservation]:
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
        )
        return StepResult(
            observation=observation,
            reward=observation.reward,
            done=observation.done,
        )

    def _parse_state(self, payload: Dict[str, Any]) -> RetailReplenishState:
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
