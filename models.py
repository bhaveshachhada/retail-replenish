from dataclasses import dataclass
from typing import Dict, List

from openenv.core.env_server import Action, Observation


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
