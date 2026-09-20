from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class SupplyChainExOEnv(gym.Env):
    """
    Single-product warehouse / replenishment environment with stochastic lead times.

    The exact remaining lead time of individual shipments is INTERNAL and is NOT
    part of the agent's observation.

    Observation:
        inventory:
            Current warehouse inventory.
        demand:
            Demand of the current period.
        in_transit:
            Total quantity currently on the way.
        order_history:
            Quantities ordered in previous periods.
            order_history[0] = order placed 1 period ago,
            order_history[1] = order placed 2 periods ago, ...

        shipment_quantities:
            Quantities of shipments that are currently still in transit.
            Fixed-length vector padded with zeros.

        shipment_ages:
            Number of periods each open shipment has already been in transit.
            Aligned with shipment_quantities. The exact remaining lead time
            remains hidden from the agent.

        mean_lead_time:
            Mean lead time of the known lead-time distribution.

    Action:
        Integer order quantity in {0, ..., max_order}.

    Demand:
        Lost-sales setting. Unfulfilled demand disappears after the current period
        and is NOT carried over as backlog.

    Timing of one step / one period:
        1. Current demand is fulfilled from current inventory.
        2. Time advances by one period.
        3. Shipments that were ALREADY in transit are advanced.
        4. Shipments whose hidden lead time expires arrive.
        5. Overflow above capacity is discarded and penalized.
        6. Holding cost is charged on end-of-period inventory.
        7. ONLY NOW, at the end of the period, the agent's order is placed.
        8. Next period's demand is sampled.
        9. The next observation is returned.

    Important lead-time convention:
        A new order is added only AFTER existing shipments were advanced.
        Therefore, an order with lead time 1 placed at the end of period t
        arrives at the end of period t+1 and is available for period t+2 demand.

    The action is selected from the observation at the beginning of the period,
    but is operationally executed only at the end of that period.
    """

    metadata = {
    "render_modes": [
        "human",
        "rgb_array",
    ],
    "render_fps": 30,
    }
    
    def __init__(
        self,
        capacity: int = 100,
        max_order: Optional[int] = None,
        demand_lambda: float = 10.0,
        lead_time_mean: float = 60.0,
        lead_time_std: float = 2,
        max_lead_time: int = 100,
        selling_price: float = 20.0,
        purchase_cost: float = 2.0,
        holding_cost: float = 0.25,
        overflow_cost: float = 8.0,
        lost_sales_cost: float = 10,
        episode_length: int = 730,
        initial_inventory: Optional[int] = None,
        order_history_length: Optional[int] = None,
        render_mode: Optional[str] = None,

    ):
        super().__init__()

        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        if max_lead_time <= 0:
            raise ValueError("max_lead_time must be > 0")
        if episode_length <= 0:
            raise ValueError("episode_length must be > 0")
        if demand_lambda < 0:
            raise ValueError("demand_lambda must be >= 0")
        if lead_time_std < 0:
            raise ValueError("lead_time_std must be >= 0")
        if lead_time_mean <= 0:
            raise ValueError("lead_time_mean must be > 0")

        self.capacity = int(capacity)
        self.max_order = int(max_order if max_order is not None else capacity)
        if self.max_order < 0:
            raise ValueError("max_order must be >= 0")

        self.demand_lambda = float(demand_lambda)

        self.lead_time_mean = float(lead_time_mean)
        self.lead_time_std = float(lead_time_std)
        self.max_lead_time = int(max_lead_time)

        self.selling_price = float(selling_price)
        self.purchase_cost = float(purchase_cost)
        self.holding_cost = float(holding_cost)
        self.overflow_cost = float(overflow_cost)
        self.lost_sales_cost = float(lost_sales_cost)

        self.episode_length = int(episode_length)

        self.initial_inventory = (
            int(initial_inventory)
            if initial_inventory is not None
            else self.capacity // 2
        )
        if not 0 <= self.initial_inventory <= self.capacity:
            raise ValueError(
                "initial_inventory must be between 0 and capacity"
            )

        self.order_history_length = int(
            order_history_length
            if order_history_length is not None
            else self.max_lead_time
        )
        if self.order_history_length <= 0:
            raise ValueError("order_history_length must be > 0")

        self.action_space = spaces.Discrete(self.max_order + 1)

        self.max_visible_shipments = self.max_lead_time

        max_in_transit = float(self.max_order * self.max_lead_time)

        if render_mode not in {
            None,
            "human",
            "rgb_array",
        }:
            raise ValueError(
                f"Unknown render mode: {render_mode}"
            )

        self.render_mode = render_mode
        self.renderer = None

        self.last_info = {}

        self._next_shipment_id = 0

        self.observation_space = spaces.Dict(
            {
                "inventory": spaces.Box(
                    low=0.0,
                    high=float(self.capacity),
                    shape=(1,),
                    dtype=np.float32,
                ),
                "demand": spaces.Box(
                    low=0.0,
                    high=np.inf,
                    shape=(1,),
                    dtype=np.float32,
                ),
                "in_transit": spaces.Box(
                    low=0.0,
                    high=max_in_transit,
                    shape=(1,),
                    dtype=np.float32,
                ),
                "order_history": spaces.Box(
                    low=0.0,
                    high=float(self.max_order),
                    shape=(self.order_history_length,),
                    dtype=np.float32,
                ),
                "shipment_features": spaces.Box(
                    low=np.zeros(
                        (
                            self.max_visible_shipments,
                            2,
                        ),
                        dtype=np.float32,
                    ),
                    high=np.tile(
                        np.array(
                            [
                                self.max_order,
                                self.max_lead_time,
                            ],
                            dtype=np.float32,
                        ),
                        (
                            self.max_visible_shipments,
                            1,
                        ),
                    ),
                    dtype=np.float32,
                ),
                "mean_lead_time": spaces.Box(
                    low=1.0,
                    high=float(self.max_lead_time),
                    shape=(1,),
                    dtype=np.float32,
                ),
            }
        )

        self.inventory = 0
        self.demand = 0
        self.step_count = 0

        self._shipments: list[dict[str, int]] = []

        self.order_history = np.zeros(
            self.order_history_length,
            dtype=np.float32,
        )

    def _get_in_transit(self) -> int:
        return int(
            sum(shipment["qty"] for shipment in self._shipments)
        )

    def _get_open_shipment_features(
        self,
    ) -> np.ndarray:

        features = np.zeros(
            (
                self.max_visible_shipments,
                2,
            ),
            dtype=np.float32,
        )

        open_shipments = sorted(
            self._shipments,
            key=lambda shipment: shipment["placed_step"],
            reverse=True,
        )

        for i, shipment in enumerate(
            open_shipments[
                : self.max_visible_shipments
            ]
        ):

            quantity = float(
                shipment["qty"]
            )

            age = (
                self.step_count
                - int(
                    shipment["placed_step"]
                )
            )

            age = float(
                np.clip(
                    age,
                    0,
                    self.max_lead_time,
                )
            )

            features[i, 0] = quantity
            features[i, 1] = age

        return features

    def _get_obs(self) -> dict[str, np.ndarray]:

        return {
            "inventory": np.array(
                [self.inventory],
                dtype=np.float32,
            ),
            "demand": np.array(
                [self.demand],
                dtype=np.float32,
            ),
            "in_transit": np.array(
                [self._get_in_transit()],
                dtype=np.float32,
            ),
            "order_history": self.order_history.copy(),
            "shipment_features": self._get_open_shipment_features(),
            "mean_lead_time": np.array(
                [self.lead_time_mean],
                dtype=np.float32,
            ),
        }

    def _sample_demand(self) -> int:
        if self.step_count < self.max_lead_time:
            demand = 0
        else:
            demand = self.np_random.poisson(self.demand_lambda)
        return int(demand)

    def _sample_lead_time(self) -> int:
        if self.lead_time_std == 0:
            lead_time = int(round(self.lead_time_mean))
        else:
            lead_time = int(
                round(
                    self.np_random.normal(
                        loc=self.lead_time_mean,
                        scale=self.lead_time_std,
                    )
                )
            )
        #print("lt:", int(np.clip(lead_time, 1, self.max_lead_time,)))

        return int(
            np.clip(
                lead_time,
                1,
                self.max_lead_time,
            )
        )

    def _advance_existing_shipments(self) -> int:

        arriving = 0
        still_in_transit: list[dict[str, int]] = []
        for shipment in self._shipments:
            shipment["remaining"] -= 1

            if shipment["remaining"] <= 0:
                arriving += shipment["qty"]
            else:
                still_in_transit.append(shipment)

        self._shipments = still_in_transit
        return arriving

    def _place_end_of_period_order(
        self,
        order_qty: int,
    ) -> Optional[int]:
        if order_qty <= 0:
            return None

        sampled_lead_time = self._sample_lead_time()

        self._shipments.append(
            {
                "id": self._next_shipment_id,
                "qty": int(order_qty),
                "remaining": sampled_lead_time,
                "initial_lead_time": sampled_lead_time,
                "placed_step": int(self.step_count),
            }
        )

        self._next_shipment_id += 1

        return sampled_lead_time

    def _update_order_history(
        self,
        order_qty: int,
    ) -> None:
        if self.order_history_length > 1:
            self.order_history[1:] = (
                self.order_history[:-1].copy()
            )

        self.order_history[0] = float(order_qty)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ):
        super().reset(seed=seed)

        options = options or {}

        inventory = int(
            options.get(
                "initial_inventory",
                self.initial_inventory,
            )
        )
        if not 0 <= inventory <= self.capacity:
            raise ValueError(
                "initial_inventory must be between 0 and capacity"
            )

        self.inventory = inventory
        self.demand = int(
            options.get(
                "initial_demand",
                self._sample_demand(),
            )
        )
        if self.demand < 0:
            raise ValueError(
                "initial_demand must be >= 0"
            )

        self.step_count = 0
        self._shipments = []
        self._next_shipment_id = 0
        self.last_info = {}

        self.order_history = np.zeros(
            self.order_history_length,
            dtype=np.float32,
        )

        obs = self._get_obs()

        info = {
            "inventory": self.inventory,
            "demand": self.demand,
            "in_transit": 0,
            "shipment_features": obs["shipment_features"].copy(),
        }

        return obs, info

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(
                f"Invalid action {action!r}"
            )

        order_qty = int(action)
        current_demand = int(self.demand)


        sold = min(
            self.inventory,
            current_demand,
        )
        lost_sales = current_demand - sold
        self.inventory -= sold

        arriving = self._advance_existing_shipments()


        free_capacity = self.capacity - self.inventory

        accepted = min(
            arriving,
            free_capacity,
        )
        overflow = max(
            arriving - free_capacity,
            0,
        )

        self.inventory += accepted


        revenue = (
            sold * self.selling_price
        )
        #print("Revenue:", revenue)
        inventory_holding_cost = (
            self.inventory * self.holding_cost
        )
        excess_order_cost = (
            overflow * self.overflow_cost
        )
        lost_sales_penalty = (
            lost_sales * self.lost_sales_cost
        )

        sampled_lead_time = (
            self._place_end_of_period_order(order_qty)
        )

        ordering_cost = (
            order_qty * self.purchase_cost
        )

        # Reward includes the cost of the order placed at period end.
        reward = (
            revenue
            - ordering_cost
            - inventory_holding_cost
            - excess_order_cost
            - lost_sales_penalty
        )
        reward = reward/200
        self.last_reward = float(reward)
        # print("Inventory:", self.inventory)
        # #print("reward:", reward)
        # print("total:",revenue,
        #     "-", ordering_cost,
        #     "-", inventory_holding_cost,
        #     "-", excess_order_cost,
        #     "-", lost_sales_penalty)


        self._update_order_history(order_qty)
        self.demand = self._sample_demand()

        self.step_count += 1

        terminated = False
        truncated = (
            self.step_count >= self.episode_length
        )

        info = {
            "ordered_at_period_end": order_qty,

            "sampled_lead_time": sampled_lead_time,

            "reward": float(reward),

            "current_demand": current_demand,
            "sold": sold,
            "lost_sales": lost_sales,

            "arriving_before_order": arriving,
            "accepted": accepted,
            "overflow": overflow,

            "inventory_end": self.inventory,
            "next_demand": self.demand,
            "in_transit_after_order": self._get_in_transit(),

            "revenue": float(revenue),
            "ordering_cost": float(ordering_cost),
            "holding_cost": float(inventory_holding_cost),
            "overflow_cost": float(excess_order_cost),
            "lost_sales_cost": float(lost_sales_penalty),
        }

        self.last_info = info

        if self.render_mode == "human":
            self.render()

        return (
            self._get_obs(),
            float(reward),
            terminated,
            truncated,
            info,
        )
    
    def render(self):

        if self.render_mode is None:
            return None

        if self.renderer is None:

            try:
                from .supply_chain_renderer import (
                    SupplyChainRenderer
                )
            except ImportError:
                from supply_chain_renderer import (
                    SupplyChainRenderer
                )

            self.renderer = SupplyChainRenderer(
                render_mode=self.render_mode,
                fps=self.metadata["render_fps"],
            )

        return self.renderer.render(
            self,
            animate=self.render_mode == "human",
        )

    def close(self):

        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
