from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class SupplyChainNonS5Env(gym.Env):
    """
    Single-product warehouse / replenishment environment with stochastic lead times.

    The exact remaining lead time of individual shipments is INTERNAL and is NOT
    part of the agent's observation.

    Observation:
        inventory:
            Current warehouse inventory.
        demand:
            Demand of the current period.
        lifecycle_progress:
            Current normalized product lifecycle position in [0, 1].
        days_from_release:
            Signed number of days relative to product release. Negative values
            mean the release is still in the future, 0 is the release day, and
            positive values mean days since release.
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
        Non-stationary lost-sales demand.

        Daily demand follows a Poisson distribution with a time-varying mean:

            D_t ~ Poisson(lambda_t)

        lambda_t follows a one-time product lifecycle with introduction,
        growth, maturity and decline phases. After the decline, demand remains
        permanently at the lower market-saturation level and does not restart.
        Unfulfilled demand disappears after the current period and is NOT carried
        over as backlog.

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
        capacity: int = 1000,
        max_order: Optional[int] = None,

        # Peak mean demand during the maturity phase.
        demand_lambda: float = 100.0,

        # Non-stationary product-life-cycle demand.
        demand_lifecycle_length: int = 730,
        repeat_product_lifecycle = False,
        demand_warmup_periods: Optional[int] = None,
        demand_intro_fraction: float = 0.10,
        demand_growth_fraction: float = 0.30,
        demand_maturity_fraction: float = 0.35,
        demand_decline_fraction: float = 0.25,
        demand_start_ratio: float = 0.00,
        demand_intro_end_ratio: float = 0.40,
        demand_end_ratio: float = 0.15,

        lead_time_mean: float = 60.0,
        lead_time_std: float = 2,
        max_lead_time: int = 100,
        selling_price: float = 20.0,
        purchase_cost: float = 2.0,
        holding_cost: float = 0.25,
        overflow_cost: float = 8.0,
        lost_sales_cost: float = 10,
        episode_length: int = 1100,
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
        if demand_lifecycle_length <= 1:
            raise ValueError("demand_lifecycle_length must be > 1")

        lifecycle_fractions = (
            float(demand_intro_fraction),
            float(demand_growth_fraction),
            float(demand_maturity_fraction),
            float(demand_decline_fraction),
        )
        if any(value < 0.0 for value in lifecycle_fractions):
            raise ValueError("demand lifecycle fractions must be >= 0")
        if not np.isclose(sum(lifecycle_fractions), 1.0):
            raise ValueError(
                "demand_intro_fraction + demand_growth_fraction + "
                "demand_maturity_fraction + demand_decline_fraction "
                "must sum to 1.0"
            )

        for name, value in (
            ("demand_start_ratio", demand_start_ratio),
            ("demand_intro_end_ratio", demand_intro_end_ratio),
            ("demand_end_ratio", demand_end_ratio),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0")

        if demand_intro_end_ratio < demand_start_ratio:
            raise ValueError(
                "demand_intro_end_ratio must be >= demand_start_ratio"
            )

        if lead_time_std < 0:
            raise ValueError("lead_time_std must be >= 0")
        if lead_time_mean <= 0:
            raise ValueError("lead_time_mean must be > 0")

        self.capacity = int(capacity)
        self.max_order = int(max_order if max_order is not None else capacity)
        if self.max_order < 0:
            raise ValueError("max_order must be >= 0")

        # demand_lambda is interpreted as peak mean demand.
        self.demand_lambda = float(demand_lambda)

        self.demand_lifecycle_length = int(demand_lifecycle_length)
        self.repeat_product_lifecycle = repeat_product_lifecycle

        self.demand_warmup_periods = int(
            demand_warmup_periods
            if demand_warmup_periods is not None
            else max_lead_time
        )
        if self.demand_warmup_periods < 0:
            raise ValueError("demand_warmup_periods must be >= 0")

        self.demand_intro_fraction = float(demand_intro_fraction)
        self.demand_growth_fraction = float(demand_growth_fraction)
        self.demand_maturity_fraction = float(demand_maturity_fraction)
        self.demand_decline_fraction = float(demand_decline_fraction)

        self.demand_start_ratio = float(demand_start_ratio)
        self.demand_intro_end_ratio = float(demand_intro_end_ratio)
        self.demand_end_ratio = float(demand_end_ratio)

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

        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )
        #self.action_space = spaces.Discrete(self.max_order + 1)

        self.max_visible_shipments = self.max_lead_time

        self.max_in_transit = float(
            self.max_order * self.max_lead_time
        )


        self.days_from_release_scale = float(
            max(
                1,
                self.demand_warmup_periods,
                self.episode_length - self.demand_warmup_periods,
            )
        )

        self.demand_obs_cap = float(
            max(
                1.0,
                self.demand_lambda
                + 6.0 * np.sqrt(self.demand_lambda + 1.0),
            )
        )

        L = int(round(self.lead_time_mean))

        self.demand_forecast_horizons = np.array(
            [
                0,
                max(1, L // 2),
                L + 1,
                L + 15,
                L + 30,
            ],
            dtype=np.int32,
        )

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
                    high=1.0,
                    shape=(1,),
                    dtype=np.float32,
                ),

                "demand": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(1,),
                    dtype=np.float32,
                ),

                # "days_from_release": spaces.Box(
                #     low=-1.0,
                #     high=1.0,
                #     shape=(1,),
                #     dtype=np.float32,
                # ),

                "days_from_release": spaces.Box(
                    low=float(-self.demand_warmup_periods),
                    high=float(
                        self.episode_length - self.demand_warmup_periods
                    ),
                    shape=(1,),
                    dtype=np.float32,
                ),

                "lifecycle_progress": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(1,),
                    dtype=np.float32,
                ),

                "in_transit": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(1,),
                    dtype=np.float32,
                ),

                "order_history": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.order_history_length,),
                    dtype=np.float32,
                ),

                "shipment_features": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(
                        self.max_visible_shipments,
                        2,
                    ),
                    dtype=np.float32,
                ),

                "demand_forecast": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(
                        len(self.demand_forecast_horizons),
                    ),
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

        quantity_scale = max(
            float(self.max_order),
            1.0,
        )

        age_scale = max(
            float(self.max_lead_time),
            1.0,
        )

        for i, shipment in enumerate(
            open_shipments[:self.max_visible_shipments]
        ):

            quantity = float(
                shipment["qty"]
            )

            age = (
                self.step_count
                - int(shipment["placed_step"])
            )

            age = float(
                np.clip(
                    age,
                    0,
                    self.max_lead_time,
                )
            )

            features[i, 0] = (
                quantity / quantity_scale
            )

            features[i, 1] = (
                age / age_scale
            )

        return np.clip(
            features,
            0.0,
            1.0,
        )

    def _get_demand_forecast_features(self) -> np.ndarray:
        """
        Normalized expected demand at several horizons relative
        to the mean replenishment lead time.

        All values are in [0, 1], where 1 corresponds to the
        peak expected demand self.demand_lambda.
        """

        demand_scale = max(
            self.demand_lambda,
            1e-8,
        )

        forecast = [
            self._get_future_demand_lambda(int(horizon))
            / demand_scale
            for horizon in self.demand_forecast_horizons
        ]

        return np.clip(
            np.asarray(
                forecast,
                dtype=np.float32,
            ),
            0.0,
            1.0,
        )
    
    def _get_obs(self) -> dict[str, np.ndarray]:

        inventory_normalized = (
            self.inventory
            / max(float(self.capacity), 1.0)
        )

        demand_normalized = (
            self.demand
            / self.demand_obs_cap
        )

        demand_normalized = np.clip(
            demand_normalized,
            0.0,
            1.0,
        )

        days_from_release_normalized = (
            self._get_days_from_release()
            / self.days_from_release_scale
        )

        days_from_release_normalized = np.clip(
            days_from_release_normalized,
            -1.0,
            1.0,
        )

        if self.max_in_transit > 0.0:
            in_transit_normalized = (
                self._get_in_transit()
                / self.max_in_transit
            )
        else:
            in_transit_normalized = 0.0

        order_scale = max(
            float(self.max_order),
            1.0,
        )

        order_history_normalized = (
            self.order_history
            / order_scale
        )

        return {
            "inventory": np.array(
                [inventory_normalized],
                dtype=np.float32,
            ),

            "demand": np.array(
                [demand_normalized],
                dtype=np.float32,
            ),

            # "days_from_release": np.array(
            #     [days_from_release_normalized],
            #     dtype=np.float32,
            # ),
            "days_from_release": np.array(
                [self._get_days_from_release()],
                dtype=np.float32,
            ),

            "lifecycle_progress": np.array(
                [self._get_lifecycle_progress()],
                dtype=np.float32,
            ),

            "in_transit": np.array(
                [in_transit_normalized],
                dtype=np.float32,
            ),

            "order_history": np.clip(
                order_history_normalized,
                0.0,
                1.0,
            ).astype(np.float32),

            "shipment_features":
                self._get_open_shipment_features(),

            "demand_forecast":
                self._get_demand_forecast_features(),
        }
    
    @staticmethod
    def _smoothstep(value: float) -> float:
        x = float(np.clip(value, 0.0, 1.0))
        return x * x * (3.0 - 2.0 * x)

    def _get_days_from_release(self) -> int:
        """
        Signed time relative to product release.

        Examples for demand_warmup_periods=100:
            step_count =   0 -> -100
            step_count =  99 ->   -1
            step_count = 100 ->    0
            step_count = 101 ->    1
        """
        return int(self.step_count) - self.demand_warmup_periods

    def _get_product_age(self) -> int:
        return max(
            0,
            self._get_days_from_release(),
        )
    
    def _get_product_age_at_step(self, step: int) -> int:
        return max(
            0,
            self._get_days_from_release_at_step(step),
        )

    
    def _get_days_from_release_at_step(self, step: int) -> int:
        return step - self.demand_warmup_periods

    def _get_lifecycle_progress(self) -> float:
        if self.step_count < self.demand_warmup_periods:
            return 0.0

        age = self._get_product_age()
        length = self.demand_lifecycle_length

        if self.repeat_product_lifecycle:
            cycle_age = age % length
            return float(
                cycle_age / max(1, length - 1)
            )

        return float(
            np.clip(
                age / max(1, length - 1),
                0.0,
                1.0,
            )
        )

    def _get_lifecycle_progress_at_step(self, step: int) -> float:
        if step < self.demand_warmup_periods:
            return 0.0

        age = self._get_product_age_at_step(step)
        length = self.demand_lifecycle_length

        if self.repeat_product_lifecycle:
            cycle_age = age % length
            return float(
                cycle_age / max(1, length - 1)
            )

        return float(
            np.clip(
                age / max(1, length - 1),
                0.0,
                1.0,
            )
        )

    def _get_lifecycle_phase(self) -> str:
        if self.step_count < self.demand_warmup_periods:
            return "warmup"

        age = self._get_product_age()
        progress = self._get_lifecycle_progress()

        intro_end = self.demand_intro_fraction
        growth_end = intro_end + self.demand_growth_fraction
        maturity_end = growth_end + self.demand_maturity_fraction

        if progress < intro_end:
            return "introduction"
        if progress < growth_end:
            return "growth"
        if progress < maturity_end:
            return "maturity"

        if age >= self.demand_lifecycle_length:
            return "saturation"

        return "decline"

    def _get_current_demand_lambda(self) -> float:
        if self.step_count < self.demand_warmup_periods:
            return 0.0

        peak = self.demand_lambda
        start = peak * self.demand_start_ratio
        intro_target = peak * self.demand_intro_end_ratio
        end = peak * self.demand_end_ratio

        progress = self._get_lifecycle_progress()

        intro_end = self.demand_intro_fraction
        growth_end = intro_end + self.demand_growth_fraction
        maturity_end = growth_end + self.demand_maturity_fraction

        if progress < intro_end:
            local = (
                progress / intro_end
                if intro_end > 0.0
                else 1.0
            )
            weight = self._smoothstep(local)
            return float(
                start
                + weight * (intro_target - start)
            )

        if progress < growth_end:
            width = self.demand_growth_fraction
            local = (
                (progress - intro_end) / width
                if width > 0.0
                else 1.0
            )
            weight = self._smoothstep(local)
            return float(
                intro_target
                + weight * (peak - intro_target)
            )

        if progress < maturity_end:
            return float(peak)

        width = self.demand_decline_fraction
        local = (
            (progress - maturity_end) / width
            if width > 0.0
            else 1.0
        )
        weight = self._smoothstep(local)

        return float(
            peak + weight * (end - peak)
        )

    def _sample_demand(self) -> int:
        demand_mean = self._get_current_demand_lambda()

        if demand_mean <= 0.0:
            return 0

        return int(
            self.np_random.poisson(
                demand_mean
            )
        )


    def _get_future_demand_lambda(self, days_ahead: int) -> float:
        """
        Return the expected demand (Poisson lambda) `days_ahead`
        days from the current step.

        days_ahead = 0  -> current expected demand
        days_ahead = 1  -> expected demand tomorrow
        days_ahead = 60 -> expected demand in 60 days
        """

        if days_ahead < 0:
            raise ValueError("days_ahead must be >= 0")

        future_step = self.step_count + days_ahead

        if future_step < self.demand_warmup_periods:
            return 0.0

        peak = self.demand_lambda
        start = peak * self.demand_start_ratio
        intro_target = peak * self.demand_intro_end_ratio
        end = peak * self.demand_end_ratio

        progress = self._get_lifecycle_progress_at_step(future_step)

        intro_end = self.demand_intro_fraction
        growth_end = intro_end + self.demand_growth_fraction
        maturity_end = growth_end + self.demand_maturity_fraction

        if progress < intro_end:
            local = (
                progress / intro_end
                if intro_end > 0.0
                else 1.0
            )

            weight = self._smoothstep(local)

            return float(
                start
                + weight * (intro_target - start)
            )

        if progress < growth_end:
            width = self.demand_growth_fraction

            local = (
                (progress - intro_end) / width
                if width > 0.0
                else 1.0
            )

            weight = self._smoothstep(local)

            return float(
                intro_target
                + weight * (peak - intro_target)
            )

        if progress < maturity_end:
            return float(peak)

        width = self.demand_decline_fraction

        local = (
            (progress - maturity_end) / width
            if width > 0.0
            else 1.0
        )

        local = np.clip(local, 0.0, 1.0)

        weight = self._smoothstep(local)

        return float(
            peak
            + weight * (end - peak)
        )

    def _sample_lead_time(self) -> int:
        """
        Sample a hidden integer lead time in [1, max_lead_time].
        """
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
        """
        Place the action/order at the end of the current period.
        """
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

        self.step_count = 0
        self._shipments = []
        self._next_shipment_id = 0
        self.last_info = {}

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

        self.order_history = np.zeros(
            self.order_history_length,
            dtype=np.float32,
        )

        obs = self._get_obs()

        info = {
            "inventory": self.inventory,
            "demand": self.demand,
            "demand_lambda_current": self._get_current_demand_lambda(),
            "lifecycle_progress": self._get_lifecycle_progress(),
            "days_from_release": self._get_days_from_release(),
            "lifecycle_phase": self._get_lifecycle_phase(),
            "in_transit": 0,
            #"shipment_features": obs["shipment_features"].copy(),
        }

        return obs, info

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(
                f"Invalid action {action!r}"
            )

        order_qty = int(
            np.clip(action[0], 0.0, 1.0)
            * self.max_order
        )
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

        reward = (
            revenue
            - ordering_cost
            - inventory_holding_cost
            - excess_order_cost
            - lost_sales_penalty
        )
        reward = reward/2000
        #print("R:", reward)
        self.last_reward = float(reward)
        # print("Inventory:", self.inventory)
        # print("reward:", reward)
        # print("total:",revenue,
        #     "-", ordering_cost,
        #     "-", inventory_holding_cost,
        #     "-", excess_order_cost,
        #     "-", lost_sales_penalty)


        self._update_order_history(order_qty)

        self.step_count += 1
        self.demand = self._sample_demand()

        terminated = False
        truncated = (
            self.step_count >= self.episode_length
        )

        info = {
            "ordered_at_period_end": order_qty,

            "sampled_lead_time": sampled_lead_time,

            "reward": float(reward),

            "current_demand": current_demand,
            "next_demand": self.demand,
            "demand_lambda_next": self._get_current_demand_lambda(),
            "lifecycle_progress": self._get_lifecycle_progress(),
            "days_from_release": self._get_days_from_release(),
            "lifecycle_phase": self._get_lifecycle_phase(),

            "sold": sold,
            "lost_sales": lost_sales,

            "arriving_before_order": arriving,
            "accepted": accepted,
            "overflow": overflow,

            "inventory_end": self.inventory,
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
                from .supply_chain_renderer_nons import (
                    SupplyChainRendererNonS
                )
            except ImportError:
                from RLII_26.rl_zoo3.custom_envs.supply_chain_renderer_nons import (
                    SupplyChainRendererNonS
                )

            self.renderer = SupplyChainRendererNonS(
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

