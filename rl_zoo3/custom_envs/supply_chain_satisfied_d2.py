from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np


_FLOAT32 = np.dtype(np.float32)
_SERVICE_PHASES = ("introduction", "growth", "maturity", "decline", "saturation")
_ACTION_MODES = ("base_stock", "relative_order", "order_qty")
_REWARD_MODES = ("rolling", "window")
_REWARD_SHAPES = ("linear", "quadratic")


class SupplyChainSatisfiedD2Env(gym.Env):
    """Single-product replenishment with non-stationary lost-sales demand.

    Orders placed at the end of period t with lead time L arrive at the end
    of period t+L and can meet demand starting in period t+L+1.

    Reward (per day):
        reward = service_reward - inventory_penalty - overflow_penalty

        reward_mode="rolling": service_reward = f(fill rate of the last
            service_rolling_days days), paid every day once the window
            holds at least service_min_rolling_demand units of demand.
        reward_mode="window":  service_reward = f(fill rate of the fixed
            8-day window), paid on window close (or as daily increments if
            dense_service_reward=True).

        f has its peak at service_target (default 0.98); falling short is
        penalised with service_under_weight, overshooting with
        service_over_weight.

    Fill rate = units sold / units demanded (unit fill rate).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        capacity: int = 1000,
        max_order: Optional[int] = None,
        demand_lambda: float = 100.0,
        demand_lifecycle_length: int = 730,
        repeat_product_lifecycle: bool = False,
        demand_warmup_periods: Optional[int] = None,
        demand_intro_fraction: float = 0.10,
        demand_growth_fraction: float = 0.30,
        demand_maturity_fraction: float = 0.35,
        demand_decline_fraction: float = 0.25,
        demand_start_ratio: float = 0.00,
        demand_intro_end_ratio: float = 0.40,
        demand_end_ratio: float = 0.15,
        lead_time_mean: float = 20.0,
        lead_time_std: float = 2,
        max_lead_time: int = 100,
        # Finanzgrößen: nur im info-Dict, nicht im Reward.
        selling_price: float = 20.0,
        purchase_cost: float = 2.0,
        holding_cost: float = 0.25,
        overflow_cost: float = 8.0,
        lost_sales_cost: float = 10.0,
        episode_length: int = 1100,
        initial_inventory: Optional[int] = None,
        order_history_length: Optional[int] = None,
        render_mode: Optional[str] = None,
        # --- Aktion ------------------------------------------------------
        # Alle Modi außer "order_qty" sind zentriert: a = 0 ist eine
        # vernünftige Politik, die Grenzen a = -1 / +1 sind weit weg vom Optimum.
        # "base_stock":     a in [-1, 1] -> Bestellniveau k * Lieferzeit-Nachfrage,
        #                   Bestellung = Niveau - Inventory Position
        # "relative_order": a in [-1, 1] -> Bestellmenge k * erwartete Tagesnachfrage
        #                   am Ankunftstag (ohne Rückkopplung über das Lager)
        # "order_qty":      a in [0, 1]  -> Bestellmenge a * max_order (alt, nur
        #                   zum Vergleich; Optimum liegt direkt am Rand a = 0)
        action_mode: str = "base_stock",
        base_stock_spread: float = 1.0,      # k = 1 + spread * a, also k in [0, 2]
        # --- Service-Ziel ------------------------------------------------
        service_target: float = 0.98,
        service_reward_scale: float = 1.0,
        # Überschreiten bewusst teurer: siehe "MESSUNG MIT FESTER POLITIK" oben.
        service_under_weight: float = 25.0,  # Strafe pro Fill-Rate-Einheit unter Ziel
        service_over_weight: float = 75.0,   # Strafe pro Fill-Rate-Einheit über Ziel
        service_reward_shape: str = "linear",
        service_reward_floor: float = -3.0,  # Untergrenze für f, begrenzt Ausreißer
        # --- Reward-Modus ------------------------------------------------
        reward_mode: str = "rolling",
        service_rolling_days: int = 14,
        service_min_rolling_demand: float = 20.0,
        service_window_days: int = 8,        # feste Fenster (Reporting / "window")
        dense_service_reward: bool = False,  # nur für reward_mode="window"
        # --- Tägliche Strafen (bezogen auf volle Kapazität) --------------
        inventory_penalty_weight: float = 0.1,
        overflow_penalty_weight: float = 1.0,
        # --- Observation -------------------------------------------------
        include_shipment_details: bool = True,
    ):
        super().__init__()

        # ---------------- Validierung ----------------
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
            raise ValueError("demand lifecycle fractions must sum to 1.0")
        for name, value in (
            ("demand_start_ratio", demand_start_ratio),
            ("demand_intro_end_ratio", demand_intro_end_ratio),
            ("demand_end_ratio", demand_end_ratio),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0")
        if demand_intro_end_ratio < demand_start_ratio:
            raise ValueError("demand_intro_end_ratio must be >= demand_start_ratio")
        if lead_time_std < 0 or lead_time_mean <= 0:
            raise ValueError("lead_time_std must be >= 0 and lead_time_mean > 0")
        if action_mode not in _ACTION_MODES:
            raise ValueError(f"action_mode must be one of {_ACTION_MODES}")
        if base_stock_spread <= 0.0:
            raise ValueError("base_stock_spread must be > 0")
        if not 0.0 < service_target <= 1.0:
            raise ValueError("service_target must be in (0, 1]")
        if service_reward_scale <= 0.0:
            raise ValueError("service_reward_scale must be > 0")
        if service_under_weight < 0.0 or service_over_weight < 0.0:
            raise ValueError("service weights must be >= 0")
        if service_reward_shape not in _REWARD_SHAPES:
            raise ValueError(f"service_reward_shape must be one of {_REWARD_SHAPES}")
        if reward_mode not in _REWARD_MODES:
            raise ValueError(f"reward_mode must be one of {_REWARD_MODES}")
        for name, value in (
            ("service_window_days", service_window_days),
            ("service_rolling_days", service_rolling_days),
        ):
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")
        if service_min_rolling_demand < 0.0:
            raise ValueError("service_min_rolling_demand must be >= 0")
        if inventory_penalty_weight < 0.0 or overflow_penalty_weight < 0.0:
            raise ValueError("penalty weights must be >= 0")

        # ---------------- Parameter ----------------
        self.capacity = int(capacity)
        self.max_order = int(max_order if max_order is not None else capacity)
        if self.max_order < 0:
            raise ValueError("max_order must be >= 0")
        self.demand_lambda = float(demand_lambda)
        self.demand_lifecycle_length = int(demand_lifecycle_length)
        self.repeat_product_lifecycle = bool(repeat_product_lifecycle)
        self.demand_warmup_periods = int(
            demand_warmup_periods if demand_warmup_periods is not None else max_lead_time
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
        self.initial_inventory = int(
            initial_inventory if initial_inventory is not None else self.capacity // 2
        )
        if not 0 <= self.initial_inventory <= self.capacity:
            raise ValueError("initial_inventory must be between 0 and capacity")
        self.order_history_length = int(
            order_history_length if order_history_length is not None else max_lead_time
        )
        if self.order_history_length <= 0:
            raise ValueError("order_history_length must be > 0")

        self.action_mode = action_mode
        self.base_stock_spread = float(base_stock_spread)

        self.service_target = float(service_target)
        self.service_reward_scale = float(service_reward_scale)
        self.service_under_weight = float(service_under_weight)
        self.service_over_weight = float(service_over_weight)
        self.service_reward_shape = service_reward_shape
        self.service_reward_floor = float(service_reward_floor)

        self.reward_mode = reward_mode
        self.service_rolling_days = int(service_rolling_days)
        self.service_min_rolling_demand = float(service_min_rolling_demand)
        self.service_window_days = int(service_window_days)
        self.dense_service_reward = bool(dense_service_reward)

        self.inventory_penalty_weight = float(inventory_penalty_weight)
        self.overflow_penalty_weight = float(overflow_penalty_weight)
        self.include_shipment_details = bool(include_shipment_details)

        if render_mode not in {None, "human", "rgb_array"}:
            raise ValueError(f"Unknown render mode: {render_mode}")
        self.render_mode = render_mode
        self.renderer = None
        self.last_info: dict[str, Any] = {}
        self.last_reward = 0.0

        # ---------------- Aktionsraum ----------------
        if self.action_mode == "order_qty":
            self._action_low, self._action_high = 0.0, 1.0
        else:
            self._action_low, self._action_high = -1.0, 1.0
        self.action_space = spaces.Box(
            self._action_low, self._action_high, shape=(1,), dtype=np.float32
        )

        # ---------------- Skalen ----------------
        self.max_visible_shipments = self.max_lead_time
        self.max_in_transit = float(self.max_order * self.max_lead_time)
        self.days_from_release_scale = float(max(
            1, self.demand_warmup_periods,
            self.episode_length - self.demand_warmup_periods,
        ))
        self.demand_obs_cap = float(max(
            1.0, self.demand_lambda + 6.0 * np.sqrt(self.demand_lambda + 1.0)
        ))
        L = int(round(self.lead_time_mean))
        self._protection_days = L + 1          # Lieferzeit + 1 Tag Review
        self.demand_forecast_horizons = np.array(
            [0, max(1, L // 2), L + 1, L + 15, L + 30], dtype=np.int32
        )
        # Lieferzeit-Nachfrage bei Spitzenlast; Inventory Position bis 2x davon.
        self._lt_demand_scale = max(1.0, self.demand_lambda * self._protection_days)
        self._ip_scale = 2.0 * self._lt_demand_scale

        # ---------------- Observation ----------------
        obs_spaces: dict[str, spaces.Space] = {
            "inventory": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            "demand": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            "days_from_release": spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
            "lifecycle_progress": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            "in_transit": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            # NEU: Lager + unterwegs, die zentrale Größe für Nachbestellung.
            "inventory_position": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            # NEU: erwartete Nachfrage der nächsten L+1 Tage (Bezug für Base-Stock).
            "lead_time_demand": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            "demand_forecast": spaces.Box(
                0.0, 1.0, shape=(len(self.demand_forecast_horizons),), dtype=np.float32
            ),
            # [laufende Fill Rate, Fortschritt, Nachfrage] des festen Fensters
            "service_period_status": spaces.Box(0.0, 1.0, shape=(3,), dtype=np.float32),
            # NEU: [Fill Rate, Nachfrage] des gleitenden Fensters
            "rolling_service": spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float32),
        }
        if self.include_shipment_details:
            obs_spaces["order_history"] = spaces.Box(
                0.0, 1.0, shape=(self.order_history_length,), dtype=np.float32
            )
            obs_spaces["shipment_features"] = spaces.Box(
                0.0, 1.0, shape=(self.max_visible_shipments, 2), dtype=np.float32
            )
        self.observation_space = spaces.Dict(obs_spaces)

        # ---------------- Zustand ----------------
        self.inventory = 0
        self.demand = 0
        self.step_count = 0
        self._next_shipment_id = 0
        self._shipments: list[dict[str, int]] = []
        self._in_transit_total = 0
        self.order_history = np.zeros(self.order_history_length, dtype=np.float32)
        self._capacity_scale = max(float(self.capacity), 1.0)
        self._order_scale = max(float(self.max_order), 1.0)
        self._quantity_scale = max(float(self.max_order), 1.0)
        self._age_scale = max(float(self.max_lead_time), 1.0)
        self._max_lead_time_float = float(self.max_lead_time)
        self._demand_forecast_scale = max(self.demand_lambda, 1e-8)

        # Gleitendes Fenster als Ringpuffer
        self._roll_sold = np.zeros(self.service_rolling_days, dtype=np.int64)
        self._roll_demand = np.zeros(self.service_rolling_days, dtype=np.int64)
        self._roll_pos = 0

        # Feste Fenster
        self._service_windows: list[dict[str, Any]] = []
        self._service_window_id_by_step = np.full(self.episode_length, -1, dtype=np.int32)
        self._service_window_demand = 0
        self._service_window_sold = 0
        self._service_windows_closed = 0
        self._service_windows_scored = 0
        self._service_windows_meeting_target = 0
        self._episode_total_demand = 0
        self._episode_total_sold = 0
        self._build_lookup_tables()

    # ==================================================================
    # Reward-Funktion
    # ==================================================================
    def service_reward_function(self, x: float) -> float:
        """Spitze bei service_target.

        Standardwerte (linear, unter 25 / über 75, Ziel 0.98):
            x = 0.90 -> -1.00     x = 0.98 ->  1.00
            x = 0.95 ->  0.25     x = 0.99 ->  0.25
            x = 0.97 ->  0.75     x = 1.00 -> -0.50
        Untergrenze: service_reward_floor.

        Warum Überschreiten teurer ist: Die Fill Rate ist bei 1.0 gedeckelt,
        an den meisten Tagen liegt sie genau dort, Abweichungen sind seltene,
        tiefe Einbrüche nach unten. Bei gleichen Gewichten liegt das Optimum
        deshalb bei ~100 %. Erst ein ~3x höheres over_weight verschiebt es
        auf ~98 % (gemessen, siehe Modul-Docstring).
        """
        x = float(np.clip(x, 0.0, 1.0))
        d = x - self.service_target
        w = self.service_over_weight if d > 0 else self.service_under_weight
        if self.service_reward_shape == "quadratic":
            penalty = w * d * d
        else:
            penalty = w * abs(d)
        value = self.service_reward_scale * (1.0 - penalty)
        return max(value, self.service_reward_floor)

    # ==================================================================
    # Lookup-Tabellen und feste Service-Fenster
    # ==================================================================
    def _build_lookup_tables(self) -> None:
        horizons = [int(h) for h in self.demand_forecast_horizons]
        self._forecast_horizons_int = tuple(horizons)
        n_steps = self.episode_length + 1
        n_lambda = n_steps + max(max(horizons), self._protection_days) + 1
        self._lambda_table = [
            self._compute_demand_lambda_at_step(s) for s in range(n_lambda)
        ]
        self._progress_table = [
            self._get_lifecycle_progress_at_step(s) for s in range(n_steps)
        ]
        self._phase_table = [
            self._compute_lifecycle_phase_at_step(s) for s in range(n_steps)
        ]
        lam = np.asarray(self._lambda_table, dtype=np.float64)
        indices = (
            np.arange(n_steps, dtype=np.int64)[:, None]
            + np.asarray(horizons, dtype=np.int64)[None, :]
        )
        forecast = (lam[indices] / self._demand_forecast_scale).astype(np.float32)
        np.clip(forecast, 0.0, 1.0, out=forecast)
        self._forecast_table = forecast

        # Erwartete Nachfrage der Tage t+1 ... t+L+1 für eine Bestellung an Tag t.
        cumsum = np.concatenate([[0.0], np.cumsum(lam)])
        t = np.arange(n_steps)
        self._lt_demand_table = cumsum[t + self._protection_days + 1] - cumsum[t + 1]

    def _build_service_windows(self) -> None:
        """Feste Fenster je Lebenszyklusphase (Reporting bzw. reward_mode="window")."""
        self._service_windows = []
        self._service_window_id_by_step.fill(-1)
        t = 0
        while t < self.episode_length:
            phase = self._phase_table[t]
            run_end = t + 1
            while run_end < self.episode_length and self._phase_table[run_end] == phase:
                run_end += 1
            if phase in _SERVICE_PHASES:
                starts = list(range(t, run_end, self.service_window_days))
                # NEU: zu kurzes Restfenster dem vorherigen Fenster zuschlagen
                if len(starts) > 1 and run_end - starts[-1] < self.service_window_days // 2:
                    starts.pop()
                for i, start in enumerate(starts):
                    end = starts[i + 1] if i + 1 < len(starts) else run_end
                    window_id = len(self._service_windows)
                    self._service_windows.append({
                        "id": window_id, "phase": phase, "start": start, "end": end,
                    })
                    self._service_window_id_by_step[start:end] = window_id
            t = run_end

        self._service_window_demand = 0
        self._service_window_sold = 0
        self._service_windows_closed = 0
        self._service_windows_scored = 0
        self._service_windows_meeting_target = 0
        self._episode_total_demand = 0
        self._episode_total_sold = 0

    # ==================================================================
    # Observation
    # ==================================================================
    def _lambda_at_step(self, step: int) -> float:
        if 0 <= step < len(self._lambda_table):
            return self._lambda_table[step]
        return self._compute_demand_lambda_at_step(step)

    def _get_lead_time_demand(self) -> float:
        step = self.step_count
        if 0 <= step < len(self._lt_demand_table):
            return float(self._lt_demand_table[step])
        return float(sum(
            self._lambda_at_step(step + h) for h in range(1, self._protection_days + 1)
        ))

    def _get_inventory_position(self) -> int:
        return int(self.inventory + self._in_transit_total)

    def _get_in_transit(self) -> int:
        return int(self._in_transit_total)

    def _get_open_shipment_features(self) -> np.ndarray:
        features = np.zeros((self.max_visible_shipments, 2), dtype=np.float32)
        if not self._shipments:
            return features
        visible = self._shipments[::-1][:self.max_visible_shipments]
        data = np.array(
            [(s["qty"], s["placed_step"]) for s in visible], dtype=np.float64
        )
        age = self.step_count - data[:, 1]
        np.clip(age, 0.0, self._max_lead_time_float, out=age)
        features[:len(visible), 0] = data[:, 0] / self._quantity_scale
        features[:len(visible), 1] = age / self._age_scale
        np.clip(features, 0.0, 1.0, out=features)
        return features

    def _get_demand_forecast_features(self) -> np.ndarray:
        step = self.step_count
        if 0 <= step < self._forecast_table.shape[0]:
            return self._forecast_table[step].copy()
        forecast = [
            self._lambda_at_step(step + h) / self._demand_forecast_scale
            for h in self._forecast_horizons_int
        ]
        return np.clip(np.asarray(forecast, dtype=np.float32), 0.0, 1.0)

    def _get_service_period_status(self) -> np.ndarray:
        if not 0 <= self.step_count < self.episode_length:
            return np.zeros(3, dtype=np.float32)
        window_id = int(self._service_window_id_by_step[self.step_count])
        if window_id < 0:
            return np.zeros(3, dtype=np.float32)
        window = self._service_windows[window_id]
        length = window["end"] - window["start"]
        fill_rate = (
            self._service_window_sold / self._service_window_demand
            if self._service_window_demand > 0 else 0.0
        )
        progress = (self.step_count - window["start"]) / length
        demand_scale = max(1.0, length * self.demand_lambda)
        return np.asarray([
            fill_rate,
            progress,
            min(1.0, self._service_window_demand / demand_scale),
        ], dtype=np.float32)

    def _get_rolling_fill_rate(self) -> Optional[float]:
        total_demand = int(self._roll_demand.sum())
        if total_demand <= 0:
            return None
        return float(self._roll_sold.sum()) / total_demand

    def _get_rolling_service_obs(self) -> np.ndarray:
        fr = self._get_rolling_fill_rate()
        scale = max(1.0, self.service_rolling_days * self.demand_lambda)
        return np.asarray([
            1.0 if fr is None else fr,
            min(1.0, float(self._roll_demand.sum()) / scale),
        ], dtype=np.float32)

    def _get_obs(self) -> dict[str, np.ndarray]:
        inventory = self.inventory / self._capacity_scale
        demand = np.clip(self.demand / self.demand_obs_cap, 0.0, 1.0)
        days_from_release = np.clip(
            (self.step_count - self.demand_warmup_periods) / self.days_from_release_scale,
            -1.0, 1.0,
        )
        in_transit = (
            self._in_transit_total / self.max_in_transit
            if self.max_in_transit > 0.0 else 0.0
        )
        inventory_position = min(1.0, self._get_inventory_position() / self._ip_scale)
        lead_time_demand = min(1.0, self._get_lead_time_demand() / self._lt_demand_scale)
        obs = {
            "inventory": np.array([inventory], dtype=np.float32),
            "demand": np.array([demand], dtype=np.float32),
            "days_from_release": np.array([days_from_release], dtype=np.float32),
            "lifecycle_progress": np.array(
                [self._get_lifecycle_progress()], dtype=np.float32
            ),
            "in_transit": np.array([in_transit], dtype=np.float32),
            "inventory_position": np.array([inventory_position], dtype=np.float32),
            "lead_time_demand": np.array([lead_time_demand], dtype=np.float32),
            "demand_forecast": self._get_demand_forecast_features(),
            "service_period_status": self._get_service_period_status(),
            "rolling_service": self._get_rolling_service_obs(),
        }
        if self.include_shipment_details:
            obs["order_history"] = np.clip(
                self.order_history / self._order_scale, 0.0, 1.0
            ).astype(np.float32)
            obs["shipment_features"] = self._get_open_shipment_features()
        return obs

    # ==================================================================
    # Nachfrage und Lebenszyklus (unverändert)
    # ==================================================================
    @staticmethod
    def _smoothstep(value: float) -> float:
        x = float(np.clip(value, 0.0, 1.0))
        return x * x * (3.0 - 2.0 * x)

    def _get_days_from_release(self) -> int:
        return int(self.step_count) - self.demand_warmup_periods

    def _get_product_age(self) -> int:
        return max(0, self._get_days_from_release())

    def _get_product_age_at_step(self, step: int) -> int:
        return max(0, self._get_days_from_release_at_step(step))

    def _get_days_from_release_at_step(self, step: int) -> int:
        return step - self.demand_warmup_periods

    def _get_lifecycle_progress(self) -> float:
        step = self.step_count
        if 0 <= step < len(self._progress_table):
            return self._progress_table[step]
        return self._get_lifecycle_progress_at_step(step)

    def _get_lifecycle_progress_at_step(self, step: int) -> float:
        if step < self.demand_warmup_periods:
            return 0.0
        age = self._get_product_age_at_step(step)
        length = self.demand_lifecycle_length
        if self.repeat_product_lifecycle:
            return float((age % length) / max(1, length - 1))
        return float(np.clip(age / max(1, length - 1), 0.0, 1.0))

    def _get_lifecycle_phase(self) -> str:
        step = self.step_count
        if 0 <= step < len(self._phase_table):
            return self._phase_table[step]
        return self._compute_lifecycle_phase_at_step(step)

    def _compute_lifecycle_phase_at_step(self, step: int) -> str:
        if step < self.demand_warmup_periods:
            return "warmup"
        age = self._get_product_age_at_step(step)
        progress = self._get_lifecycle_progress_at_step(step)
        intro_end = self.demand_intro_fraction
        growth_end = intro_end + self.demand_growth_fraction
        maturity_end = growth_end + self.demand_maturity_fraction
        if progress < intro_end:
            return "introduction"
        if progress < growth_end:
            return "growth"
        if progress < maturity_end:
            return "maturity"
        if not self.repeat_product_lifecycle and age >= self.demand_lifecycle_length:
            return "saturation"
        return "decline"

    def _get_current_demand_lambda(self) -> float:
        return self._lambda_at_step(self.step_count)

    def _compute_demand_lambda_at_step(self, step: int) -> float:
        if step < self.demand_warmup_periods:
            return 0.0
        peak = self.demand_lambda
        start = peak * self.demand_start_ratio
        intro_target = peak * self.demand_intro_end_ratio
        end = peak * self.demand_end_ratio
        progress = self._get_lifecycle_progress_at_step(step)
        intro_end = self.demand_intro_fraction
        growth_end = intro_end + self.demand_growth_fraction
        maturity_end = growth_end + self.demand_maturity_fraction
        if progress < intro_end:
            local = progress / intro_end if intro_end > 0.0 else 1.0
            return float(start + self._smoothstep(local) * (intro_target - start))
        if progress < growth_end:
            width = self.demand_growth_fraction
            local = (progress - intro_end) / width if width > 0.0 else 1.0
            return float(intro_target + self._smoothstep(local) * (peak - intro_target))
        if progress < maturity_end:
            return float(peak)
        width = self.demand_decline_fraction
        local = (progress - maturity_end) / width if width > 0.0 else 1.0
        weight = self._smoothstep(float(np.clip(local, 0.0, 1.0)))
        return float(peak + weight * (end - peak))

    def _sample_demand(self) -> int:
        mean = self._lambda_at_step(self.step_count)
        return int(self.np_random.poisson(mean)) if mean > 0.0 else 0

    def _get_future_demand_lambda(self, days_ahead: int) -> float:
        if days_ahead < 0:
            raise ValueError("days_ahead must be >= 0")
        return self._lambda_at_step(self.step_count + days_ahead)

    # ==================================================================
    # Lieferungen
    # ==================================================================
    def _sample_lead_time(self) -> int:
        if self.lead_time_std == 0:
            lead_time = int(round(self.lead_time_mean))
        else:
            lead_time = int(round(self.np_random.normal(
                loc=self.lead_time_mean, scale=self.lead_time_std
            )))
        return int(np.clip(lead_time, 1, self.max_lead_time))

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
        self._in_transit_total -= arriving
        return arriving

    def _place_end_of_period_order(self, order_qty: int) -> Optional[int]:
        if order_qty <= 0:
            return None
        lead_time = self._sample_lead_time()
        self._shipments.append({
            "id": self._next_shipment_id,
            "qty": int(order_qty),
            "remaining": lead_time,
            "initial_lead_time": lead_time,
            "placed_step": int(self.step_count),
        })
        self._in_transit_total += int(order_qty)
        self._next_shipment_id += 1
        return lead_time

    def _update_order_history(self, order_qty: int) -> None:
        if self.order_history_length > 1:
            self.order_history[1:] = self.order_history[:-1]
        self.order_history[0] = float(order_qty)

    # ==================================================================
    # Aktion
    # ==================================================================
    def _is_valid_action(self, action) -> bool:
        if (
            type(action) is np.ndarray
            and action.dtype is _FLOAT32
            and action.shape == (1,)
        ):
            return self._action_low <= float(action[0]) <= self._action_high
        return self.action_space.contains(action)

    def _action_to_order(self, a: float) -> tuple[int, Optional[float], Optional[float]]:
        """Übersetzt die Aktion in eine Bestellmenge.

        Muss NACH Verkauf und Ankünften aufgerufen werden, damit die
        Inventory Position aktuell ist.
        Rückgabe: (Bestellmenge, Faktor k, Bestellniveau S)
        """
        if self.action_mode == "order_qty":
            return int(np.clip(a, 0.0, 1.0) * self.max_order), None, None
        k = max(0.0, 1.0 + self.base_stock_spread * float(np.clip(a, -1.0, 1.0)))
        if self.action_mode == "relative_order":
            # Erwartete Nachfrage am ersten Tag, an dem die Lieferung verkauft
            # werden kann (t + L + 1).
            reference = self._lambda_at_step(self.step_count + self._protection_days)
            qty = int(np.clip(round(k * reference), 0, self.max_order))
            return qty, k, None
        level = k * self._get_lead_time_demand()
        qty = int(np.clip(round(level - self._get_inventory_position()), 0, self.max_order))
        return qty, k, float(level)

    # ==================================================================
    # Reset
    # ==================================================================
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ):
        super().reset(seed=seed)
        options = options or {}
        inventory = int(options.get("initial_inventory", self.initial_inventory))
        if not 0 <= inventory <= self.capacity:
            raise ValueError("initial_inventory must be between 0 and capacity")
        self.inventory = inventory
        self.step_count = 0
        self._shipments = []
        self._in_transit_total = 0
        self._next_shipment_id = 0
        self.last_info = {}
        self.last_reward = 0.0
        self._roll_sold.fill(0)
        self._roll_demand.fill(0)
        self._roll_pos = 0
        self._build_service_windows()
        self.demand = int(options.get("initial_demand", self._sample_demand()))
        if self.demand < 0:
            raise ValueError("initial_demand must be >= 0")
        self.order_history = np.zeros(self.order_history_length, dtype=np.float32)
        obs = self._get_obs()
        info = {
            "inventory": self.inventory,
            "demand": self.demand,
            "demand_lambda_current": self._get_current_demand_lambda(),
            "lifecycle_progress": self._get_lifecycle_progress(),
            "days_from_release": self._get_days_from_release(),
            "lifecycle_phase": self._get_lifecycle_phase(),
            "in_transit": 0,
            "service_target": self.service_target,
            "service_period_count": len(self._service_windows),
            "service_window_days": self.service_window_days,
            "service_periods": [w.copy() for w in self._service_windows],
        }
        return obs, info

    # ==================================================================
    # Service-Reward
    # ==================================================================
    def _rolling_service_reward(self, *, sold: int, demand: int) -> tuple[float, Optional[float]]:
        """Täglicher Reward aus der Fill Rate der letzten service_rolling_days Tage."""
        if self.step_count < self.demand_warmup_periods:
            return 0.0, None
        i = self._roll_pos % self.service_rolling_days
        self._roll_sold[i] = sold
        self._roll_demand[i] = demand
        self._roll_pos += 1
        if float(self._roll_demand.sum()) < self.service_min_rolling_demand:
            return 0.0, self._get_rolling_fill_rate()
        fr = self._get_rolling_fill_rate()
        return self.service_reward_function(fr), fr

    def _update_service_reward(
        self, *, sold: int, demand: int
    ) -> tuple[float, dict[str, Any]]:
        """Feste Fenster: Zähler aktualisieren und Fenster-Reward berechnen.

        Der Reward wird nur bei reward_mode="window" benutzt; die Zähler
        laufen immer mit (Bewertungsmetrik im info-Dict).
        """
        self._episode_total_demand += demand
        self._episode_total_sold += sold

        window_id = int(self._service_window_id_by_step[self.step_count])
        window = self._service_windows[window_id] if window_id >= 0 else None
        fill_rate = None
        score = None
        closed = False
        reward = 0.0

        if window is not None:
            previous_score = 0.0
            if self.dense_service_reward and self._service_window_demand > 0:
                previous_rate = self._service_window_sold / self._service_window_demand
                previous_score = self.service_reward_function(previous_rate)

            self._service_window_demand += demand
            self._service_window_sold += sold
            if self._service_window_demand > 0:
                fill_rate = self._service_window_sold / self._service_window_demand
                score = self.service_reward_function(fill_rate)

            closed = self.step_count + 1 == window["end"]
            if self.dense_service_reward:
                reward = (score if score is not None else 0.0) - previous_score
            elif closed and score is not None:
                reward = score

            if closed:
                self._service_windows_closed += 1
                if fill_rate is not None:
                    self._service_windows_scored += 1
                    if fill_rate >= self.service_target:
                        self._service_windows_meeting_target += 1

        # ÄNDERUNG: keine Division durch die Fensteranzahl mehr.
        info = self._get_service_reward_info(window, fill_rate, score, closed, reward)

        if closed:
            self._service_window_sold = 0
            self._service_window_demand = 0

        return float(reward), info

    def _get_service_reward_info(
        self,
        window: Optional[dict[str, Any]],
        fill_rate: Optional[float],
        score: Optional[float],
        closed: bool,
        reward: float,
    ) -> dict[str, Any]:
        """Reporting der festen Fenster."""
        scored = self._service_windows_scored
        return {
            "service_period_id": window["id"] if window else None,
            "service_period_phase": window["phase"] if window else None,
            "service_period_start": window["start"] if window else None,
            "service_period_end_exclusive": window["end"] if window else None,
            "service_period_length_days": (
                window["end"] - window["start"] if window else None
            ),
            "service_period_closed": closed,
            "service_period_fill_rate": fill_rate,
            "service_period_demand": self._service_window_demand if window else None,
            "service_period_sold": self._service_window_sold if window else None,
            "service_period_score": score,
            "window_service_reward": float(reward),
            "service_periods_closed": self._service_windows_closed,
            "service_periods_scored": scored,
            "service_periods_meeting_target": self._service_windows_meeting_target,
            "service_periods_target_rate": (
                self._service_windows_meeting_target / scored if scored else None
            ),
            "episode_fill_rate": (
                self._episode_total_sold / self._episode_total_demand
                if self._episode_total_demand else None
            ),
            "episode_total_demand": self._episode_total_demand,
            "episode_total_sold": self._episode_total_sold,
        }

    # ==================================================================
    # Step
    # ==================================================================
    def step(self, action):
        if not self._is_valid_action(action):
            raise ValueError(f"Invalid action {action!r}")
        a = float(action[0])

        # 1. Verkauf aus dem Lager
        current_demand = int(self.demand)
        inventory = self.inventory
        sold = min(inventory, current_demand)
        lost_sales = current_demand - sold
        inventory -= sold

        # 2. Ankünfte (begrenzt durch Kapazität)
        arriving = self._advance_existing_shipments()
        free_capacity = self.capacity - inventory
        accepted = min(arriving, free_capacity)
        overflow = max(arriving - free_capacity, 0)
        inventory += accepted
        self.inventory = inventory

        # 3. Bestellung (ÄNDERUNG: erst nach den Ankünften berechnet,
        #    damit die Inventory Position stimmt)
        inventory_position_before_order = self._get_inventory_position()
        order_qty, base_stock_k, base_stock_level = self._action_to_order(a)
        sampled_lead_time = self._place_end_of_period_order(order_qty)

        # 4. Finanzgrößen (nur Reporting)
        revenue = sold * self.selling_price
        inventory_holding_cost = inventory * self.holding_cost
        excess_order_cost = overflow * self.overflow_cost
        lost_sales_penalty = lost_sales * self.lost_sales_cost
        ordering_cost = order_qty * self.purchase_cost

        # 5. Reward
        rolling_reward, rolling_fill_rate = self._rolling_service_reward(
            sold=sold, demand=current_demand
        )
        window_reward, service_info = self._update_service_reward(
            sold=sold, demand=current_demand
        )
        service_reward = rolling_reward if self.reward_mode == "rolling" else window_reward
        # ÄNDERUNG: tägliche Strafen in beiden Modi
        inventory_penalty = self.inventory_penalty_weight * inventory / self._capacity_scale
        overflow_penalty = self.overflow_penalty_weight * overflow / self._capacity_scale
        reward = float(service_reward - inventory_penalty - overflow_penalty)
        self.last_reward = reward

        # 6. Weiterschalten
        self._update_order_history(order_qty)
        self.step_count += 1
        step = self.step_count
        self.demand = self._sample_demand()
        terminated = False
        truncated = step >= self.episode_length

        info = {
            "ordered_at_period_end": order_qty,
            "sampled_lead_time": sampled_lead_time,
            "base_stock_k": base_stock_k,
            "base_stock_level": base_stock_level,
            "inventory_position_before_order": inventory_position_before_order,
            "reward": reward,
            "service_reward": float(service_reward),
            "rolling_service_reward": float(rolling_reward),
            "rolling_fill_rate": rolling_fill_rate,
            "inventory_penalty": float(inventory_penalty),
            "overflow_penalty": float(overflow_penalty),
            "service_target": self.service_target,
            "current_demand": current_demand,
            "next_demand": self.demand,
            "demand_lambda_next": self._lambda_at_step(step),
            "lifecycle_progress": self._get_lifecycle_progress(),
            "days_from_release": step - self.demand_warmup_periods,
            "lifecycle_phase": self._get_lifecycle_phase(),
            "sold": sold,
            "lost_sales": lost_sales,
            "arriving_before_order": arriving,
            "accepted": accepted,
            "overflow": overflow,
            "inventory_end": inventory,
            "in_transit_after_order": self._in_transit_total,
            "revenue": float(revenue),
            "ordering_cost": float(ordering_cost),
            "holding_cost": float(inventory_holding_cost),
            "overflow_cost": float(excess_order_cost),
            "lost_sales_cost": float(lost_sales_penalty),
            **service_info,
        }
        self.last_info = info
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), reward, terminated, truncated, info

    # ==================================================================
    # Rendering
    # ==================================================================
    def render(self):
        if self.render_mode is None:
            return None
        if self.renderer is None:
            try:
                from .supply_chain_renderer_nons import SupplyChainRendererNonS
            except ImportError:
                from RLII_26.rl_zoo3.custom_envs.supply_chain_renderer_nons import (
                    SupplyChainRendererNonS
                )
            self.renderer = SupplyChainRendererNonS(
                render_mode=self.render_mode, fps=self.metadata["render_fps"]
            )
        return self.renderer.render(self, animate=self.render_mode == "human")

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None