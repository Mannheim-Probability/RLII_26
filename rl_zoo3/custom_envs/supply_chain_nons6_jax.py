"""JAX/Jaxnasium port of SupplyChainNonS6Env (standalone module).

Install in the Python environment that runs your training:
    python -m pip install jaxnasium

Single environment (state is explicit; keys must not be reused):
    import jax
    import jax.numpy as jnp
    from supply_chain_nons6_jax import SupplyChainNonS6Env

    env = SupplyChainNonS6Env()
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    key = jax.random.key(0)
    key, reset_key = jax.random.split(key)
    obs, state = reset(reset_key)
    key, step_key = jax.random.split(key)
    (obs, reward, terminated, truncated, info), state = step(
        step_key, state, jnp.array([0.5], dtype=jnp.float32)
    )

Vectorized environments:
    batch_reset = jax.jit(jax.vmap(env.reset))
    batch_step = jax.jit(jax.vmap(env.step))
    key, reset_key = jax.random.split(key)
    obs, states = batch_reset(jax.random.split(reset_key, 64))
    key, step_key = jax.random.split(key)
    timesteps, states = batch_step(
        jax.random.split(step_key, 64), states, jnp.zeros((64, 1))
    )

Jaxnasium PPO (requires `python -m pip install "jaxnasium[algs]"`):
    import jaxnasium as jym
    from jaxnasium.algorithms import PPO
    env = jym.FlattenObservationWrapper(SupplyChainNonS6Env())
    env = jym.LogWrapper(env)
    agent, metrics = PPO(total_timesteps=100_000).train(jax.random.key(0), env)

Migration notes:
* This is a Jaxnasium environment, not a drop-in Gymnasium / SB3 environment.
  reset(key) returns (observation, state); step(key, state, action) returns
  (TimeStep(observation, reward, terminated, truncated, info), state).
* Jaxnasium's inherited step() automatically resets at episode end. For the
  original manual-reset behavior, use step_env() instead. step() retains the
  final observation in info["_TERMINAL_OBSERVATION"].
* Actions are finite float arrays of shape (1,). As in the original active
  code, order_qty = int(clip(action[0], 0, 1) * max_order).
  0 means no order, 0.5 means half the maximum, and 1 means the maximum order.
  Bounds are clipped; data-dependent Python exceptions are not used under JIT.
* Observation names, shapes, normalization, reward / 2000, and order timing
  match the supplied code. A lead-time-1 order cannot satisfy the next demand;
  it arrives after that demand is served. A newly observed order has age 1,
  matching the original placement before the step_count increment.
* shipment_features has shape (max_lead_time, 2): [quantity / max_order,
  age / max_lead_time], with active shipments newest first, then zero rows.
  The order_history observation is also retained, newest order first.
  Jaxnasium's default PPO network requires FlattenObservationWrapper for this
  two-dimensional feature array, as shown above. The wrapper keeps the dict
  keys and flattens shipment_features to a vector of length 2 * max_lead_time.
* Demand, days_from_release, and in_transit use this S6 version's original
  normalization scales; these differ from the S7 environment.
* The Python lookup tables are replaced by compiled, vectorized JAX formulas.
  They also work beyond episode_length without out-of-bounds table accesses.
  Configuration is immutable; construct a new environment to change it.
* Hidden remaining lead times live only in state. As in the source, diagnostic
  info includes the sampled lead time; never feed hidden state/info to a policy.
* info["sampled_lead_time"] is 0 for no order (previously None).
  info["lifecycle_phase"] is an integer index into LIFECYCLE_PHASE_NAMES
  (previously a string). All info leaves have fixed shapes and JAX dtypes.
* Initial inventory and optional initial demand are constructor settings;
  Gymnasium reset(seed=..., options=...) is replaced by reset(key).
* JAX and NumPy have different random generators: equal integer seeds do not
  imply identical trajectories. The demand and lead-time distributions match.
* The external GUI renderer was not supplied. render_mode must be None.

API references:
https://ponseko.github.io/jaxnasium/api/Environment/
https://ponseko.github.io/jaxnasium/api/Spaces/
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import jaxnasium as jym


LIFECYCLE_PHASE_NAMES = (
    "warmup", "introduction", "growth", "maturity", "decline", "saturation"
)


class SupplyChainState(jym.EnvState):
    """Immutable PyTree; every leaf is an array, including scalar counters.

    A zero shipment quantity denotes a free slot. There are max_lead_time
    slots; at most one order is placed per period and all lead times are capped.
    The agent observes get_observation(state), not this internal simulator state.
    """

    inventory: jax.Array
    demand: jax.Array
    step_count: jax.Array
    shipment_quantities: jax.Array
    shipment_remaining: jax.Array
    shipment_placed_steps: jax.Array
    order_history: jax.Array


class SupplyChainNonS6EnvJax(jym.Environment[SupplyChainState]):
    """Single-product lost-sales environment with stochastic replenishment.

    Configuration is static during a compiled rollout. Create a new environment
    and compiled functions to change configuration. All mutable episode data
    is in SupplyChainState. The eight observation fields form one agent's dict.
    """

    capacity: int = eqx.field(static=True, default=1000)
    max_order: int | None = eqx.field(static=True, default=None)
    demand_lambda: float = eqx.field(static=True, default=100.0)
    demand_lifecycle_length: int = eqx.field(static=True, default=730)
    repeat_product_lifecycle: bool = eqx.field(static=True, default=False)
    demand_warmup_periods: int | None = eqx.field(static=True, default=None)
    demand_intro_fraction: float = eqx.field(static=True, default=0.10)
    demand_growth_fraction: float = eqx.field(static=True, default=0.30)
    demand_maturity_fraction: float = eqx.field(static=True, default=0.35)
    demand_decline_fraction: float = eqx.field(static=True, default=0.25)
    demand_start_ratio: float = eqx.field(static=True, default=0.00)
    demand_intro_end_ratio: float = eqx.field(static=True, default=0.40)
    demand_end_ratio: float = eqx.field(static=True, default=0.15)
    lead_time_mean: float = eqx.field(static=True, default=60.0)
    lead_time_std: float = eqx.field(static=True, default=2.0)
    max_lead_time: int = eqx.field(static=True, default=100)
    selling_price: float = eqx.field(static=True, default=20.0)
    purchase_cost: float = eqx.field(static=True, default=2.0)
    holding_cost: float = eqx.field(static=True, default=0.25)
    overflow_cost: float = eqx.field(static=True, default=8.0)
    lost_sales_cost: float = eqx.field(static=True, default=10.0)
    episode_length: int = eqx.field(static=True, default=1024)
    initial_inventory: int | None = eqx.field(static=True, default=None)
    order_history_length: int | None = eqx.field(static=True, default=None)
    render_mode: str | None = eqx.field(static=True, default=None)
    initial_demand: int | None = eqx.field(static=True, default=None)

    def __post_init__(self):
        # Host-side validation only: no traced values are converted to Python.
        for name in ("capacity", "max_lead_time", "episode_length",
                     "demand_lifecycle_length"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value:
                raise ValueError(f"{name} must be an integer")
            object.__setattr__(self, name, int(value))
            if value <= (1 if name == "demand_lifecycle_length" else 0):
                raise ValueError(f"{name} is too small")

        defaults = {
            "max_order": self.capacity,
            "demand_warmup_periods": self.max_lead_time,
            "initial_inventory": self.capacity // 2,
            "order_history_length": self.max_lead_time,
        }
        for name, default in defaults.items():
            value = getattr(self, name)
            value = default if value is None else value
            if isinstance(value, bool) or int(value) != value:
                raise ValueError(f"{name} must be an integer")
            object.__setattr__(self, name, int(value))

        if self.max_order < 0 or self.demand_warmup_periods < 0:
            raise ValueError("max_order and demand_warmup_periods must be >= 0")
        if not 0 <= self.initial_inventory <= self.capacity:
            raise ValueError("initial_inventory must be between 0 and capacity")
        if self.order_history_length <= 0:
            raise ValueError("order_history_length must be > 0")
        if self.initial_demand is not None:
            if int(self.initial_demand) != self.initial_demand or self.initial_demand < 0:
                raise ValueError("initial_demand must be a nonnegative integer")
            object.__setattr__(self, "initial_demand", int(self.initial_demand))

        float_fields = (
            "demand_lambda", "demand_intro_fraction", "demand_growth_fraction",
            "demand_maturity_fraction", "demand_decline_fraction",
            "demand_start_ratio", "demand_intro_end_ratio", "demand_end_ratio",
            "lead_time_mean", "lead_time_std", "selling_price", "purchase_cost",
            "holding_cost", "overflow_cost", "lost_sales_cost",
        )
        for name in float_fields:
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)

        fractions = (
            self.demand_intro_fraction, self.demand_growth_fraction,
            self.demand_maturity_fraction, self.demand_decline_fraction,
        )
        if any(value < 0 for value in fractions) or not math.isclose(
            sum(fractions), 1.0, rel_tol=1e-5, abs_tol=1e-8
        ):
            raise ValueError("demand lifecycle fractions must be >= 0 and sum to 1")
        if min(self.demand_lambda, self.demand_start_ratio,
               self.demand_intro_end_ratio, self.demand_end_ratio) < 0:
            raise ValueError("demand_lambda and demand ratios must be >= 0")
        if self.demand_intro_end_ratio < self.demand_start_ratio:
            raise ValueError("demand_intro_end_ratio must be >= demand_start_ratio")
        if self.lead_time_mean <= 0 or self.lead_time_std < 0:
            raise ValueError("lead_time_mean must be > 0 and lead_time_std >= 0")
        if self.render_mode is not None:
            raise NotImplementedError(
                "The external Gymnasium GUI renderer is not part of this JAX port; "
                "use render_mode=None."
            )

    @property
    def action_space(self):
        return jym.Box(low=0.0, high=1.0, shape=(1,), dtype=jnp.float32)

    @property
    def observation_space(self):
        return {
            "inventory": jym.Box(0.0, 1.0, (1,)),
            "demand": jym.Box(0.0, 1.0, (1,)),
            "days_from_release": jym.Box(-1.0, 1.0, (1,)),
            "lifecycle_progress": jym.Box(0.0, 1.0, (1,)),
            "in_transit": jym.Box(0.0, 1.0, (1,)),
            "order_history": jym.Box(0.0, 1.0, (self.order_history_length,)),
            "shipment_features": jym.Box(0.0, 1.0, (self.max_lead_time, 2)),
            "demand_forecast": jym.Box(0.0, 1.0, (5,)),
        }

    @property
    def demand_obs_cap(self):
        return max(1.0, self.demand_lambda + 6.0 * math.sqrt(self.demand_lambda + 1.0))

    @property
    def days_from_release_scale(self):
        return float(max(1, self.demand_warmup_periods,
                         self.episode_length - self.demand_warmup_periods))

    @property
    def max_in_transit(self):
        return float(self.max_order * self.max_lead_time)

    @property
    def max_visible_shipments(self):
        return self.max_lead_time

    @property
    def demand_forecast_horizons(self):
        lead = int(round(self.lead_time_mean))
        return jnp.asarray([0, max(1, lead // 2), lead + 1, lead + 15, lead + 30],
                           dtype=jnp.int32)

    @staticmethod
    def _smoothstep(value):
        x = jnp.clip(value, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    def lifecycle_progress(self, step):
        age = jnp.maximum(jnp.asarray(step, jnp.int32) - self.demand_warmup_periods, 0)
        if self.repeat_product_lifecycle:  # Static configuration branch.
            age = age % self.demand_lifecycle_length
        return jnp.clip(age.astype(jnp.float32) / (self.demand_lifecycle_length - 1),
                        0.0, 1.0)

    def demand_mean(self, step):
        """Poisson mean at a scalar step or array of forecast steps."""
        progress = self.lifecycle_progress(step)
        intro_end = self.demand_intro_fraction
        growth_end = intro_end + self.demand_growth_fraction
        maturity_end = growth_end + self.demand_maturity_fraction
        peak = self.demand_lambda
        start = peak * self.demand_start_ratio
        intro_target = peak * self.demand_intro_end_ratio
        end = peak * self.demand_end_ratio

        # Avoid zero denominators even in inactive branches of jnp.select.
        intro_local = progress / intro_end if intro_end > 0 else jnp.ones_like(progress)
        growth_local = ((progress - intro_end) / self.demand_growth_fraction
                        if self.demand_growth_fraction > 0 else jnp.ones_like(progress))
        decline_local = ((progress - maturity_end) / self.demand_decline_fraction
                         if self.demand_decline_fraction > 0 else jnp.ones_like(progress))
        mean = jnp.select(
            [progress < intro_end, progress < growth_end, progress < maturity_end],
            [start + self._smoothstep(intro_local) * (intro_target - start),
             intro_target + self._smoothstep(growth_local) * (peak - intro_target),
             jnp.full_like(progress, peak)],
            default=peak + self._smoothstep(decline_local) * (end - peak),
        )
        return jnp.where(jnp.asarray(step) < self.demand_warmup_periods,
                         0.0, mean).astype(jnp.float32)

    def lifecycle_phase(self, step):
        """JIT-safe phase ID, using the same branch order as the source."""
        progress = self.lifecycle_progress(step)
        intro_end = self.demand_intro_fraction
        growth_end = intro_end + self.demand_growth_fraction
        maturity_end = growth_end + self.demand_maturity_fraction
        return jnp.select(
            [step < self.demand_warmup_periods, progress < intro_end,
             progress < growth_end, progress < maturity_end,
             step - self.demand_warmup_periods >= self.demand_lifecycle_length],
            [0, 1, 2, 3, 5], default=4,
        ).astype(jnp.int32)

    def _sample_demand(self, key, step):
        mean = self.demand_mean(step)
        # Skipping Poisson at lambda=0 also guarantees exactly zero warmup demand.
        return jax.lax.cond(
            mean > 0,
            lambda k: jax.random.poisson(k, mean, shape=(), dtype=jnp.int32),
            lambda k: jnp.asarray(0, jnp.int32), key,
        )

    def _sample_lead_time(self, key):
        if self.lead_time_std == 0:
            return jnp.asarray(min(max(round(self.lead_time_mean), 1),
                                   self.max_lead_time), jnp.int32)
        sample = (self.lead_time_mean
                  + self.lead_time_std * jax.random.normal(key, dtype=jnp.float32))
        return jnp.clip(jnp.rint(sample), 1, self.max_lead_time).astype(jnp.int32)

    def get_shipment_features(self, state):
        """Compact active shipments newest first, then pad with zero rows.

        The internal circular buffer is reordered chronologically before
        compaction. nonzero's fixed `size` keeps output shapes static under JIT;
        ordinary boolean array indexing would produce a dynamic output shape.
        """
        slots_newest_first = (
            state.step_count - 1 - jnp.arange(self.max_lead_time, dtype=jnp.int32)
        ) % self.max_lead_time
        quantities = state.shipment_quantities[slots_newest_first]
        placed_steps = state.shipment_placed_steps[slots_newest_first]
        active = quantities > 0
        indices = jnp.nonzero(active, size=self.max_lead_time, fill_value=0)[0]
        valid_rows = jnp.arange(self.max_lead_time) < jnp.sum(active, dtype=jnp.int32)
        ages = jnp.clip(state.step_count - placed_steps[indices], 0, self.max_lead_time)
        features = jnp.stack([
            quantities[indices].astype(jnp.float32) / max(float(self.max_order), 1.0),
            ages.astype(jnp.float32) / self.max_lead_time,
        ], axis=-1)
        return jnp.where(valid_rows[:, None], jnp.clip(features, 0.0, 1.0), 0.0)

    def get_observation(self, state):
        days = state.step_count - self.demand_warmup_periods

        def scalar_feature(value):
            return jnp.asarray(value, jnp.float32).reshape((1,))

        return {
            "inventory": scalar_feature(state.inventory / self.capacity),
            "demand": scalar_feature(jnp.clip(state.demand / self.demand_obs_cap, 0.0, 1.0)),
            "days_from_release": scalar_feature(jnp.clip(
                days / self.days_from_release_scale, -1.0, 1.0)),
            "lifecycle_progress": scalar_feature(self.lifecycle_progress(state.step_count)),
            "in_transit": scalar_feature(jnp.sum(state.shipment_quantities)
                                         / max(self.max_in_transit, 1.0)),
            "order_history": jnp.clip(state.order_history / max(float(self.max_order), 1.0),
                                      0.0, 1.0),
            "shipment_features": self.get_shipment_features(state),
            "demand_forecast": jnp.clip(
                self.demand_mean(state.step_count + self.demand_forecast_horizons)
                / max(self.demand_lambda, 1e-8), 0.0, 1.0),
        }

    def reset_env(self, key):
        step_count = jnp.asarray(0, jnp.int32)
        demand = (self._sample_demand(key, step_count) if self.initial_demand is None
                  else jnp.asarray(self.initial_demand, jnp.int32))
        state = SupplyChainState(
            inventory=jnp.asarray(self.initial_inventory, jnp.int32),
            demand=demand,
            step_count=step_count,
            shipment_quantities=jnp.zeros((self.max_lead_time,), jnp.int32),
            shipment_remaining=jnp.zeros((self.max_lead_time,), jnp.int32),
            shipment_placed_steps=jnp.zeros((self.max_lead_time,), jnp.int32),
            order_history=jnp.zeros((self.order_history_length,), jnp.float32),
        )
        return self.get_observation(state), state

    def step_env(self, key, state, action):
        """Advance one period without auto-reset (use inherited step for training)."""
        action = jnp.asarray(action, dtype=jnp.float32)
        if action.shape != (1,):
            raise ValueError(f"action must have shape (1,), received {action.shape}")
        lead_key, demand_key = jax.random.split(key)
        order_qty = (jnp.clip(action[0], 0.0, 1.0) * self.max_order).astype(jnp.int32)

        # Fulfil current demand before any shipment arrives.
        sold = jnp.minimum(state.inventory, state.demand)
        lost_sales = state.demand - sold
        inventory = state.inventory - sold

        # Advance only shipments already present in the incoming state.
        active = state.shipment_quantities > 0
        arriving_mask = active & (state.shipment_remaining <= 1)
        arriving = jnp.sum(jnp.where(arriving_mask, state.shipment_quantities, 0),
                           dtype=jnp.int32)
        still_active = active & ~arriving_mask
        quantities = jnp.where(still_active, state.shipment_quantities, 0)
        remaining = jnp.where(still_active, state.shipment_remaining - 1, 0)
        placed = jnp.where(still_active, state.shipment_placed_steps, 0)

        free_capacity = self.capacity - inventory
        accepted = jnp.minimum(arriving, free_capacity)
        overflow = jnp.maximum(arriving - free_capacity, 0)
        inventory = inventory + accepted

        # Slots rotate once every max_lead_time steps. The previous shipment in
        # this slot has necessarily arrived by now, even at the maximum lead time.
        # Place the new order AFTER advancing all old shipments.
        sampled_lead_time = jnp.where(order_qty > 0, self._sample_lead_time(lead_key), 0)
        slot = state.step_count % self.max_lead_time
        quantities = quantities.at[slot].set(order_qty)
        remaining = remaining.at[slot].set(sampled_lead_time)
        placed = placed.at[slot].set(jnp.where(order_qty > 0, state.step_count, 0))

        revenue = sold.astype(jnp.float32) * self.selling_price
        ordering_cost = order_qty.astype(jnp.float32) * self.purchase_cost
        holding_cost = inventory.astype(jnp.float32) * self.holding_cost
        overflow_cost = overflow.astype(jnp.float32) * self.overflow_cost
        lost_sales_cost = lost_sales.astype(jnp.float32) * self.lost_sales_cost
        reward = ((revenue - ordering_cost - holding_cost - overflow_cost
                   - lost_sales_cost) / 2000.0).astype(jnp.float32)
        next_step = state.step_count + jnp.asarray(1, jnp.int32)
        next_demand = self._sample_demand(demand_key, next_step)
        next_state = SupplyChainState(
            inventory=inventory,
            demand=next_demand,
            step_count=next_step,
            shipment_quantities=quantities,
            shipment_remaining=remaining,
            shipment_placed_steps=placed,
            order_history=jnp.concatenate([
                order_qty.astype(jnp.float32).reshape((1,)), state.order_history[:-1]
            ]),
        )
        info = {
            "ordered_at_period_end": order_qty,
            "sampled_lead_time": sampled_lead_time,
            "reward": reward,
            "current_demand": state.demand,
            "next_demand": next_demand,
            "demand_lambda_next": self.demand_mean(next_step),
            "lifecycle_progress": self.lifecycle_progress(next_step),
            "days_from_release": next_step - self.demand_warmup_periods,
            "lifecycle_phase": self.lifecycle_phase(next_step),
            "sold": sold,
            "lost_sales": lost_sales,
            "arriving_before_order": arriving,
            "accepted": accepted,
            "overflow": overflow,
            "inventory_end": inventory,
            "in_transit_after_order": jnp.sum(quantities, dtype=jnp.int32),
            "revenue": revenue,
            "ordering_cost": ordering_cost,
            "holding_cost": holding_cost,
            "overflow_cost": overflow_cost,
            "lost_sales_cost": lost_sales_cost,
        }
        timestep = jym.TimeStep(
            observation=self.get_observation(next_state),
            reward=reward,
            terminated=jnp.asarray(False),
            truncated=next_step >= self.episode_length,
            info=info,
        )
        return timestep, next_state


if __name__ == "__main__":
    # Run `python supply_chain_nons6_jax.py` for a short compiled smoke test.
    env = SupplyChainNonS6EnvJax()
    key = jax.random.key(0)
    key, reset_key = jax.random.split(key)
    obs, state = jax.jit(env.reset)(reset_key)
    compiled_step = jax.jit(env.step)
    for _ in range(5):
        key, step_key = jax.random.split(key)
        timestep, state = compiled_step(step_key, state, jnp.array([0.5], jnp.float32))
        print(f"step={int(state.step_count)} inventory={int(state.inventory)} "
              f"in_transit={int(timestep.info['in_transit_after_order'])} "
              f"reward={float(timestep.reward):.4f}")
