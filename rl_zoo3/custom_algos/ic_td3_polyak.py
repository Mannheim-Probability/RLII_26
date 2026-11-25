from operator import index
from typing import Any

from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.td3.policies import TD3Policy

from rl_zoo3.custom_algos.ic_td3 import geometric_cycle_length
from rl_zoo3.custom_algos.td3_target_interval import TD3WithTargetInterval


class ICTD3Polyak(TD3WithTargetInterval):
    """Update targets every critic step using ``tau = 2 / (K_n + 1)`` for each IC cycle.

    :param initial_tau_cycle_length: Number of critic steps in the first cycle.
        Later cycles grow geometrically as in IC-TD3.
    """

    def __init__(
        self,
        policy: str | type[TD3Policy],
        env: GymEnv | str | None,
        *args: Any,
        initial_tau_cycle_length: int = 2,
        **kwargs: Any,
    ) -> None:
        initial_length = index(initial_tau_cycle_length)
        if initial_length < 1:
            raise ValueError("initial_tau_cycle_length must be positive")
        if "tau" in kwargs:
            raise ValueError("tau is derived from initial_tau_cycle_length and gamma")
        if kwargs.pop("target_update_interval", 1) != 1:
            raise ValueError("ICTD3Polyak updates targets after every critic step")

        super().__init__(policy, env, *args, tau=2.0 / (initial_length + 1), target_update_interval=1, **kwargs)
        if not 0 < self.gamma <= 1:
            raise ValueError("Geometric tau cycles require 0 < gamma <= 1")

        self.initial_tau_cycle_length = initial_length
        self._tau_cycle_index = 0
        self._tau_cycle_length = initial_length
        self._tau_cycle_end = initial_length
        self.target_parameter_age = 0.0

    def _update_target_networks(self) -> None:
        # The last critic step of a cycle still uses that cycle's coefficient.
        if self._n_updates > self._tau_cycle_end:
            self._tau_cycle_index += 1
            self._tau_cycle_length = geometric_cycle_length(self.initial_tau_cycle_length, self.gamma, self._tau_cycle_index)
            self._tau_cycle_end += self._tau_cycle_length

        self.tau = 2.0 / (self._tau_cycle_length + 1)
        super()._update_target_networks()
        self.target_parameter_age = (1 - self.tau) * (self.target_parameter_age + 1)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        super().train(gradient_steps, batch_size)
        self.logger.record("train/tau", self.tau)
        self.logger.record("train/tau_cycle_index", self._tau_cycle_index)
        self.logger.record("train/tau_cycle_length", self._tau_cycle_length)
        self.logger.record("train/target_parameter_age", self.target_parameter_age)
