from math import ceil
from typing import Any

from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.td3.policies import TD3Policy

from rl_zoo3.custom_algos.td3_target_interval import TD3WithTargetInterval


class ICTD3(TD3WithTargetInterval):
    """TD3 with increasing intervals between target network updates.

    :param initial_target_update_interval: Initial interval in critic gradient steps.
        Defaults to ``policy_delay``, which still controls online actor updates.
    :param geometric_target_update_intervals: Grow the interval in cycle ``n`` to
        ``ceil(initial_target_update_interval * gamma ** (-2 * n / 3))``.
        When disabled, use the initial interval throughout training.
    """

    def __init__(
        self,
        policy: str | type[TD3Policy],
        env: GymEnv | str | None,
        *args: Any,
        initial_target_update_interval: int | None = None,
        geometric_target_update_intervals: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(policy, env, *args, target_update_interval=initial_target_update_interval, **kwargs)
        if geometric_target_update_intervals and not 0 < self.gamma <= 1:
            raise ValueError("Geometric target update intervals require 0 < gamma <= 1")

        self.initial_target_update_interval = self.target_update_interval
        self.geometric_target_update_intervals = geometric_target_update_intervals

    def _get_next_target_update_interval(self) -> int:
        if not self.geometric_target_update_intervals:
            return self.current_target_update_interval
        growth = self.gamma ** (-2 * self._n_target_updates / 3)
        return ceil(self.initial_target_update_interval * growth)
