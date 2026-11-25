from typing import Any

from stable_baselines3 import DDPG
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.td3.policies import TD3Policy

from rl_zoo3.custom_algos.td3_target_interval import TD3WithTargetInterval


class DDPGWithTargetInterval(DDPG, TD3WithTargetInterval):
    """SB3 DDPG with an independent actor and critic target update interval.

    :param target_update_interval: Critic gradient steps between target updates.
        Defaults to one, matching SB3 DDPG's target update cadence.
    """

    def __init__(
        self,
        policy: str | type[TD3Policy],
        env: GymEnv | str | None,
        *args: Any,
        target_update_interval: int | None = None,
        **kwargs: Any,
    ) -> None:
        # SB3's DDPG constructor applies its defaults before setting up the networks.
        super().__init__(policy, env, *args, **kwargs)  # type: ignore[arg-type]
        if target_update_interval is not None:
            self._initialize_target_update_schedule(target_update_interval)
