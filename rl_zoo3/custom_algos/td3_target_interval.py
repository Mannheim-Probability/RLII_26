from operator import index
from typing import Any

import numpy as np
import torch as th
from stable_baselines3 import TD3
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.td3.policies import TD3Policy
from torch.nn import functional as F


class TD3WithTargetInterval(TD3):
    """TD3 with a fixed target update interval independent of the online actor.

    :param target_update_interval: Number of critic gradient steps between
        target updates. Defaults to ``policy_delay``, which controls the actor.
    """

    def __init__(
        self,
        policy: str | type[TD3Policy],
        env: GymEnv | str | None,
        *args: Any,
        target_update_interval: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(policy, env, *args, **kwargs)  # type: ignore[arg-type]
        interval = self.policy_delay if target_update_interval is None else target_update_interval
        self._initialize_target_update_schedule(interval)

    def _initialize_target_update_schedule(self, target_update_interval: int) -> None:
        interval = index(target_update_interval)
        if interval < 1:
            raise ValueError("target_update_interval must be positive")

        self.target_update_interval = interval
        self.current_target_update_interval = interval
        self._n_target_updates = 0
        self._next_target_update = interval

    def _get_next_target_update_interval(self) -> int:
        return self.target_update_interval

    def _update_target_networks(self) -> None:
        polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
        polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
        # Copy BatchNorm running statistics, as in SB3.
        polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)
        polyak_update(self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Follow SB3 TD3.train() with a separate condition for target updates.
        self.policy.set_training_mode(True)
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses = [], []
        for _ in range(gradient_steps):
            self._n_updates += 1
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            with th.no_grad():
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            if self._n_updates % self.policy_delay == 0:
                actor_loss = -self.critic.q1_forward(replay_data.observations, self.actor(replay_data.observations)).mean()
                actor_losses.append(actor_loss.item())
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

            if self._n_updates >= self._next_target_update:
                self._update_target_networks()
                self._n_target_updates += 1
                self.current_target_update_interval = self._get_next_target_update_interval()
                self._next_target_update = self._n_updates + self.current_target_update_interval

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/n_target_updates", self._n_target_updates)
        self.logger.record("train/target_update_interval", self.current_target_update_interval)
        if actor_losses:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
