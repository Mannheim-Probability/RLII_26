"""Student harness: modify these local subclasses, leaving installed SB3 unchanged.

discounting="none" runs ordinary PPO. Complete the four TODO areas for the
two homework modes. train() follows SB3 2.9.0; see SB3_LICENSE.
"""
from typing import NamedTuple

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.utils import explained_variance


class WeightedRolloutSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    weights: th.Tensor


class DiscountedRolloutBuffer(RolloutBuffer):
    def __init__(self, *args, discounting="none", **kwargs):
        self.discounting = discounting
        super().__init__(*args, **kwargs)
        self.episode_steps = np.zeros(self.n_envs, dtype=np.int64)

    def reset(self):
        super().reset()
        self.discount_weights = np.ones((self.buffer_size, self.n_envs), dtype=np.float64)
        self.loss_weights = None
        # episode_steps counts environment time and survives a rollout reset.

    def add(self, obs, action, reward, episode_start, value, log_prob):
        if self.discounting != "none":
            # TODO 1: reset episode counters where episode_start is true,
            # store gamma**t at self.pos, then advance each counter.
            raise NotImplementedError("Store episode-time discount weights in add().")
        super().add(obs, action, reward, episode_start, value, log_prob)

    def get(self, batch_size=None):
        assert self.full
        size = self.buffer_size * self.n_envs
        if self.discounting == "weighted_sampling":
            # TODO 2: replace the permutation with N weighted draws with replacement.
            # Indices must match SB3's environment-first flattened transition order.
            raise NotImplementedError("Implement weighted index selection in get().")
        indices = np.random.permutation(size)

        if not self.generator_ready:
            for name in ("observations", "actions", "values", "log_probs", "advantages", "returns", "discount_weights"):
                self.__dict__[name] = self.swap_and_flatten(self.__dict__[name])
            self.discount_weights = self.discount_weights.reshape(-1)
            self.loss_weights = np.ones(size, dtype=np.float32)
            if self.discounting == "loss_weighted":
                # TODO 3: normalize over the full rollout so the mean loss weight is one.
                raise NotImplementedError("Compute full-rollout loss weights in get().")
            self.generator_ready = True

        batch_size = batch_size or size
        for start in range(0, size, batch_size):
            yield self._get_samples(indices[start:start + batch_size])

    def _get_samples(self, batch_inds, env=None):
        data = super()._get_samples(batch_inds, env)
        return WeightedRolloutSamples(*data, self.to_torch(self.loss_weights[batch_inds]).float())


class DiscountedPPO(PPO):
    def __init__(self, *args, discounting="none", **kwargs):
        if discounting not in ("none", "loss_weighted", "weighted_sampling"):
            raise ValueError("Unknown discounting mode.")
        self.discounting = discounting
        kwargs["rollout_buffer_class"] = DiscountedRolloutBuffer
        kwargs["rollout_buffer_kwargs"] = {**(kwargs.get("rollout_buffer_kwargs") or {}), "discounting": discounting}
        super().__init__(*args, **kwargs)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        if self.discounting != "none":
            # TODO 4: use rollout_data.weights in the three loss reductions below.
            raise NotImplementedError("Implement the loss reductions in DiscountedPPO.train().")

        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                # TODO 4: loss-weighted mode uses the minibatch weights; other modes keep the mean.
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                # TODO 4: loss-weighted mode uses the minibatch weights; other modes keep the mean.
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                # TODO 4: loss-weighted mode uses the minibatch weights; other modes keep the mean.
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
