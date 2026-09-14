"""Worked example: deliberately reverse PPO's advantage signal."""
from stable_baselines3 import PPO


class SillyPPO(PPO):
    def train(self):
        advantages_before = self.rollout_buffer.advantages.copy()
        self.rollout_buffer.advantages *= 0
        self.logger.record("debug/advantage_mean_before", float(advantages_before.mean()))
        self.logger.record("debug/advantage_mean_after", float(self.rollout_buffer.advantages.mean()))
        super().train()
