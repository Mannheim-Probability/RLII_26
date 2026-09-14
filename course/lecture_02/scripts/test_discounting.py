"""Checks for the local homework harness; run after completing its TODOs."""

import gymnasium as gym
import numpy as np
import pytest
import torch
from course.lecture_02.scripts.discounted_ppo import DiscountedPPO as PPO
from course.lecture_02.scripts.discounted_ppo import DiscountedRolloutBuffer as RolloutBuffer
from stable_baselines3.common.logger import configure
from types import SimpleNamespace


@pytest.fixture(scope="module", autouse=True)
def one_torch_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def new_buffer(discounting="loss_weighted", gamma=0.9):
    return RolloutBuffer(
        3, gym.spaces.Box(-1000, 1000, (1,), dtype=np.float32),
        gym.spaces.Discrete(2), device="cpu", n_envs=2,
        gamma=gamma, gae_lambda=0.95, discounting=discounting,
    )


def push(buffer, starts, step):
    buffer.add(
        np.array([[step], [100 + step]], dtype=np.float32),
        np.array([[0], [1]]), np.array([1.0, 2.0]), np.asarray(starts),
        torch.tensor([0.25, 0.5]), torch.tensor([-0.7, -0.7]),
    )


def filled_buffer(discounting="loss_weighted", gamma=0.9):
    buffer = new_buffer(discounting, gamma)
    for step, starts in enumerate(([1, 1], [0, 0], [0, 1])):
        push(buffer, starts, step)
    return buffer


def test_baseline_harness_still_trains():
    model = PPO("MlpPolicy", "CartPole-v1", n_steps=16, batch_size=8, n_epochs=1, seed=0, device="cpu")
    try:
        model.learn(32)
        assert model.num_timesteps == 32
    finally:
        model.get_env().close()


def test_episode_weights_survive_rollouts():
    buffer = filled_buffer()
    np.testing.assert_allclose(buffer.discount_weights, 0.9 ** np.array([[0, 0], [1, 1], [2, 0]]))
    buffer.reset()
    for step, starts in enumerate(([0, 0], [1, 0], [0, 0])):
        push(buffer, starts, step)
    np.testing.assert_allclose(buffer.discount_weights, 0.9 ** np.array([[3, 1], [0, 2], [1, 3]]))


@pytest.mark.parametrize("discounting", ["none", "loss_weighted", "weighted_sampling"])
def test_gae_targets_are_unchanged(discounting):
    buffer = filled_buffer(discounting)
    rewards = buffer.rewards.copy()
    np.testing.assert_array_equal(rewards, np.tile([1.0, 2.0], (3, 1)))
    values = buffer.values.copy()
    expected = np.empty_like(rewards)
    advantage = np.zeros(2)
    for step in reversed(range(3)):
        next_values = np.array([0.3, 0.6]) if step == 2 else values[step + 1]
        nonterminal = np.array([1.0, 0.0]) if step == 2 else 1 - buffer.episode_starts[step + 1]
        delta = rewards[step] + 0.9 * next_values * nonterminal - values[step]
        advantage = delta + 0.9 * 0.95 * nonterminal * advantage
        expected[step] = advantage
    buffer.compute_returns_and_advantage(torch.tensor([0.3, 0.6]), np.array([False, True]))
    np.testing.assert_array_equal(buffer.rewards, rewards)
    np.testing.assert_allclose(buffer.advantages, expected, atol=1e-6)
    np.testing.assert_allclose(buffer.returns, expected + values, atol=1e-6)


def test_flattening_and_uniform_permutations():
    buffer = filled_buffer()
    ids = buffer.observations.reshape(-1)
    weights = buffer.discount_weights.reshape(-1)
    expected = dict(zip(ids, 6 * weights / weights.sum()))
    for _ in range(3):
        seen = []
        for batch in buffer.get(batch_size=2):
            assert batch.weights.shape == (2,)
            for observation, weight in zip(batch.observations[:, 0].numpy(), batch.weights.numpy()):
                assert weight == pytest.approx(expected[observation])
                seen.append(observation)
        assert sorted(seen) == sorted(expected)


@pytest.mark.parametrize("discounting", ["none", "loss_weighted", "weighted_sampling"])
def test_gamma_one_has_unit_minibatch_weights(discounting):
    buffer = filled_buffer(discounting, gamma=1.0)
    np.testing.assert_allclose(buffer.discount_weights, np.ones((3, 2)))
    torch.testing.assert_close(next(buffer.get()).weights, torch.ones(6))


@pytest.mark.parametrize("gamma", [0.9, 1.0])
def test_weighted_sampling_distribution(gamma):
    buffer = filled_buffer("weighted_sampling", gamma)
    ids = buffer.observations.reshape(-1).copy()
    probabilities = buffer.discount_weights.reshape(-1).copy()
    probabilities /= probabilities.sum()
    np.random.seed(17)
    sampled = []
    duplicate_epochs = 0
    for _ in range(2000):
        batch = next(buffer.get())
        torch.testing.assert_close(batch.weights, torch.ones(6))
        observations = batch.observations[:, 0].numpy()
        assert len(observations) == 6
        duplicate_epochs += len(np.unique(observations)) < 6
        sampled.extend(observations)
    frequencies = np.array([(np.asarray(sampled) == identifier).mean() for identifier in ids])
    np.testing.assert_allclose(frequencies, probabilities, atol=0.02)
    assert duplicate_epochs > 0, "Weighted sampling must draw with replacement."


@pytest.mark.parametrize("discounting", ["none", "loss_weighted", "weighted_sampling"])
def test_ppo_loss_gradients(discounting, tmp_path, monkeypatch):
    model = PPO(
        "MlpPolicy", "CartPole-v1", discounting=discounting,
        n_steps=4, batch_size=2, n_epochs=1, normalize_advantage=False,
        ent_coef=0.1, vf_coef=0.5, max_grad_norm=1e6, seed=7, device="cpu",
    )
    model.set_logger(configure(str(tmp_path), []))
    observations = torch.tensor([[0.2, -0.7, 0.4, 0.9], [-0.3, 0.5, -0.8, 0.1]])
    actions = torch.tensor([0, 1])
    with torch.no_grad():
        model.policy.action_net.weight.mul_(20)
        old_values, old_log_prob, _ = model.policy.evaluate_actions(observations, actions)
    batch = SimpleNamespace(
        observations=observations, actions=actions,
        old_values=old_values.flatten().detach(), old_log_prob=old_log_prob.detach(),
        advantages=torch.tensor([1.0, -1.5]), returns=torch.tensor([0.3, -0.6]),
        # These two transitions are only part of a larger rollout: do not renormalize them.
        weights=torch.tensor([0.2, 0.3]) if discounting == "loss_weighted" else torch.ones(2),
    )
    monkeypatch.setattr(model.rollout_buffer, "get", lambda batch_size: iter([batch]))
    parameters = list(model.policy.parameters())
    checked = []

    def check_gradients():
        actual = [parameter.grad.clone() for parameter in parameters]
        values, log_prob, entropy = model.policy.evaluate_actions(observations, actions)
        ratio = torch.exp(log_prob - batch.old_log_prob)
        policy_losses = -torch.minimum(
            ratio * batch.advantages, torch.clamp(ratio, 0.8, 1.2) * batch.advantages,
        )
        value_losses = (values.flatten() - batch.returns).square()
        per_transition = policy_losses + 0.5 * value_losses - 0.1 * entropy
        expected_loss = (batch.weights * per_transition).mean()
        expected = torch.autograd.grad(expected_loss, parameters)
        for gradient, target in zip(actual, expected):
            torch.testing.assert_close(gradient, target, rtol=2e-4, atol=2e-6)
        checked.append(True)

    monkeypatch.setattr(model.policy.optimizer, "step", check_gradients)
    try:
        model.train()
        assert checked == [True]
    finally:
        model.get_env().close()


@pytest.mark.parametrize("discounting", ["loss_weighted", "weighted_sampling"])
def test_short_training_and_save_load(discounting, tmp_path):
    model = PPO(
        "MlpPolicy", "CartPole-v1", discounting=discounting,
        n_steps=16, batch_size=8, n_epochs=1, normalize_advantage=False,
        gamma=0.99, seed=3, device="cpu",
    )
    try:
        model.learn(32)
        assert np.isfinite(model.rollout_buffer.advantages).all()
        observations = np.array([[0, 0, 0, 0], [0.1, 0.2, -0.1, 0.1]], dtype=np.float32)
        expected, _ = model.predict(observations, deterministic=True)
        model.save(tmp_path / "model")
        loaded = PPO.load(tmp_path / "model", device="cpu")
        actual, _ = loaded.predict(observations, deterministic=True)
        np.testing.assert_array_equal(actual, expected)
        assert loaded.discounting == discounting
    finally:
        model.get_env().close()
