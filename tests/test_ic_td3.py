from copy import deepcopy

import pytest
import torch as th
from stable_baselines3 import TD3
from stable_baselines3.common.logger import configure
from stable_baselines3.common.type_aliases import ReplayBufferSamples

from rl_zoo3.custom_algos.ic_td3 import ICTD3
from rl_zoo3.custom_algos.td3_target_interval import TD3WithTargetInterval


@pytest.fixture
def prepare_training(monkeypatch, tmp_path):
    # Use the same nontrivial batch for both algorithms, including per-sample discounts.
    observations = th.arange(12, dtype=th.float32).reshape(4, 3) / 12
    batch = ReplayBufferSamples(
        observations=observations,
        actions=observations[:, :1],
        next_observations=observations + 0.1,
        dones=th.tensor([[0.0], [1.0], [0.0], [0.0]]),
        rewards=th.ones(4, 1),
        discounts=th.tensor([[0.9], [0.0], [0.8], [0.7]]),
    )

    def prepare(model):
        model.set_logger(configure(str(tmp_path), format_strings=[]))
        monkeypatch.setattr(model.replay_buffer, "sample", lambda *args, **kwargs: batch)

    return prepare


@pytest.mark.parametrize(
    ("algorithm", "schedule_kwargs"),
    [(TD3WithTargetInterval, {}), (ICTD3, {"geometric_target_update_intervals": False})],
)
def test_constant_interval_matches_td3(prepare_training, algorithm, schedule_kwargs):
    kwargs = dict(env="Pendulum-v1", policy_kwargs={"net_arch": [8]}, policy_delay=3, buffer_size=16, seed=0, device="cpu")
    baseline = TD3("MlpPolicy", **kwargs)
    scheduled = algorithm("MlpPolicy", **schedule_kwargs, **kwargs)
    assert scheduled.current_target_update_interval == 3

    for model in (baseline, scheduled):
        prepare_training(model)

    for gradient_steps in (2, 3, 2):
        th.manual_seed(gradient_steps)
        baseline.train(gradient_steps, batch_size=4)
        th.manual_seed(gradient_steps)
        scheduled.train(gradient_steps, batch_size=4)
        th.testing.assert_close(baseline.policy.state_dict(), scheduled.policy.state_dict(), rtol=0, atol=0)

    assert scheduled._n_target_updates == 2
    assert scheduled._next_target_update == 9


def test_independent_fixed_interval_and_logging(prepare_training):
    model = TD3WithTargetInterval(
        "MlpPolicy",
        "Pendulum-v1",
        policy_delay=2,
        target_update_interval=3,
        policy_kwargs={"net_arch": [8]},
        buffer_size=16,
        seed=0,
        device="cpu",
    )
    prepare_training(model)
    actor_steps = []
    model.actor.optimizer.register_step_post_hook(lambda *_: actor_steps.append(model._n_updates))

    for step in range(1, 8):
        model.train(1, batch_size=4)
        assert model.logger.name_to_value["train/target_update_interval"] == 3
        assert model.logger.name_to_value["train/n_target_updates"] == step // 3

    assert actor_steps == [2, 4, 6]
    assert model._next_target_update == 9


def test_geometric_intervals_and_checkpoint(tmp_path, prepare_training):
    model = ICTD3(
        "MlpPolicy",
        "Pendulum-v1",
        initial_target_update_interval=2,
        policy_delay=3,
        gamma=0.7,
        tau=1.0,
        policy_kwargs={"net_arch": [8]},
        buffer_size=16,
        seed=0,
        device="cpu",
    )
    prepare_training(model)
    actor_steps = []
    model.actor.optimizer.register_step_post_hook(lambda *_, model=model: actor_steps.append(model._n_updates))
    target_steps = {2, 5, 9, 14}

    for step in range(1, 16):
        actor_target_before = deepcopy(model.actor_target.state_dict())
        critic_target_before = deepcopy(model.critic_target.state_dict())
        model.train(1, batch_size=4)

        if step in target_steps:
            th.testing.assert_close(model.actor_target.state_dict(), model.actor.state_dict(), rtol=0, atol=0)
            th.testing.assert_close(model.critic_target.state_dict(), model.critic.state_dict(), rtol=0, atol=0)
        else:
            th.testing.assert_close(model.actor_target.state_dict(), actor_target_before, rtol=0, atol=0)
            th.testing.assert_close(model.critic_target.state_dict(), critic_target_before, rtol=0, atol=0)
        assert model._n_target_updates == len({target_step for target_step in target_steps if target_step <= step})

        if step == 4:
            checkpoint = tmp_path / "ic_td3.zip"
            model.save(checkpoint)
            model = ICTD3.load(checkpoint, env=model.get_env(), device="cpu")
            assert model._next_target_update == 5
            assert model.current_target_update_interval == 3
            prepare_training(model)
            model.actor.optimizer.register_step_post_hook(lambda *_, model=model: actor_steps.append(model._n_updates))

    assert actor_steps == [3, 6, 9, 12, 15]
    assert model._next_target_update == 20
