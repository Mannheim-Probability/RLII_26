from copy import deepcopy

import pytest
import torch as th
from stable_baselines3 import DDPG, TD3
from stable_baselines3.common.logger import configure
from stable_baselines3.common.type_aliases import ReplayBufferSamples

from rl_zoo3.custom_algos.ddpg_target_interval import DDPGWithTargetInterval
from rl_zoo3.custom_algos.ic_ddpg import ICDDPG
from rl_zoo3.custom_algos.ic_td3 import ICTD3
from rl_zoo3.custom_algos.ic_td3_polyak import ICTD3Polyak
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


@pytest.mark.parametrize(
    ("algorithm", "schedule_kwargs"),
    [(DDPGWithTargetInterval, {}), (ICDDPG, {"geometric_target_update_intervals": False})],
)
def test_constant_interval_matches_ddpg(prepare_training, algorithm, schedule_kwargs):
    kwargs = dict(env="Pendulum-v1", policy_kwargs={"net_arch": [8]}, buffer_size=16, seed=0, device="cpu")
    baseline = DDPG("MlpPolicy", **kwargs)
    scheduled = algorithm("MlpPolicy", **schedule_kwargs, **kwargs)
    assert scheduled.current_target_update_interval == 1
    assert scheduled.policy_delay == 1
    assert scheduled.critic.n_critics == 1
    assert scheduled.target_noise_clip == 0.0

    for model in (baseline, scheduled):
        prepare_training(model)

    th.manual_seed(0)
    baseline.train(5, batch_size=4)
    th.manual_seed(0)
    scheduled.train(5, batch_size=4)
    th.testing.assert_close(baseline.policy.state_dict(), scheduled.policy.state_dict(), rtol=0, atol=0)
    assert scheduled._n_target_updates == 5


@pytest.mark.parametrize(
    ("algorithm", "actor_kwargs"),
    [(TD3WithTargetInterval, {"policy_delay": 2}), (DDPGWithTargetInterval, {})],
)
def test_independent_fixed_interval_and_logging(prepare_training, algorithm, actor_kwargs):
    model = algorithm(
        "MlpPolicy",
        "Pendulum-v1",
        target_update_interval=3,
        policy_kwargs={"net_arch": [8]},
        buffer_size=16,
        seed=0,
        device="cpu",
        **actor_kwargs,
    )
    prepare_training(model)
    actor_steps = []
    model.actor.optimizer.register_step_post_hook(lambda *_: actor_steps.append(model._n_updates))

    for step in range(1, 8):
        model.train(1, batch_size=4)
        assert model.logger.name_to_value["train/target_update_interval"] == 3
        assert model.logger.name_to_value["train/n_target_updates"] == step // 3

    actor_delay = actor_kwargs.get("policy_delay", 1)
    assert actor_steps == list(range(actor_delay, 8, actor_delay))
    assert model._next_target_update == 9


@pytest.mark.parametrize(("algorithm", "actor_kwargs"), [(ICTD3, {"policy_delay": 3}), (ICDDPG, {})])
def test_geometric_intervals_and_checkpoint(tmp_path, prepare_training, algorithm, actor_kwargs):
    model = algorithm(
        "MlpPolicy",
        "Pendulum-v1",
        initial_target_update_interval=2,
        gamma=0.7,
        tau=1.0,
        policy_kwargs={"net_arch": [8]},
        buffer_size=16,
        seed=0,
        device="cpu",
        **actor_kwargs,
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
            checkpoint = tmp_path / "model.zip"
            model.save(checkpoint)
            model = algorithm.load(checkpoint, env=model.get_env(), device="cpu")
            assert model._next_target_update == 5
            assert model.current_target_update_interval == 3
            prepare_training(model)
            model.actor.optimizer.register_step_post_hook(lambda *_, model=model: actor_steps.append(model._n_updates))

    actor_delay = actor_kwargs.get("policy_delay", 1)
    assert actor_steps == list(range(actor_delay, 16, actor_delay))
    assert model._next_target_update == 20


def test_polyak_cycle_updates_and_checkpoint(tmp_path, prepare_training):
    model = ICTD3Polyak(
        "MlpPolicy",
        "Pendulum-v1",
        initial_tau_cycle_length=2,
        target_update_interval=1,
        gamma=0.7,
        policy_delay=3,
        policy_kwargs={"net_arch": [8]},
        buffer_size=16,
        seed=0,
        device="cpu",
    )
    assert model.tau == 2 / 3
    actor_steps = []
    prepare_training(model)
    model.actor.optimizer.register_step_post_hook(lambda *_, model=model: actor_steps.append(model._n_updates))

    step = 0
    parameter_age = 0.0
    for cycle_index, cycle_length in enumerate((2, 3, 4, 5)):
        tau = 2 / (cycle_length + 1)
        for _ in range(cycle_length):
            step += 1
            actor_target_before = deepcopy(model.actor_target.state_dict())
            critic_target_before = deepcopy(model.critic_target.state_dict())
            model.train(1, batch_size=4)

            for online, target, before in (
                (model.actor, model.actor_target, actor_target_before),
                (model.critic, model.critic_target, critic_target_before),
            ):
                expected = {name: th.lerp(before[name], parameter, tau) for name, parameter in online.state_dict().items()}
                th.testing.assert_close(target.state_dict(), expected)

            parameter_age = (1 - tau) * (parameter_age + 1)
            metrics = model.logger.name_to_value
            assert metrics["train/n_target_updates"] == step
            assert metrics["train/target_update_interval"] == 1
            assert metrics["train/tau"] == tau
            assert metrics["train/tau_cycle_index"] == cycle_index
            assert metrics["train/tau_cycle_length"] == cycle_length
            assert metrics["train/target_parameter_age"] == pytest.approx(parameter_age)

            if step == 4:
                checkpoint = tmp_path / "polyak_model.zip"
                model.save(checkpoint)
                model = ICTD3Polyak.load(checkpoint, env=model.get_env(), device="cpu")
                assert model.tau == tau
                assert model.target_parameter_age == pytest.approx(parameter_age)
                prepare_training(model)
                model.actor.optimizer.register_step_post_hook(lambda *_, model=model: actor_steps.append(model._n_updates))

    assert actor_steps == [3, 6, 9, 12]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"initial_tau_cycle_length": 0}, "initial_tau_cycle_length must be positive"),
        ({"gamma": 0}, "require 0 < gamma <= 1"),
        ({"tau": 0.005}, "tau is derived"),
        ({"target_update_interval": 2}, "targets after every critic step"),
    ],
)
def test_polyak_cycle_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ICTD3Polyak("MlpPolicy", "Pendulum-v1", _init_setup_model=False, **kwargs)
