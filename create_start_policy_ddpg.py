import json

import gymnasium as gym
import torch as th
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np

import rl_zoo3.custom_envs


# ---------------------------------------------------------
# Create environment
# ---------------------------------------------------------

env = gym.make("SupplyChainNonS-v6")


# ---------------------------------------------------------
# Create DDPG
# ---------------------------------------------------------

policy_kwargs = dict(
    activation_fn=th.nn.Tanh,
    net_arch=dict(
        pi=[64, 64],
        qf=[64, 64],
    ),
)

# DDPG exploration is handled using action noise,
# not PPO-style log_std.

n_actions = env.action_space.shape[-1]

action_noise = NormalActionNoise(
    mean=np.zeros(n_actions),
    sigma=0.1 * np.ones(n_actions),
)

model = DDPG(
    "MultiInputPolicy",
    env,
    policy_kwargs=policy_kwargs,
    action_noise=action_noise,
    verbose=1,
)

def load_ppo_actor_into_ddpg(model, json_path):

    with open(json_path, "r") as f:
        data = json.load(f)

    tensors = data["tensors"]

    # Mapping:
    # your PPO JSON name -> DDPG actor parameter name
    mapping = {
        "mlp_extractor.policy_net.0.weight": "mu.0.weight",
        "mlp_extractor.policy_net.0.bias": "mu.0.bias",
        "mlp_extractor.policy_net.2.weight": "mu.2.weight",
        "mlp_extractor.policy_net.2.bias": "mu.2.bias",
        "action_net.weight": "mu.4.weight",
        "action_net.bias": "mu.4.bias",
    }

    actor_state = model.actor.state_dict()

    print("\nDDPG actor parameters:")
    for name, tensor in actor_state.items():
        print(name, tuple(tensor.shape))

    for ppo_name, ddpg_name in mapping.items():

        spec = tensors[ppo_name]

        if ddpg_name not in actor_state:
            raise ValueError(
                f"{ddpg_name} not found in DDPG actor."
            )

        expected_shape = tuple(actor_state[ddpg_name].shape)
        supplied_shape = tuple(spec["shape"])

        if expected_shape != supplied_shape:
            raise ValueError(
                f"Shape mismatch:\n"
                f"{ppo_name}: {supplied_shape}\n"
                f"{ddpg_name}: {expected_shape}"
            )

        # Start tensor at default value
        tensor = th.full_like(
            actor_state[ddpg_name],
            float(spec.get("default", 0.0)),
        )

        # Insert sparse entries
        for entry in spec.get("entries", []):
            *indices, value = entry
            tensor[tuple(indices)] = float(value)

        actor_state[ddpg_name] = tensor

        print(
            f"{ppo_name}"
            f"  ->  {ddpg_name}"
        )

    # Load handcrafted actor
    model.actor.load_state_dict(actor_state)

    # VERY IMPORTANT:
    # DDPG also has a target actor.
    # It should initially be identical.
    model.actor_target.load_state_dict(
        model.actor.state_dict()
    )

    print("\nHandcrafted actor loaded into DDPG.")

    load_ppo_actor_into_ddpg(
    model,
    "start_weights.json",
)


# Test
obs, info = env.reset()

action, _ = model.predict(
    obs,
    deterministic=True,
)

print("\nInitial DDPG action:")
print(action)


# Save
model.save("start_policy_ddpg")

print("\nSaved as start_policy_ddpg.zip")

env.close()