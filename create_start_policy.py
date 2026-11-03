import json

import gymnasium as gym
import torch as th
from stable_baselines3 import PPO

# Important:
# importing this usually triggers the registration of your custom environments
import rl_zoo3.custom_envs


def load_sparse_ppo(model, json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    state_dict = model.policy.state_dict()

    for name, spec in data["tensors"].items():

        if name not in state_dict:
            raise ValueError(
                f"Unknown SB3 parameter: {name}"
            )

        expected_shape = tuple(state_dict[name].shape)
        json_shape = tuple(spec["shape"])

        if expected_shape != json_shape:
            raise ValueError(
                f"Shape mismatch for {name}: "
                f"SB3={expected_shape}, JSON={json_shape}"
            )

        tensor = th.full_like(
            state_dict[name],
            float(spec.get("default", 0.0)),
        )

        for entry in spec.get("entries", []):
            *indices, value = entry
            tensor[tuple(indices)] = float(value)

        state_dict[name] = tensor

        print(f"Loaded {name}")

    model.policy.load_state_dict(
        state_dict,
        strict=True,
    )


# ----------------------------------------------------------------------
# Create environment
# ----------------------------------------------------------------------

env = gym.make("SupplyChainNonS-v6")


# ----------------------------------------------------------------------
# Create PPO
# ----------------------------------------------------------------------

policy_kwargs = dict(
    activation_fn=th.nn.Tanh,
    net_arch=dict(
        pi=[64, 64],
        vf=[64, 64],
    ),
)

model = PPO(
    "MultiInputPolicy",
    env,
    policy_kwargs=policy_kwargs,
    verbose=1,
)


# ----------------------------------------------------------------------
# Load handcrafted policy
# ----------------------------------------------------------------------

load_sparse_ppo(
    model,
    "start_weights.json",
)


# ----------------------------------------------------------------------
# Test / inspect policy
# ----------------------------------------------------------------------

obs, info = env.reset()

action, _ = model.predict(
    obs,
    deterministic=True,
)

print("\nInitial observation:")
print(obs)

print("\nAction produced by handcrafted policy:")
print(action)


# ----------------------------------------------------------------------
# Save as normal SB3 model
# ----------------------------------------------------------------------

model.save("start_policy")

print("\nSaved model as start_policy.zip")

env.close()