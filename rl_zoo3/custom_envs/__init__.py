import gymnasium as gym

from rl_zoo3.custom_envs.supply_chain import SupplyChainEnv
from rl_zoo3.custom_envs.supply_chain_nons import SupplyChainNonSEnv #kleiner state
from rl_zoo3.custom_envs.supply_chain_nons2 import SupplyChainNonS2Env #order history
from rl_zoo3.custom_envs.supply_chain_nons3 import SupplyChainNonS3Env #ein/zwei tricks
from rl_zoo3.custom_envs.supply_chain_nons4 import SupplyChainNonS4Env
from rl_zoo3.custom_envs.supply_chain_nons5 import SupplyChainNonS5Env #enl state mit tricks
from rl_zoo3.custom_envs.supply_chain_nons6 import SupplyChainNonS6Env #fast
from rl_zoo3.custom_envs.supply_chain_nons7 import SupplyChainNonS7Env #andere history, mit probabilistic forecast
from rl_zoo3.custom_envs.supply_chain_nons8 import SupplyChainNonS8Env
from rl_zoo3.custom_envs.supply_chain_nons_da9 import SupplyChainNonSDA9Env
from rl_zoo3.custom_envs.supply_chain_exo import SupplyChainExOEnv
from rl_zoo3.custom_envs.supply_chain_exo2 import SupplyChainExO2Env
from rl_zoo3.custom_envs.supply_chain_nons_lt import SupplyChainNonSLtEnv
from rl_zoo3.custom_envs.supply_chain_satisfied_d import SupplyChainSatisfiedDEnv
from rl_zoo3.custom_envs.supply_chain_satisfied_d2 import SupplyChainSatisfiedD2Env
from rl_zoo3.custom_envs.supply_chain_satisfied_d3 import SupplyChainSatisfiedD3Env








if "SupplyChain-v0" not in gym.envs.registry:
    gym.register(
        id="SupplyChain-v0",
        entry_point=SupplyChainEnv,
    )

if "SupplyChainNonS-v0" not in gym.envs.registry:
    gym.register(
        id="SupplyChainNonS-v0",
        entry_point=SupplyChainNonSEnv,
    )

if "SupplyChain-v1" not in gym.envs.registry:
    gym.register(
        id="SupplyChain-v1",
        entry_point=SupplyChainExOEnv,
    )

if "SupplyChain-v2" not in gym.envs.registry:
    gym.register(
        id="SupplyChain-v2",
        entry_point=SupplyChainExO2Env,
    )

if "SupplyChainNonS-v1" not in gym.envs.registry:
    gym.register(
        id="SupplyChainNonS-v1",
        entry_point=SupplyChainNonS2Env,
    )

if "SupplyChainNonS-v2" not in gym.envs.registry:
    gym.register(
        id="SupplyChainNonS-v2",
        entry_point=SupplyChainNonS3Env,
    )

if "SupplyChainNonS-v3" not in gym.envs.registry:
    gym.register(
        id="SupplyChainNonS-v3",
        entry_point=SupplyChainNonS4Env,
    )

if "SupplyChainNonS-v4" not in gym.envs.registry:
    gym.register(
        id="SupplyChainNonS-v4",
        entry_point=SupplyChainNonS5Env,
    )

if "SupplyChainNonS-v6" not in gym.envs.registry:
    gym.register(
        id="SupplyChainNonS-v6",
        entry_point=SupplyChainNonS6Env,
    )

if "SupplyChainNonS-v7" not in gym.envs.registry:
    gym.register(
        id="SupplyChainNonS-v7",
        entry_point=SupplyChainNonS7Env,
    )

if "SupplyChainNonS-v8" not in gym.envs.registry:
    gym.register(
        id="SupplyChainNonS-v8",
        entry_point=SupplyChainNonS8Env,
    )

if "SupplyChainNonS_DA-v9" not in gym.envs.registry:
    gym.register(
        id="SupplyChainNonS_DA-v9",
        entry_point=SupplyChainNonSDA9Env,
    )

if "SupplyChainNonSLt-v0" not in gym.envs.registry:
    gym.register(
        id="SupplyChainNonSLt-v0",
        entry_point=SupplyChainNonSLtEnv,
    )

if "SupplyChainSatisfiedD-v0" not in gym.envs.registry:
    gym.register(
        id="SupplyChainSatisfiedD-v0",
        entry_point=SupplyChainSatisfiedDEnv,
    )

if "SupplyChainSatisfiedD-v2" not in gym.envs.registry:
    gym.register(
        id="SupplyChainSatisfiedD-v2",
        entry_point=SupplyChainSatisfiedD2Env,
    )

if "SupplyChainSatisfiedD-v3" not in gym.envs.registry:
    gym.register(
        id="SupplyChainSatisfiedD-v3",
        entry_point=SupplyChainSatisfiedD3Env,
    )