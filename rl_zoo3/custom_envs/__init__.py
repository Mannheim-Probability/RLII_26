import gymnasium as gym

from rl_zoo3.custom_envs.supply_chain import SupplyChainEnv
from rl_zoo3.custom_envs.supply_chain_nons4 import SupplyChainNonS4Env
from rl_zoo3.custom_envs.supply_chain_nons5 import SupplyChainNonS5Env #enl state mit tricks
from rl_zoo3.custom_envs.supply_chain_nons6 import SupplyChainNonS6Env #fast
from rl_zoo3.custom_envs.supply_chain_nons7 import SupplyChainNonS7Env #andere history, mit probabilistic forecast
from rl_zoo3.custom_envs.supply_chain_exo import SupplyChainExOEnv
from rl_zoo3.custom_envs.supply_chain_exo2 import SupplyChainExO2Env






if "SupplyChain-v0" not in gym.envs.registry:
    gym.register(
        id="SupplyChain-v0",
        entry_point=SupplyChainEnv,
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