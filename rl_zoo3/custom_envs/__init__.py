import gymnasium as gym

from rl_zoo3.custom_envs.supply_chain import SupplyChainEnv
from rl_zoo3.custom_envs.supply_chain_ns import SupplyChainNSEnv
from rl_zoo3.custom_envs.supply_chain_exo import SupplyChainExOEnv




if "SupplyChain-v0" not in gym.envs.registry:
    gym.register(
        id="SupplyChain-v0",
        entry_point=SupplyChainEnv,
    )

if "SupplyChainNS-v0" not in gym.envs.registry:
    gym.register(
        id="SupplyChainNS-v0",
        entry_point=SupplyChainNSEnv,
    )

if "SupplyChain-v1" not in gym.envs.registry:
    gym.register(
        id="SupplyChain-v1",
        entry_point=SupplyChainExOEnv,
    )
