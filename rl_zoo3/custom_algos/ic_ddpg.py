from rl_zoo3.custom_algos.ddpg_target_interval import DDPGWithTargetInterval
from rl_zoo3.custom_algos.ic_td3 import ICTD3


class ICDDPG(ICTD3, DDPGWithTargetInterval):
    """SB3 DDPG with geometrically increasing target update intervals.

    :param initial_target_update_interval: Initial interval in critic gradient
        steps. Defaults to one.
    :param geometric_target_update_intervals: Grow the interval in cycle ``n`` to
        ``ceil(initial_target_update_interval * gamma ** (-2 * n / 3))``.
        When disabled, use the initial interval throughout training.
    """
