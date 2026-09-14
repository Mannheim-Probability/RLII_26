"""Register the lecture's local subclasses, then use Zoo's normal CLI."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rl_zoo3.train import train
from rl_zoo3.utils import ALGOS
from course.lecture_02.scripts.silly_ppo import SillyPPO
from course.lecture_02.scripts.discounted_ppo import DiscountedPPO


if __name__ == "__main__":
    ALGOS.update(ppo_silly=SillyPPO, ppo_homework=DiscountedPPO)
    train()
