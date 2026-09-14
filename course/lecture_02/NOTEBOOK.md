# Lecture 02 notebook

Open `lecture_02.ipynb` using the repository's Python environment:

~~~bash
source .venv/bin/activate
python -m jupyterlab course/lecture_02/lecture_02.ipynb
~~~

In VS Code, select the repository's `.venv` with **Select Kernel**.
On Windows, activate `.venv\Scripts\Activate.ps1`; use a Bash-capable terminal
for the shell scripts.

## Teaching workflow

1. Register for W&B and grant write access to the demo project
   [RL2_2026/Lecture_02_demo](https://wandb.ai/RL2_2026/Lecture_02_demo)
   and homework project [RL2_2026/RL2_2026](https://wandb.ai/RL2_2026/RL2_2026).
2. Discuss lecture one's homework in its original notebook.
3. Trace Zoo's training setup, callbacks, and metric logging together in Section 2.
4. Execute the configured demo runs and inspect TensorBoard and W&B in the browser.
5. Build the discounting modification from last week's PPO loss.
6. Use the silly advantage-sign modification to connect breakpoints to automated tests.
7. Assign the local discounting harness and its tests.

No precomputed results are needed. The tracking demo currently requests one
million transitions for each of two seeds. The worked debugging example requests
only 256 transitions. Execute the final homework cells only after completing the TODOs.

The TensorBoard cell starts a background server and displays a browser link.
Alternatively, from the repository root:

~~~bash
tensorboard --logdir logs/lecture_02/demo/runs --port 6006
~~~

## Local code and outputs

| File | Role |
|---|---|
| `scripts/train.py` | Register `ppo_silly` and `ppo_homework`, then call Zoo's native CLI |
| `scripts/silly_ppo.py` | Complete worked modification |
| `scripts/discounted_ppo.py` | Student PPO/buffer subclasses and TODOs |
| `scripts/SB3_LICENSE` | License for the adapted SB3 method |
| `scripts/test_silly_ppo.py` | Passing worked-example and baseline checks |
| `scripts/test_discounting.py` | Homework checks, to run after implementation |

Both local algorithms use `--conf-file hyperparams/ppo.yml` for the usual PPO defaults.
The entry point changes class selection; Zoo still creates environments and callbacks,
trains, tracks, and saves models. Only the necessary PPO update is copied into the
homework class; other behavior is inherited. Edit course files and use installed SB3 as a reference.

~~~text
logs/lecture_02/demo/          ordinary Zoo tracking runs
logs/lecture_02/debug/         short debugger runs
logs/lecture_02/homework/      student experiments
logs/lecture_02/tests/         test outputs
~~~

Under each experiment root, Zoo creates `<algorithm>/LunarLander-v3_<id>/`
for models, evaluations, Monitor files, and configuration.
Tracked shell runs use that root as their working directory, keeping
`runs/` (TensorBoard) and `wandb/` there too.

The final two code cells run the same 18 homework configurations: one saves results locally,
the other also tracks in W&B. Choose one; each uses one million transitions per run.
Their outputs go to `logs/lecture_02/homework/local/` or `logs/lecture_02/homework/tracked/`.
All supporting Python files, shell scripts, and tests are in `scripts/`.
Source links use the Python 3.12 macOS/Linux `.venv` layout; Windows uses
`.venv/Lib/site-packages/`. Each shell/debug launch loads the latest course files.
Save or close notebooks before external edits; clear outputs before committing.
