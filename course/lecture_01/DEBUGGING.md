# Debugging PPO on LunarLander in VS Code

The assignment uses `scripts/train_ppo_example.sh`. The workspace provides a
separate short LunarLander debugging example: 10,000 requested transitions, one
environment, 1,024 rollout steps, minibatches of 64 and two epochs. Five evaluation
episodes run every 1,024 transitions. Logs go to `logs/lecture_01/debug_ppo/`.
These runs are excluded from the assignment plots.

## Start the workspace debugger

1. In VS Code, use **File → Open Workspace from File** and select
   [RLII_26.code-workspace](../../RLII_26.code-workspace).
2. Install the recommended **Python** and **Python Debugger** extensions if missing.
3. Open **Run and Debug** and select **Lecture 01: PPO LunarLander (debug)**.
4. Press **F5**. Execution pauses at the entry point.
5. Open a source file below, set a breakpoint by clicking its gutter, and press
   **F5** to continue.

The configuration selects `.venv/bin/python` on macOS/Linux and
`.venv/Scripts/python.exe` on Windows, sets the working directory to the repository
root, and uses `justMyCode: false`. No `PYTHONPATH` override is needed after the
editable project installation.

The previous workspace configurations remain available. Use the **Lecture 01**
configuration for this walkthrough. The assignment script and debug configuration serve separate purposes.

## Five breakpoint stations

Line numbers refer to the currently installed sources. Locate the named statement
if a package update moves it. Library paths below are relative to
`.venv/lib/python3.12/site-packages/`; on Windows use `.venv/Lib/site-packages/`.

| Station | File and line | Statement / method | Inspect |
|---|---|---|---|
| 1 | `rl_zoo3/exp_manager.py:237` (repository) | `setup_experiment()`: `model = ALGOS[...]` | `self._hyperparams`, `self.seed`, `env.num_envs`, `env.envs[0]` |
| 2 | `stable_baselines3/common/on_policy_algorithm.py:202` | `collect_rollouts()`: `actions, values, log_probs = self.policy(...)` | `obs_tensor.shape`; after stepping: `actions`, `values`, `log_probs` |
| 3 | `gymnasium/envs/box2d/lunar_lander.py:620` | `step()`: `self.world.Step(...)` | `action`, `m_power`, `s_power`; further down: `state`, `shaping`, `reward` |
| 4 | `stable_baselines3/common/buffers.py:434` | `compute_returns_and_advantage()`: `last_gae_lam = delta + ...` | `delta`, `next_non_terminal`, `self.gae_lambda`, `last_gae_lam` |
| 5 | `stable_baselines3/ppo/ppo.py:222` | `train()`: `ratio = th.exp(...)` | `advantages.shape`, `log_prob`; after stepping: `ratio`; further down: `policy_loss`, `loss` |

For a short walkthrough use stations 1, 2 and 5. Add physics and GAE when useful.
Breakpoints stop before the line executes; use **Step Over (F10)** to inspect
newly assigned values. Add expressions to **Watch** or enter them in the
**Debug Console**. **Step Into (F11)** enters a function; **Continue (F5)**
runs to the next breakpoint.

Disable a breakpoint after the first useful stop: policy/physics lines execute
many times, and GAE iterates backward over every rollout step. LunarLander calls
`step()` internally during `reset()`, so station 3 can be reached before the first
policy action. Evaluation also executes policy prediction and environment steps.

Breakpoints are stored by VS Code in your workspace state, not in the
`.code-workspace` file. Set them once in your local workspace.

## Inspect the result

After training completes:

~~~bash
python course/lecture_01/scripts/visualize_ppo_lunarlander.py \
  --run-dir logs/lecture_01/debug_ppo/ppo/LunarLander-v3_1
~~~

Use the run number printed by Zoo if it differs from 1.
