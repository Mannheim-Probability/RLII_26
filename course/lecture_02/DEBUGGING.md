# Debugging a PPO modification

Open [RLII_26.code-workspace](../../RLII_26.code-workspace).
Lecture two uses local subclasses; installed SB3 remains the reference implementation.

## Worked example: reverse the advantages

Select **Lecture 02: Silly PPO (debug)** and set breakpoints in
[scripts/silly_ppo.py](scripts/silly_ppo.py):

| Breakpoint | Inspect |
|---|---|
| `self.rollout_buffer.advantages *= -1` | Before stepping: `advantages_before`; after F10: every sign reverses |
| `super().train()` | F11 enters SB3; inspect `rollout_data.advantages` during the policy update |

F5 starts/continues, F10 steps over a statement, and F11 enters a function.
A breakpoint stops before the highlighted statement executes.

`program` selects the course entry point; `args` are native Zoo flags.
`--algo ppo_silly` selects the local class. `cwd` and `python` select the
repository and environment. `justMyCode: false` permits stepping into SB3.
`stopOnEntry: true` pauses at startup.

Run the worked example's tests:

~~~bash
python -m pytest course/lecture_02/scripts/test_silly_ppo.py -q \
  --basetemp=logs/lecture_02/tests/silly
~~~

Changing the multiplier to -2 must fail the sign-reversal test. Restore -1 afterwards.

## Homework: inspect a small, predictable case

Select **Lecture 02: Discounted PPO (debug)**. Its baseline mode runs immediately.
Change the `discounting` value in `args` to test an implemented variant.

| Location in scripts/discounted_ppo.py | Inspect |
|---|---|
| `add()` | `episode_start`, `self.pos`, `self.episode_steps`, stored weights |
| `get()` | Index order, full-rollout weight sum, repeated draws |
| `_get_samples()` | Observations and their corresponding weights |
| `train()` | Per-transition losses, minibatch weights, final scalar loss |

Select **Lecture 02: Discounting tests (debug)** to reproduce a failing test
with breakpoints in the local harness. Its small inputs make resets, weight
alignment, and gradient errors easier to inspect than a full training run.

All debug outputs go under `logs/lecture_02/debug/`. Launch configurations
remain in the workspace file; VS Code stores breakpoints separately.
