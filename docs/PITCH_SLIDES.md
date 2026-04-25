# GreenMARL-Coder Pitch Deck (2 min)

## Slide 1 — Problem
- LLM coding agents waste compute on repeated, low-uncertainty tasks.
- Static single-agent decoding misses coordination signals from planner/debugger roles.
- We need an environment that teaches **quality + efficiency**, not only pass/fail.

## Slide 2 — Environment Innovation
- `CodingGym` evaluates generated Python with **visible + hidden tests**.
- Rewards combine pass-rate, hidden robustness, time pressure, and sleep penalties.
- Multi-agent loop: `Planner -> Coder -> Debugger` with explicit feedback pathways.

## Slide 3 — Method
- **MARLIN-style switching** (`G_ADS`/`G_IAN`) for adaptive control flow.
- **ETD gate**: sleep when entropy is low and reward history is stable.
- **BPTA-inspired feedback**: debugger deltas injected into future planner/coder prompts.

## Slide 4 — Training Pipeline
- OpenEnv client: `CodingGymToolEnv` (`submit_python_solution` tool).
- TRL training: `training/grpo_coding_gym.py` + `GRPOTrainer`.
- Remote GPU execution: `training/hf_gpu_pipeline.sh` via HF Jobs.

## Slide 5 — Evidence
- Episode metrics: reward, pass rate, hidden pass, inference count, sleep count.
- Plot artifact: `assets/figures/reward_and_pass_by_episode.png`.
- GRPO config and logs persisted under `logs/grpo_coding/`.

## Slide 6 — Why This Matters
- Demonstrates compute-aware coding agents with measurable behavior changes.
- Provides a reproducible OpenEnv benchmark and training route for judges.
- Practical base for dual-gated ETD (entropy + epistemic divergence) extension.
