# Hackathon / judge checklist (GreenMARL-Coder + OpenEnv)

This document maps the project to the **OpenEnv + training + presentation** bar and what is implemented where.

## Environment (40% — innovation & challenge)

- **Novelty**: A multi-agent *coding* loop (planner, coder, debugger) with **MARLIN-style** generator switching, **ETD** sleep when the policy is confident, and **BPTA-style** feedback injection — not a grid-world clone.
- **Code**: `env/coding_gym.py` (rubric-style rewards, visible + hidden tests), `env/tasks.json` task pack.
- **OpenEnv surface**: `openenv.yaml` + `greenmarl_openenv/coding_trl_env.py` (`CodingGymToolEnv` with `submit_python_solution`; reserved names `reset`/`step`/`state`/`close` are **not** used as tool names).
- **HF Space (demo)**: linked from the main `README.md` (Gradio `app.py` for interactive runs).

## Storytelling (30%)

- **README** — problem, agents, how rewards work, how to run modes, where plots live.
- **Slide deck writeup** — `docs/PITCH_SLIDES.md` (linked from README).

## Improvement in rewards (20%)

- **Quantitative**
  - `python main.py --compare --episodes N --save-metrics` → `logs/metrics_*.json`.
  - `python training/plot_run_metrics.py --input logs/metrics_full.json` → `assets/figures/reward_and_pass_by_episode.png` (commit the PNG for reviewers).
- **GRPO (TRL)**: `training/grpo_coding_gym.py` with `--smoke` (CI) or full steps (GPU). Save TRL/terminal logs; optional W&B link if you use it.
- **Interpretation**
  - **ETD = 0%** on short, cold-start runs is common: sleep requires *low entropy*, *stable reward history*, and a *reusable prior solution* — new tasks per episode break that.
  - **Compare mode** can show **worse** “full” than a single lucky `--full` run: stochastic LLM, ordering, and first-episode cold start. Use more episodes, fixed `--task` / task id if available, or lower temperature in agent settings for a stable demo.

## Reward & training pipeline (10%)

- **Main loop** (`core/bpta_coordinator.py`, `main.py`): test-based + shaped rewards; BPTA delta from debugger; ETD gating in coder.
- **TRL** (`training/grpo_coding_gym.py`): `GRPOTrainer` + `environment_factory=CodingGymToolEnv`; reward from `env.reward` after tool execution.

## Research alignment (for writeups, not a judging column)

- **BPTA / BPPO** (arXiv): backward peer feedback; this repo uses **in-context** propagation + a stub `StateAdapter`, not full Equation-6 style BPPO in PyTorch.
- **Dual-gated ETD** (entropy + epistemic / twin-critic in papers): the implementation tracks **Shannon entropy** and reward stability; **twin V₁/V₂** epistemic divergence is a natural extension to reach paper-style “sleep when both gates say confident.”
- **MARLIN**: planner-driven switching between `G_ADS` and `G_IAN` in `agents/planner.py`.

## Technical requirements (table stakes)

| Requirement        | Status |
|--------------------|--------|
| OpenEnv base pattern | `CodingGymToolEnv` + `openenv.yaml` |
| Client/server hygiene | No server imports in client code paths |
| Training script     | `training/grpo_coding_gym.py` (TRL) |
| Plots in repo      | From `plot_run_metrics.py` (after real `--save-metrics` run) |
| README links       | Space, training, Colab, and slide deck link |
| No huge videos in Space | Use hosted URLs only |

## Colab

Use `training/Colab_GRPO_CodingGym.ipynb` (GPU runtime): install `requirements.txt` + `requirements-train.txt`, set `USE_TORCH=1` / `USE_TF=0`, then run a smoke or short training.

## HF GPU-only execution

- Script: `training/hf_gpu_pipeline.sh`
- Launch (example): `hf jobs run --namespace YOUR_USER --flavor a10g-small --timeout 4h --detach --secrets HF_TOKEN python:3.10 bash -lc "<clone repo && git checkout <commit> && bash training/hf_gpu_pipeline.sh>"`
- Artifacts are uploaded to the Space repo under `artifacts/`.
- After a successful run, link the `artifacts/*` files from the README (see “Latest HF GPU run” section).
