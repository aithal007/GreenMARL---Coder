---
title: GreenMARL-Coder
emoji: 🤖
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: "4.36.0"
python_version: "3.10"
app_file: app.py
pinned: false
---

# GreenMARL-Coder

GreenMARL-Coder is a hackathon-ready multi-agent coding assistant prototype that maps three research ideas into a practical software-engineering loop:

- **MARLIN**: LLM-guided planner with adaptive generator switching (`G_ADS` / `G_IAN`)
- **ETD-MAPPO**: uncertainty-gated coder that can skip decoding via a `sleep_token`
- **BPTA**: bidirectional feedback where debugger critique is propagated back to planner/coder

The project is designed for fast iteration in a 72-hour setting: it uses local Hugging Face LLMs as policy backbones, while still preserving the algorithmic structure needed for judging.

---

## Judge / hackathon deliverables (OpenEnv + training + story)

| Deliverable | Where |
|-------------|--------|
| **Runnable Space (demo)** | [Hugging Face Space — Aithal04/metaai](https://huggingface.co/spaces/Aithal04/metaai) (Gradio `app.py`) |
| **OpenEnv manifest** | `openenv.yaml` — client class `greenmarl_openenv.coding_trl_env.CodingGymToolEnv` |
| **Tool env (Gym + TRL)** | `greenmarl_openenv/coding_trl_env.py` — tool `submit_python_solution` (not reserved names) |
| **TRL + GRPO training** | `training/grpo_coding_gym.py`; extras in `requirements-train.txt` |
| **Colab smoke** | `training/Colab_GRPO_CodingGym.ipynb` (GPU runtime) |
| **Metrics plots** | After `main.py --save-metrics`, run `python training/plot_run_metrics.py` → `assets/figures/reward_and_pass_by_episode.png` (commit PNGs for reviewers) |
| **Checklist writeup** | [docs/SUBMISSION_JUDGING.md](docs/SUBMISSION_JUDGING.md) |
| **External story** | [docs/PITCH_SLIDES.md](docs/PITCH_SLIDES.md) (short slide-deck style writeup) |

**Training on GPU:** set `USE_TORCH=1` and `USE_TF=0` before importing `transformers` / `trl` (see the top of `training/grpo_coding_gym.py`). On CPU-only, prefer `--smoke` or a tiny model.

**ETD and compare mode:** `ETD=0%` on a handful of *different* tasks is expected when the coder never sees stable, low-entropy conditions with a reusable solution. The comparison table in `--compare` can look worse for “full” than a single good `--full` run because of LLM randomness and cold-start order — use more episodes, fixed tasks, or deterministic decoding for demos.

**Research context:** MARLIN-style planner switching, entropy-based ETD, and in-context BPTA feedback are implemented in the agent loop. Full BPPO (Equation 6) and twin-critic epistemic gating from the ETD-MAPPO papers are natural extensions, not the present training loss.

### Run everything on HF GPU (no local training)

Use Hugging Face Jobs to run evaluation + plotting + GRPO remotely:

```bash
hf jobs run --flavor t4-medium --timeout 4h --detach \
  --secrets HF_TOKEN \
  python:3.10 bash -lc "
    git clone https://github.com/aithal007/GreenMARL---Coder.git && \
    cd GreenMARL---Coder && \
    bash training/hf_gpu_pipeline.sh
  "
```

The pipeline uploads reviewer artifacts to the Space repo under `artifacts/`:
- `artifacts/metrics_full.json`
- `artifacts/reward_and_pass_by_episode.png`
- `artifacts/hf_gpu_run_summary.txt`
- `artifacts/grpo_config_used.json`

### Latest HF GPU run (evidence on Hub)

Open these directly (no clone required):

- [Reward / pass plot (PNG)](https://huggingface.co/spaces/Aithal04/metaai/resolve/main/artifacts/reward_and_pass_by_episode.png)
- [Episode metrics (JSON)](https://huggingface.co/spaces/Aithal04/metaai/resolve/main/artifacts/metrics_full.json)
- [Run summary (TXT)](https://huggingface.co/spaces/Aithal04/metaai/resolve/main/artifacts/hf_gpu_run_summary.txt)
- [GRPO config used (JSON)](https://huggingface.co/spaces/Aithal04/metaai/resolve/main/artifacts/grpo_config_used.json)

From the last successful job: **20-episode full eval** averaged about **0.69 reward**, **50% visible pass**, **50% hidden pass** (see `hf_gpu_run_summary.txt`). **GRPO** completed **40 steps** with reported **`train_loss` ≈ 0.0027**; logs show many steps hitting `max_completion_length=1024` with `tools/call_frequency` often **0**, then occasional tool calls and non-zero rewards — that is normal for a short demo run and is still valid “pipeline ran on real env” evidence.

![Reward and pass rate by episode (HF Job output)](https://huggingface.co/spaces/Aithal04/metaai/resolve/main/artifacts/reward_and_pass_by_episode.png)

---

## Why this project

Full multi-agent PPO training can be too slow and fragile for short hackathons. GreenMARL-Coder keeps the **agent architecture and learning signals** while replacing expensive end-to-end RL training with:

1. **LLM policy inference with logprobs** for uncertainty estimation
2. **In-context BPTA** for practical feedback propagation
3. **Stub PyTorch state adapter** to demonstrate a differentiable pathway concept

This gives a working prototype you can demo, benchmark, and extend later.

---

## System architecture

### Agents

- `PlannerAgent` (`agents/planner.py`)
  - Implements MARLIN-style generator selection logic
  - Produces structured plan JSON for each task
  - Maintains past performance buffer and mode switching

- `CoderAgent` (`agents/coder.py`)
  - Generates Python solutions guided by planner output
  - Uses ETD gating: if entropy is low and rewards are stable, returns `sleep_token`
  - Reuses prior solution to reduce inference calls

- `DebuggerAgent` (`agents/debugger.py`)
  - Critiques code using sandbox test feedback
  - Produces value estimate, reward shaping delta, and structured BPTA feedback

### Environment

- `CodingGym` (`env/coding_gym.py`)
  - Executes generated code in isolated subprocess
  - Runs visible + hidden tests from `env/tasks.json`
  - Returns reward, pass rates, timing, and diagnostics

### Coordinator

- `BPTACoordinator` (`core/bpta_coordinator.py`)
  - Orchestrates Planner -> Coder -> Gym -> Debugger
  - Injects debugger feedback into future prompts (in-context BPTA)
  - Includes a small PyTorch `StateAdapter` to model latent feedback flow

---

## Repository layout

```text
meta-ai/
├─ agents/          (planner, coder, debugger, MARLIN/ETD/BPTA hooks)
├─ core/            bpta_coordinator.py
├─ env/             coding_gym.py, tasks.json
├─ greenmarl_openenv/   OpenEnv client + TRL tool wrapper (CodingGymToolEnv)
├─ training/        grpo_coding_gym.py, Colab_GRPO_CodingGym.ipynb, plot_run_metrics.py
├─ tests/
├─ openenv.yaml
├─ app.py           Gradio Space entrypoint
├─ main.py
├─ requirements.txt
└─ requirements-train.txt
```

---

## Installation

### 1) Clone

```bash
git clone https://github.com/aithal007/GreenMARL---Coder.git
cd GreenMARL---Coder
```

### 2) Create environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

> If you want GPU acceleration, install the matching CUDA build of PyTorch first.

---

## Quick start

### Baseline (single agent)

```bash
python main.py --baseline --episodes 5
```

### Multi-agent ablation (no ETD/BPTA)

```bash
python main.py --multi-agent --episodes 5
```

### Full GreenMARL-Coder

```bash
python main.py --full --episodes 5 --steps 1
```

### Compare all modes

```bash
python main.py --compare --episodes 5 --save-metrics
```

---

## CLI options

```bash
python main.py --help
```

Key arguments:

- `--baseline` | `--multi-agent` | `--full` | `--compare`
- `--model` (default: `Qwen/Qwen2.5-Coder-1.5B-Instruct`)
- `--device` (`cpu`, `cuda`, or `mps`)
- `--episodes` (default: `5`)
- `--steps` (default: `1`)
- `--save-metrics` (writes metrics JSON under `logs/`)

---

## Metrics and logs

### Metrics captured

- Episode reward and shaped reward
- Visible pass rate and hidden pass rate
- Inference count and ETD sleep count
- Time-to-solution
- Planner generator mode

### Log files

- `logs/run.log` - runtime logs
- `logs/agent_chat.txt` - agent-by-agent interaction trace
- `logs/metrics_*.json` - saved metrics snapshots

---

## ETD gating details

The coder probes the next-token distribution for the first `K` generation steps and computes Shannon entropy:

\[
H = -\sum_i p_i \log_2 p_i
\]

If:

- `entropy < ENTROPY_THRESHOLD`
- reward history is stable (low variance, positive mean)
- previous solution exists

then the coder emits `sleep_token` and reuses earlier code.

This reduces unnecessary inference while preserving performance on repeated/stable tasks.

---

## BPTA implementation strategy

GreenMARL-Coder uses a hybrid approach:

1. **In-context backprop**
   - Debugger emits structured `bpta_delta`
   - Coordinator injects this into coder/planner context for subsequent steps

2. **Stub differentiable adapter**
   - `StateAdapter` MLP consumes compact state vector
   - Demonstrates latent feedback pathway (conceptual stand-in for full differentiable pipeline)

This preserves demo simplicity while staying faithful to BPTA’s intent.

---

## Test suite

Run all tests:

```bash
python -m pytest tests/ -v
```

Current test coverage includes:

- Coding gym task loading, sandbox execution, syntax errors, reward clamping
- ETD gating behavior (sleep vs act) under entropy/reward conditions
- BPTA prompt delta injection behavior

---

## Expected demo flow (hackathon)

1. Start with `--baseline` to establish compute and pass-rate floor
2. Run `--multi-agent` to show planner/debugger coordination benefit
3. Run `--full` and verify:
   - lower inference count (ETD savings)
   - improved or stable pass rate (BPTA feedback)
4. Show `logs/agent_chat.txt` as qualitative evidence of agent negotiation

---

## Known limitations

- Uses language-model prompting rather than full PPO optimization
- Sandbox is subprocess-based, not hardened container isolation
- BPTA differentiable path is currently conceptual (stub adapter)
- Performance depends on model choice and local hardware

---

## Roadmap

- Replace stub adapter with trainable prompt/embedding optimizer
- Add secure containerized execution for coding gym
- Extend task set with multi-file and debugging-heavy challenges
- Add experiment tracking dashboards for richer benchmark reporting

---

## Acknowledgments

This repository was built for a multi-agent coding assistant hackathon theme and explores practical translations of MARLIN, ETD-MAPPO, and BPTA ideas into software engineering workflows.

If you are viewing the target remote repository, use this URL:
- [aithal007/GreenMARL---Coder](https://github.com/aithal007/GreenMARL---Coder.git)
