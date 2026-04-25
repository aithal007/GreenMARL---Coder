#!/usr/bin/env bash
set -euo pipefail

# Runs the full submission pipeline on remote HF GPU jobs:
# 1) eval in full mode + metrics
# 2) reward/pass plot
# 3) GRPO training run
# 4) upload artifacts to HF Space repo for reviewers
#
# Required env:
#   HF_SPACE_REPO (default: Aithal04/metaai)
# Optional env:
#   MODEL (default: Qwen/Qwen2.5-Coder-1.5B-Instruct)
#   GRPO_MODEL (default: Qwen/Qwen2.5-0.5B-Instruct)
#   EVAL_EPISODES (default: 20)
#   GRPO_STEPS (default: 80)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT_DIR"

HF_SPACE_REPO="${HF_SPACE_REPO:-Aithal04/metaai}"
MODEL="${MODEL:-Qwen/Qwen2.5-Coder-1.5B-Instruct}"
GRPO_MODEL="${GRPO_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
GRPO_STEPS="${GRPO_STEPS:-80}"

export USE_TORCH=1
export USE_TF=0
export TRL_EXPERIMENTAL_SILENCE=1
export PYTHONUNBUFFERED=1

python -m pip install --upgrade pip
# Install only runtime + training deps needed for CLI eval/GRPO to avoid
# resolver backtracking from app-specific pins.
python -m pip install \
  "torch>=2.2.0" \
  "accelerate>=1.4.0" \
  "sentencepiece>=0.2.0" \
  "protobuf>=3.20.0" \
  "numpy>=1.26.0" \
  "trl>=1.2.0" \
  "datasets>=4.8.0" \
  "peft>=0.19.0" \
  "matplotlib>=3.8.0" \
  "jmespath>=1.0.1" \
  "openenv-core[core]>=0.2.3"

# GRPOTrainer `environment_factory` requires transformers main branch.
python -m pip install --upgrade "git+https://github.com/huggingface/transformers.git@main"

mkdir -p logs assets/figures

echo "==> Running FULL eval (${EVAL_EPISODES} episodes) on GPU"
python main.py --full --episodes "${EVAL_EPISODES}" --save-metrics --device cuda

echo "==> Building reward/pass figure"
python training/plot_run_metrics.py --input logs/metrics_full.json

echo "==> Running GRPO training (${GRPO_STEPS} steps)"
python training/grpo_coding_gym.py --model "${GRPO_MODEL}" --max-steps "${GRPO_STEPS}"

echo "==> Writing run summary"
python - <<'PY'
import json
from pathlib import Path

p = Path("logs/metrics_full.json")
d = json.loads(p.read_text(encoding="utf-8"))
avg_reward = sum(float(x.get("total_reward", 0.0)) for x in d) / max(1, len(d))
avg_pass = sum(float(x.get("pass_rate", 0.0)) for x in d) / max(1, len(d))
avg_hidden = sum(float(x.get("hidden_pass_rate", 0.0)) for x in d) / max(1, len(d))
summary = (
    f"episodes={len(d)}\n"
    f"avg_reward={avg_reward:.4f}\n"
    f"avg_pass_rate={avg_pass:.4f}\n"
    f"avg_hidden_pass_rate={avg_hidden:.4f}\n"
)
Path("logs/hf_gpu_run_summary.txt").write_text(summary, encoding="utf-8")
print(summary)
PY

echo "==> Uploading artifacts to HF Space (${HF_SPACE_REPO})"
hf upload "${HF_SPACE_REPO}" logs/metrics_full.json artifacts/metrics_full.json --repo-type space
hf upload "${HF_SPACE_REPO}" assets/figures/reward_and_pass_by_episode.png artifacts/reward_and_pass_by_episode.png --repo-type space
hf upload "${HF_SPACE_REPO}" logs/hf_gpu_run_summary.txt artifacts/hf_gpu_run_summary.txt --repo-type space
if [[ -f "logs/grpo_coding/grpo_config_used.json" ]]; then
  hf upload "${HF_SPACE_REPO}" logs/grpo_coding/grpo_config_used.json artifacts/grpo_config_used.json --repo-type space
fi

echo "HF GPU pipeline finished successfully."
