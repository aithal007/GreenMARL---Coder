#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""
GRPO training on the GreenMARL CodingGym using HuggingFace TRL + tool environments.

This follows the OpenEnv+TRL pattern (see
https://huggingface.co/docs/trl/main/openenv ).

Set before any `transformers` / `trl` import (avoids a broken optional TensorFlow on some Windows Conda envs)::

  export USE_TORCH=1
  export USE_TF=0

Example (T4+ GPU, short demo run)::

  pip install -r requirements-train.txt
  USE_TORCH=1 USE_TF=0 python training/grpo_coding_gym.py --smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

# --- Bootstrap: torch-only, no optional TF ---------------------------------
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("USE_TF", "0")

# Project root
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _reward_from_envs(environments, **kwargs) -> list[float | None]:  # noqa: ARG001
    return [float(getattr(env, "reward", 0.0)) for env in environments]


def main() -> int:
    parser = argparse.ArgumentParser(description="GRPO on CodingGym (TRL + tool env).")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Causal LM with tool / chat template (smaller = faster on T4).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=_ROOT / "logs" / "grpo_coding",
        help="Checkpoints, logs, and training_metrics.jsonl",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Minimal steps/epochs to verify the pipeline (for CI/Colab smoke).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=30,
        help="Training steps for non-smoke runs.",
    )
    args = parser.parse_args()

    import torch

    try:
        from datasets import Dataset
    except ImportError as e:
        print("ERROR: `datasets` is required. pip install -r requirements-train.txt", file=sys.stderr)
        raise SystemExit(1) from e

    try:
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as e:
        print("ERROR: `trl` is required. pip install -r requirements-train.txt", file=sys.stderr)
        raise SystemExit(1) from e

    from greenmarl_openenv.coding_trl_env import CodingGymToolEnv

    # Same prompt repeated: tool loop will run submit_python_solution
    n = 4 if not args.smoke else 2
    user_lines = [
        "Solve the programming task. Read the system message, then call "
        "`submit_python_solution` with a correct Python function body."
    ] * n
    dataset = Dataset.from_dict(
        {
            "prompt": [
                [{"role": "user", "content": t}]
                for t in user_lines
            ],
        }
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    max_s = 2 if args.smoke else int(args.max_steps)
    # num_generations must divide effective batch (per_device * accum * world_size)
    num_gen = 2
    bsz = 2
    use_cuda = torch.cuda.is_available()
    gcfg = GRPOConfig(
        output_dir=str(out),
        per_device_train_batch_size=bsz,
        gradient_accumulation_steps=1,
        num_generations=num_gen,
        max_completion_length=1024,
        max_steps=max_s,
        save_steps=1000,
        logging_steps=1,
        report_to="none",
        bf16=use_cuda,
        fp16=False,
    )

    trainer = GRPOTrainer(
        model=args.model,
        train_dataset=dataset,
        reward_funcs=_reward_from_envs,
        args=gcfg,
        environment_factory=CodingGymToolEnv,
    )
    trainer.train()
    try:
        cfg_dump = {k: repr(v) for k, v in asdict(gcfg).items()}
    except Exception:  # noqa: BLE001
        cfg_dump = {"repr": repr(gcfg)}
    (out / "grpo_config_used.json").write_text(
        json.dumps(cfg_dump, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"Done. Artifacts in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
