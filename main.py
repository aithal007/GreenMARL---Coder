"""
GreenMARL-Coder — Orchestration Script
=======================================

Three execution modes:

  --baseline      Single CoderAgent only. No Planner, no Debugger, no ETD gating,
                  no BPTA. Establishes the performance floor.

  --multi-agent   All three agents active. ETD gating is disabled (always generates).
                  BPTA delta is always zero. Isolates the coordination benefit.

  --full          Complete GreenMARL-Coder system: MARLIN planning, ETD gating,
                  Hybrid BPTA backward pass.

Tracked metrics (printed + saved to logs/metrics.json):
  - Accuracy: visible pass rate, hidden pass rate per episode
  - Efficiency: inference count, sleep count, efficiency ratio (ETD saves)
  - Time-to-solution: wall-clock seconds per episode
  - Reward: raw gym reward and BPTA-shaped reward

Usage:
  python main.py --baseline   --episodes 5 --model Qwen/Qwen2.5-Coder-1.5B-Instruct
  python main.py --multi-agent --episodes 5
  python main.py --full        --episodes 10 --steps 2
  python main.py --compare     --episodes 5       # runs all 3 modes and prints table
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Logging setup — console INFO, file DEBUG
# ---------------------------------------------------------------------------
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "run.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("main")


# ---------------------------------------------------------------------------
# Lazy imports (heavy ML deps, skip if --help)
# ---------------------------------------------------------------------------

def _build_agents(model_name: str, device: str, etd_enabled: bool):
    from agents.planner import PlannerAgent
    from agents.coder import CoderAgent
    from agents.debugger import DebuggerAgent

    planner = PlannerAgent(model_name=model_name, device=device, max_new_tokens=300)
    coder = CoderAgent(
        model_name=model_name,
        device=device,
        max_new_tokens=512,
        etd_enabled=etd_enabled,
    )
    debugger = DebuggerAgent(model_name=model_name, device=device, max_new_tokens=400)
    return planner, coder, debugger


def _build_coordinator(gym, planner, coder, debugger, mode: str, steps: int):
    from core.bpta_coordinator import BPTACoordinator
    return BPTACoordinator(
        gym=gym,
        planner=planner,
        coder=coder,
        debugger=debugger,
        mode=mode,
        steps_per_episode=steps,
    )


# ---------------------------------------------------------------------------
# Mode runners
# ---------------------------------------------------------------------------

def run_baseline(args: argparse.Namespace) -> list[dict]:
    """Single CoderAgent, no coordination, no ETD."""
    logger.info("=== BASELINE MODE ===")
    from env.coding_gym import CodingGym
    gym = CodingGym()
    planner, coder, debugger = _build_agents(
        args.model, args.device, etd_enabled=False
    )
    coordinator = _build_coordinator(gym, planner, coder, debugger, "baseline", args.steps)

    metrics_list: list[dict] = []
    for ep in range(args.episodes):
        gym.advance() if ep > 0 else None
        m = coordinator.run_episode()
        metrics_list.append(_metrics_to_dict(m))
        _print_episode(m)

    coordinator.close()
    return metrics_list


def run_multi_agent(args: argparse.Namespace) -> list[dict]:
    """All three agents, no ETD/BPTA."""
    logger.info("=== MULTI-AGENT MODE (no ETD/BPTA) ===")
    from env.coding_gym import CodingGym
    gym = CodingGym()
    planner, coder, debugger = _build_agents(
        args.model, args.device, etd_enabled=False
    )
    coordinator = _build_coordinator(
        gym, planner, coder, debugger, "multi_agent", args.steps
    )

    metrics_list: list[dict] = []
    for ep in range(args.episodes):
        if ep > 0:
            gym.advance()
        m = coordinator.run_episode()
        metrics_list.append(_metrics_to_dict(m))
        _print_episode(m)

    coordinator.close()
    return metrics_list


def run_full(args: argparse.Namespace) -> list[dict]:
    """Complete GreenMARL-Coder: MARLIN + ETD + Hybrid BPTA."""
    logger.info("=== FULL GREENMARL-CODER MODE ===")
    from env.coding_gym import CodingGym
    gym = CodingGym()
    planner, coder, debugger = _build_agents(
        args.model, args.device, etd_enabled=True
    )
    coordinator = _build_coordinator(
        gym, planner, coder, debugger, "full", args.steps
    )

    metrics_list: list[dict] = []
    for ep in range(args.episodes):
        if ep > 0:
            gym.advance()
        m = coordinator.run_episode()
        metrics_list.append(_metrics_to_dict(m))
        _print_episode(m)

    coordinator.close()
    return metrics_list


# ---------------------------------------------------------------------------
# Compare mode
# ---------------------------------------------------------------------------

def run_compare(args: argparse.Namespace) -> None:
    """Run all three modes sequentially and print a comparison table."""
    logger.info("=== COMPARE MODE ===")

    results: dict[str, list[dict]] = {}
    for mode_fn, label in [
        (run_baseline, "baseline"),
        (run_multi_agent, "multi_agent"),
        (run_full, "full"),
    ]:
        logger.info("\n--- Running %s ---", label)
        mode_args = argparse.Namespace(**vars(args))
        results[label] = mode_fn(mode_args)

    _print_comparison(results)
    _save_metrics(results, LOGS_DIR / "metrics_compare.json")


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _metrics_to_dict(m: Any) -> dict:
    return {
        "episode": m.episode,
        "task_id": m.task_id,
        "mode": m.mode,
        "total_reward": m.total_reward,
        "shaped_reward": m.shaped_reward,
        "pass_rate": m.pass_rate,
        "hidden_pass_rate": m.hidden_pass_rate,
        "inference_count": m.inference_count,
        "sleep_count": m.sleep_count,
        "efficiency_ratio": (
            m.sleep_count / max(m.inference_count + m.sleep_count, 1)
        ),
        "time_s": m.time_s,
        "planner_generator": m.planner_generator,
        "value_estimate": m.value_estimate,
    }


def _print_episode(m: Any) -> None:
    line = (
        f"  Ep {m.episode:02d} | {m.task_id} | reward={m.total_reward:.3f} "
        f"| pass={m.pass_rate:.0%} | hidden={m.hidden_pass_rate:.0%} "
        f"| infer={m.inference_count} | sleep={m.sleep_count} "
        f"| {m.time_s:.1f}s | G={m.planner_generator}"
    )
    print(line)


def _print_comparison(results: dict[str, list[dict]]) -> None:
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    headers = ["Metric", "Baseline", "MultiAgent", "GreenMARL-Full"]
    col_w = 18

    def avg(lst: list[dict], key: str) -> float:
        vals = [x[key] for x in lst if key in x]
        return sum(vals) / len(vals) if vals else 0.0

    rows = [
        ("Avg Reward", "total_reward"),
        ("Avg Pass Rate", "pass_rate"),
        ("Avg Hidden Pass", "hidden_pass_rate"),
        ("Avg Inferences", "inference_count"),
        ("ETD Efficiency", "efficiency_ratio"),
        ("Avg Time (s)", "time_s"),
    ]

    header_line = "".join(h.ljust(col_w) for h in headers)
    print(header_line)
    print("-" * (col_w * len(headers)))

    for label, key in rows:
        baseline_v = avg(results.get("baseline", []), key)
        multi_v = avg(results.get("multi_agent", []), key)
        full_v = avg(results.get("full", []), key)

        if key == "efficiency_ratio":
            row = (
                f"{label:<{col_w}}"
                f"{baseline_v:.1%}{'':>{col_w - 5}}"
                f"{multi_v:.1%}{'':>{col_w - 5}}"
                f"{full_v:.1%}"
            )
        elif key in ("inference_count",):
            row = (
                f"{label:<{col_w}}"
                f"{baseline_v:.1f}{'':>{col_w - 5}}"
                f"{multi_v:.1f}{'':>{col_w - 5}}"
                f"{full_v:.1f}"
            )
        else:
            row = (
                f"{label:<{col_w}}"
                f"{baseline_v:.3f}{'':>{col_w - 6}}"
                f"{multi_v:.3f}{'':>{col_w - 6}}"
                f"{full_v:.3f}"
            )
        print(row)

    # ETD compute savings vs baseline
    base_infer = avg(results.get("baseline", [{"inference_count": 1}]), "inference_count")
    full_infer = avg(results.get("full", [{"inference_count": 1}]), "inference_count")
    savings_pct = (1 - full_infer / max(base_infer, 1)) * 100
    print(f"\n  ETD compute savings vs baseline: {savings_pct:.1f}%")
    print(f"  Target: >30%  {'PASS' if savings_pct > 30 else 'BELOW TARGET'}")
    print("=" * 70)


def _save_metrics(data: Any, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info("Metrics saved to %s", path)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GreenMARL-Coder multi-agent coding assistant"
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--baseline", action="store_true",
                            help="Single-agent baseline (no coordination)")
    mode_group.add_argument("--multi-agent", action="store_true",
                            help="Multi-agent without ETD/BPTA")
    mode_group.add_argument("--full", action="store_true",
                            help="Full GreenMARL-Coder (MARLIN + ETD + BPTA)")
    mode_group.add_argument("--compare", action="store_true",
                            help="Run all three modes and compare")

    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        help="HuggingFace model name (default: Qwen/Qwen2.5-Coder-1.5B-Instruct)",
    )
    parser.add_argument("--device", default="cpu",
                        help="PyTorch device (cpu | cuda | mps)")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes to run per mode")
    parser.add_argument("--steps", type=int, default=1,
                        help="Steps per episode (reruns on same task)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING"],
                        help="Console log level")
    parser.add_argument("--save-metrics", action="store_true",
                        help="Save per-episode metrics to logs/metrics.json")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    logging.getLogger().setLevel(args.log_level)

    print(f"\nGreenMARL-Coder | model={args.model} | device={args.device}")
    print(f"episodes={args.episodes} | steps/ep={args.steps}\n")

    t_start = time.perf_counter()

    if args.compare:
        run_compare(args)
    elif args.baseline:
        data = run_baseline(args)
        if args.save_metrics:
            _save_metrics(data, LOGS_DIR / "metrics_baseline.json")
    elif getattr(args, "multi_agent", False):
        data = run_multi_agent(args)
        if args.save_metrics:
            _save_metrics(data, LOGS_DIR / "metrics_multi_agent.json")
    elif args.full:
        data = run_full(args)
        if args.save_metrics:
            _save_metrics(data, LOGS_DIR / "metrics_full.json")

    elapsed = time.perf_counter() - t_start
    print(f"\nTotal wall-clock time: {elapsed:.1f}s")
    print(f"Agent chat log: {LOGS_DIR / 'agent_chat.txt'}")


if __name__ == "__main__":
    main()
