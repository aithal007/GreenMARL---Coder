#!/usr/bin/env python3
"""
Plot reward / pass rate from saved `logs/metrics_*.json` (main.py --save-metrics).
Writes PNGs under assets/figures/ for README embedding.

Usage:
  python training/plot_run_metrics.py
  python training/plot_run_metrics.py --input logs/metrics_full.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=Path,
        default=_ROOT / "logs" / "metrics_full.json",
        help="JSON list of episode dicts (EpisodeMetrics as dicts).",
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=_ROOT / "assets" / "figures",
    )
    args = ap.parse_args()

    if not args.input.is_file():
        print(f"Missing input file: {args.input} — run main.py with --save-metrics first.")
        return 1

    data = json.loads(args.input.read_text(encoding="utf-8"))
    ep_idx = list(range(1, len(data) + 1))
    rewards = [float(d.get("total_reward", 0)) for d in data]
    pass_r = [float(d.get("pass_rate", 0)) for d in data]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Please `pip install matplotlib` to generate plots.", file=__import__("sys").stderr)
        return 1

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(ep_idx, rewards, marker="o", label="Total reward (gym + shaping)")
    ax2 = ax.twinx()
    ax2.plot(ep_idx, pass_r, marker="s", color="tab:green", label="Visible pass rate")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward (scalar)")
    ax2.set_ylabel("Visible pass rate (0-1)")
    ax.set_title("GreenMARL-Coder — per-episode metrics (from JSON export)")
    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, loc="lower right")
    fig.tight_layout()
    out = args.out_dir / "reward_and_pass_by_episode.png"
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
