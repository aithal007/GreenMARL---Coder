"""
GreenMARL-Coder — Gradio Space Demo
=====================================
Runs on HuggingFace Spaces (GPU T4 recommended).
Demonstrates MARLIN + ETD-MAPPO + Hybrid BPTA in a live multi-agent coding loop.

Set HF_TOKEN as a Space secret to enable model downloads.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

import gradio as gr

# ── Logging ────────────────────────────────────────────────────────────────
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
logger = logging.getLogger("app")

# ── Device / model defaults ────────────────────────────────────────────────
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"

DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

# ── Agent factory (fresh agents per run, no cross-run state) ───────────────

def _make_agents(model_name: str, etd_enabled: bool):
    from agents.planner import PlannerAgent
    from agents.coder import CoderAgent
    from agents.debugger import DebuggerAgent

    planner  = PlannerAgent(model_name=model_name, device=DEVICE, max_new_tokens=256)
    coder    = CoderAgent(
        model_name=model_name, device=DEVICE,
        max_new_tokens=400, etd_enabled=etd_enabled,
    )
    debugger = DebuggerAgent(model_name=model_name, device=DEVICE, max_new_tokens=300)
    return planner, coder, debugger


# ── Metrics formatter ──────────────────────────────────────────────────────

def _fmt_table(metrics_list) -> str:
    if not metrics_list:
        return "No results yet."
    header = (
        f"{'Ep':>4} | {'Task':<12} | {'Reward':>7} | {'Pass':>6} | "
        f"{'Hidden':>6} | {'Infer':>5} | {'Sleep':>5} | {'ETD%':>5} | "
        f"{'G':>4} | {'Time':>6}"
    )
    sep = "-" * len(header)
    rows = [header, sep]
    for m in metrics_list:
        eff = m.sleep_count / max(m.inference_count + m.sleep_count, 1)
        rows.append(
            f"{m.episode + 1:>4} | {m.task_id:<12} | {m.total_reward:>7.3f} | "
            f"{m.pass_rate:>6.0%} | {m.hidden_pass_rate:>6.0%} | "
            f"{m.inference_count:>5} | {m.sleep_count:>5} | {eff:>5.0%} | "
            f"{m.planner_generator:>4} | {m.time_s:>5.1f}s"
        )
    if len(metrics_list) > 1:
        rows.append(sep)
        avg_r = sum(m.total_reward for m in metrics_list) / len(metrics_list)
        avg_p = sum(m.pass_rate for m in metrics_list) / len(metrics_list)
        avg_h = sum(m.hidden_pass_rate for m in metrics_list) / len(metrics_list)
        ti = sum(m.inference_count for m in metrics_list)
        ts = sum(m.sleep_count for m in metrics_list)
        rows.append(
            f"{'AVG':>4} | {'':12} | {avg_r:>7.3f} | "
            f"{avg_p:>6.0%} | {avg_h:>6.0%} | "
            f"{ti:>5} | {ts:>5} | {ts / max(ti + ts, 1):>5.0%} | {'':>4} | {'':>6}"
        )
    return "\n".join(rows)


# ── Single-mode runner (generator — streams output) ────────────────────────

def run_mode(mode: str, model_name: str, episodes: int, task_id: str):
    """Yields (log_text, metrics_table, chat_log) after each episode."""
    from env.coding_gym import CodingGym
    from core.bpta_coordinator import BPTACoordinator

    chat_log_path = LOGS_DIR / f"agent_chat_{mode}.txt"
    chat_log_path.write_text("", encoding="utf-8")

    etd_enabled = (mode == "full")
    gym = CodingGym()
    planner, coder, debugger = _make_agents(model_name, etd_enabled)
    coordinator = BPTACoordinator(
        gym=gym, planner=planner, coder=coder, debugger=debugger,
        mode=mode, steps_per_episode=1, chat_log_path=chat_log_path,
    )

    all_metrics = []
    log_lines = [f"▶ [{mode.upper()}] model={model_name} device={DEVICE}\n"]

    for ep in range(episodes):
        if ep > 0:
            gym.advance()

        yield (
            "\n".join(log_lines) + f"\n⏳ Episode {ep + 1}/{episodes} running…",
            _fmt_table(all_metrics),
            chat_log_path.read_text(encoding="utf-8") if chat_log_path.exists() else "",
        )

        m = coordinator.run_episode(
            task_id=task_id if task_id != "auto" else None
        )
        all_metrics.append(m)

        eff = m.sleep_count / max(m.inference_count + m.sleep_count, 1)
        log_lines.append(
            f"Ep {m.episode + 1:02d} | {m.task_id} | reward={m.total_reward:.3f} "
            f"| pass={m.pass_rate:.0%} | hidden={m.hidden_pass_rate:.0%} "
            f"| infer={m.inference_count} | sleep={m.sleep_count} "
            f"| ETD={eff:.0%} | G={m.planner_generator} | {m.time_s:.1f}s"
        )

        yield (
            "\n".join(log_lines),
            _fmt_table(all_metrics),
            chat_log_path.read_text(encoding="utf-8") if chat_log_path.exists() else "",
        )

    coordinator.close()
    metrics_path = LOGS_DIR / f"metrics_{mode}.json"
    with open(metrics_path, "w") as f:
        json.dump(
            [m.__dict__ for m in all_metrics],
            f, indent=2, default=str,
        )
    log_lines.append(f"\n✅ Done — metrics saved to {metrics_path}")
    yield (
        "\n".join(log_lines),
        _fmt_table(all_metrics),
        chat_log_path.read_text(encoding="utf-8") if chat_log_path.exists() else "",
    )


# ── Compare runner ─────────────────────────────────────────────────────────

def run_compare(model_name: str, episodes: int):
    from env.coding_gym import CodingGym
    from core.bpta_coordinator import BPTACoordinator

    results: dict[str, list] = {}
    log_parts = ["▶ COMPARE MODE — running baseline → multi_agent → full\n"]

    for mode in ("baseline", "multi_agent", "full"):
        log_parts.append(f"\n{'=' * 50}\n▶ {mode.upper()}\n{'=' * 50}")
        chat_log_path = LOGS_DIR / f"agent_chat_{mode}.txt"
        chat_log_path.write_text("", encoding="utf-8")

        gym = CodingGym()
        planner, coder, debugger = _make_agents(model_name, etd_enabled=(mode == "full"))
        coordinator = BPTACoordinator(
            gym=gym, planner=planner, coder=coder, debugger=debugger,
            mode=mode, steps_per_episode=1, chat_log_path=chat_log_path,
        )
        mode_metrics = []
        for ep in range(episodes):
            if ep > 0:
                gym.advance()
            m = coordinator.run_episode()
            mode_metrics.append(m)
            log_parts.append(
                f"  Ep{ep + 1}: {m.task_id} reward={m.total_reward:.3f} "
                f"pass={m.pass_rate:.0%} infer={m.inference_count} sleep={m.sleep_count}"
            )
        coordinator.close()
        results[mode] = mode_metrics

    table = _compare_table(results)
    log_parts.append("\n" + table)
    full_log = "\n".join(log_parts)
    return full_log, table


def _compare_table(results: dict) -> str:
    def avg(lst, key):
        vals = [getattr(m, key) for m in lst]
        return sum(vals) / max(len(vals), 1)

    col = 18
    lines = ["", "=" * 65, "COMPARISON SUMMARY", "=" * 65]
    lines.append(
        "Metric".ljust(col) + "Baseline".ljust(col) +
        "MultiAgent".ljust(col) + "GreenMARL-Full"
    )
    lines.append("-" * 65)

    for label, key, fmt in [
        ("Avg Reward",        "total_reward",      ".3f"),
        ("Avg Pass Rate",     "pass_rate",          ".1%"),
        ("Avg Hidden Pass",   "hidden_pass_rate",   ".1%"),
        ("Avg Shaped Reward", "shaped_reward",      ".3f"),
        ("Avg Value Est.",    "value_estimate",     ".3f"),
        ("Avg Time (s)",      "time_s",             ".1f"),
    ]:
        b  = avg(results.get("baseline",    []), key)
        ma = avg(results.get("multi_agent", []), key)
        f  = avg(results.get("full",        []), key)
        lines.append(
            label.ljust(col) +
            format(b,  fmt).ljust(col) +
            format(ma, fmt).ljust(col) +
            format(f,  fmt)
        )

    lines.append("-" * 65)
    bl = results.get("baseline", [])
    fu = results.get("full", [])
    bl_i = sum(m.inference_count for m in bl)
    fu_i = sum(m.inference_count for m in fu)
    fu_s = sum(m.sleep_count for m in fu)
    savings_pct = (1 - fu_i / max(bl_i, 1)) * 100
    etd_eff     = fu_s / max(fu_i + fu_s, 1)
    lines.append(f"ETD inferences saved vs baseline:  {savings_pct:.1f}%  (target >30%)")
    lines.append(f"ETD sleep ratio (full mode):        {etd_eff:.1%}")
    status = "✅ PASS" if savings_pct > 30 or etd_eff > 0.3 else "⚠️  BELOW TARGET (run more episodes)"
    lines.append(f"Status: {status}")
    lines.append("=" * 65)
    return "\n".join(lines)


# ── Gradio UI ──────────────────────────────────────────────────────────────

DESCRIPTION = """
# GreenMARL-Coder

Multi-agent coding assistant combining three MARL research ideas:

| Component | Paper | Role |
|-----------|-------|------|
| **MARLIN** | Godfrey et al. 2025 | LLM planner — ADS/IAN generator switching |
| **ETD-MAPPO** | — | Entropy-gated coder — sleeps when confident, saves inference |
| **BPTA** | Li et al. 2023 | Debugger critique backpropagated into Coder/Planner prompts |

**Three modes:** `baseline` (single agent) · `multi_agent` (no ETD/BPTA) · `full` (all active)

[GitHub](https://github.com/aithal007/GreenMARL---Coder)
"""

TASK_LABELS = {
    "auto":     "auto — cycle through all tasks",
    "task_001": "task_001 — Two Sum (easy)",
    "task_002": "task_002 — Valid Parentheses (easy)",
    "task_003": "task_003 — Fibonacci + Memoisation (medium)",
    "task_004": "task_004 — Maximum Subarray / Kadane's (medium)",
    "task_005": "task_005 — Group Anagrams (medium)",
}

ARCHITECTURE_MD = """
## System Architecture

```
Episode loop
├── PlannerAgent  (MARLIN)
│   ├── Selects generator: G_ADS or G_IAN based on past performance buffer
│   ├── episode < m      → G_ADS   (Adaptive Decision Search)
│   ├── episode < 2m     → G_IAN   (Iterative Adaptive Negotiation)
│   ├── else             → adaptive (p_plan==1 or p<p_ILM → G_IAN, else G_ADS)
│   └── Emits structured plan JSON (strategy, algorithm, edge_cases, complexity)
│
├── CoderAgent  (ETD-MAPPO)
│   ├── Probes Shannon entropy H over first 8 next-token logprob distributions
│   ├── If H < 1.5 bits  AND  past rewards stable  AND  solution exists
│   │   → returns "sleep_token"  (reuse previous solution, save inference)
│   ├── Otherwise generates full Python solution guided by Planner's plan
│   └── Receives BPTA delta injected into prompt for next episode
│
├── CodingGym  (Environment)
│   ├── Runs code in isolated subprocess (5 s hard timeout)
│   ├── Visible test cases → pass_rate
│   ├── Hidden assertions  → hidden_pass_rate (bonus)
│   └── reward = pass_rate + 0.5 * hidden_rate − time_penalty
│
├── DebuggerAgent  (BPTA Critic)
│   ├── Receives code + test results from Gym
│   ├── Outputs: value_estimate V(s) ∈ [−1,1]
│   │            reward_shaping Δr ∈ [−0.3, 0.3]
│   │            bpta_delta — actionable fix instructions for Coder
│   └── In-context backward pass: bpta_delta injected into Coder's next prompt
│
└── BPTACoordinator
    ├── In-context BPTA: injects Debugger bpta_delta into Coder.context_delta
    └── Stub PyTorch StateAdapter MLP [6 → 64 → 32]
        pseudo_loss = −V(s)  (maximise value estimate)
        demonstrates latent differentiable feedback pathway
```

## Key metrics

| Metric | What it proves |
|--------|---------------|
| **ETD sleep ratio** | % of inference calls avoided by entropy gating |
| **Pass rate across episodes** | BPTA feedback improves quality over time |
| **Planner G column** | MARLIN ADS ↔ IAN switching is happening |
| **Shaped vs raw reward** | Debugger adds non-trivial reward signal |

## Coding tasks

5 algorithmic puzzles with visible test cases + hidden edge-case assertions:

| ID | Name | Difficulty |
|----|------|-----------|
| task_001 | Two Sum | easy |
| task_002 | Valid Parentheses | easy |
| task_003 | Fibonacci + Memoisation | medium |
| task_004 | Maximum Subarray (Kadane's) | medium |
| task_005 | Group Anagrams | medium |
"""

with gr.Blocks(title="GreenMARL-Coder", theme=gr.themes.Soft()) as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Tabs():

        # Tab 1 — Single mode
        with gr.Tab("Run Single Mode"):
            with gr.Row():
                with gr.Column(scale=1):
                    model_input = gr.Textbox(
                        value=DEFAULT_MODEL,
                        label="HuggingFace Model",
                        info="Qwen2.5-Coder-1.5B recommended (fast on CPU/T4)",
                    )
                    mode_radio = gr.Radio(
                        choices=["baseline", "multi_agent", "full"],
                        value="full",
                        label="Mode",
                        info="baseline=single agent | multi_agent=no ETD/BPTA | full=all active",
                    )
                    episodes_slider = gr.Slider(
                        minimum=1, maximum=10, value=3, step=1,
                        label="Episodes",
                        info="ETD sleep only triggers after 3+ stable reward episodes",
                    )
                    task_dd = gr.Dropdown(
                        choices=list(TASK_LABELS.keys()),
                        value="auto",
                        label="Task (or auto-cycle)",
                        type="value",
                    )
                    run_btn = gr.Button("Run", variant="primary")

                with gr.Column(scale=2):
                    log_out = gr.Textbox(
                        label="Episode Log",
                        lines=12, max_lines=20,
                        show_copy_button=True,
                    )

            metrics_out = gr.Textbox(
                label="Metrics Table",
                lines=10, max_lines=15,
                show_copy_button=True,
            )
            chat_out = gr.Textbox(
                label="Agent Chat Log (agent_chat.txt)",
                lines=15, max_lines=30,
                show_copy_button=True,
            )

            run_btn.click(
                fn=run_mode,
                inputs=[mode_radio, model_input, episodes_slider, task_dd],
                outputs=[log_out, metrics_out, chat_out],
                show_progress="minimal",
            )

        # Tab 2 — Compare
        with gr.Tab("Compare All Modes"):
            gr.Markdown(
                "Runs **baseline → multi_agent → full** back-to-back and prints "
                "a side-by-side table. This is the best view for judging ETD compute savings."
            )
            with gr.Row():
                cmp_model    = gr.Textbox(value=DEFAULT_MODEL, label="Model")
                cmp_episodes = gr.Slider(1, 5, value=2, step=1, label="Episodes per mode")
                cmp_btn      = gr.Button("Compare", variant="primary")
            cmp_log   = gr.Textbox(label="Full Log",          lines=20, show_copy_button=True)
            cmp_table = gr.Textbox(label="Comparison Table",  lines=15, show_copy_button=True)

            cmp_btn.click(
                fn=run_compare,
                inputs=[cmp_model, cmp_episodes],
                outputs=[cmp_log, cmp_table],
                show_progress="minimal",
            )

        # Tab 3 — Architecture
        with gr.Tab("Architecture"):
            gr.Markdown(ARCHITECTURE_MD)

        # Tab 4 — Logs
        with gr.Tab("Raw Logs"):
            refresh_btn = gr.Button("Refresh run.log")
            raw_log = gr.Textbox(label="run.log", lines=20, show_copy_button=True)

            def _read_log():
                p = LOGS_DIR / "run.log"
                return p.read_text(encoding="utf-8") if p.exists() else "No logs yet."

            refresh_btn.click(fn=_read_log, outputs=raw_log)
            demo.load(fn=_read_log, outputs=raw_log)

    gr.Markdown(
        f"---\nRunning on **{DEVICE.upper()}** · "
        "[GitHub](https://github.com/aithal007/GreenMARL---Coder)"
    )


if __name__ == "__main__":
    demo.launch()
