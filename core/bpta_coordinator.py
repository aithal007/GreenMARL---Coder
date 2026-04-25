"""
BPTA Coordinator — orchestrates the Coder → Gym → Debugger → Coder loop.

Fixes applied vs original:
  - inference_count / sleep_count are now per-episode (snapshot delta, not cumulative).
  - obs.update() replaced with careful merge that preserves task fields.
  - chat_log_path is a constructor parameter (supports Gradio demo & main.py).

Hybrid BPTA implementation:
  [In-Context] Debugger's bpta_delta injected into Coder's prompt context.
  [Stub PyTorch] StateAdapter MLP demonstrates differentiable feedback pathway.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim

from agents.coder import CoderAgent
from agents.debugger import DebuggerAgent, DebuggerOutput
from agents.planner import PlannerAgent
from env.coding_gym import CodingGym, StepResult

logger = logging.getLogger(__name__)

LOGS_DIR = Path(__file__).parent.parent / "logs"
CHAT_LOG = LOGS_DIR / "agent_chat.txt"


# ---------------------------------------------------------------------------
# Stub PyTorch State Adapter (BPTA conceptual component)
# ---------------------------------------------------------------------------

class StateAdapter(nn.Module):
    """
    Tiny MLP: state_vec (6,) → context_offset (32,)

    Input features:
        [pass_rate, hidden_pass_rate, reward, was_sleep, syntax_error, entropy_proxy]

    In a full BPTA system this would project into the LLM embedding space.
    """

    STATE_DIM = 6
    CONTEXT_DIM = 32

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(self.STATE_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, self.CONTEXT_DIM),
            nn.Tanh(),
        )

    def forward(self, state_vec: torch.Tensor) -> torch.Tensor:
        return self.net(state_vec)


# ---------------------------------------------------------------------------
# Episode-level metrics
# ---------------------------------------------------------------------------

@dataclass
class EpisodeMetrics:
    episode: int
    task_id: str
    mode: str
    steps: int
    total_reward: float
    shaped_reward: float
    pass_rate: float
    hidden_pass_rate: float
    inference_count: int    # per-episode delta (not cumulative)
    sleep_count: int        # per-episode delta (not cumulative)
    time_s: float
    planner_generator: str = ""
    value_estimate: float = 0.0
    bpta_delta_len: int = 0

    def to_row(self) -> list:
        return [
            self.episode,
            self.task_id,
            self.mode,
            f"{self.total_reward:.3f}",
            f"{self.shaped_reward:.3f}",
            f"{self.pass_rate:.2%}",
            f"{self.hidden_pass_rate:.2%}",
            self.inference_count,
            self.sleep_count,
            f"{self.time_s:.1f}s",
            self.planner_generator,
        ]


# ---------------------------------------------------------------------------
# Main coordinator
# ---------------------------------------------------------------------------

class BPTACoordinator:
    """
    Runs one complete episode: Planner → Coder → Gym → Debugger → BPTA.

    Args:
        chat_log_path: Where to write the agent conversation transcript.
                       Defaults to logs/agent_chat.txt. Pass a custom path
                       when running multiple modes in parallel (e.g. Gradio).
    """

    def __init__(
        self,
        gym: CodingGym,
        planner: PlannerAgent,
        coder: CoderAgent,
        debugger: DebuggerAgent,
        mode: str = "full",
        steps_per_episode: int = 1,
        adapter_lr: float = 1e-3,
        chat_log_path: Path = CHAT_LOG,
    ) -> None:
        self.gym = gym
        self.planner = planner
        self.coder = coder
        self.debugger = debugger
        self.mode = mode
        self.steps_per_episode = steps_per_episode

        self.adapter = StateAdapter()
        self.adapter_optimizer = optim.Adam(self.adapter.parameters(), lr=adapter_lr)

        LOGS_DIR.mkdir(exist_ok=True)
        self._chat_log_path = Path(chat_log_path)
        self._chat_log = open(self._chat_log_path, "a", encoding="utf-8")  # noqa: SIM115
        self._episode_count = 0

    # ------------------------------------------------------------------
    # Episode runner
    # ------------------------------------------------------------------

    def run_episode(self, task_id: str | None = None) -> EpisodeMetrics:
        """Run one complete episode (potentially multiple steps on the same task)."""
        t0 = time.perf_counter()
        ep = self._episode_count

        obs = self.gym.reset()
        if task_id:
            obs = self.gym._task_obs(self.gym._resolve_task(task_id))

        p_plan = (
            self.planner.past_performance_buffer[-1]
            if self.planner.past_performance_buffer
            else 0.0
        )
        self.planner.start_episode(self.steps_per_episode, p_plan=p_plan)

        # Snapshot cumulative counts at episode start → per-episode delta
        infer_start = self.coder.inference_count
        sleep_start = self.coder.sleep_count

        total_reward = 0.0
        total_shaped = 0.0
        best_pass_rate = 0.0
        best_hidden_rate = 0.0
        last_debugger_out: DebuggerOutput | None = None
        last_step: StepResult | None = None

        # Keep the task obs fields stable across steps
        task_obs: dict[str, Any] = dict(obs)

        for step_idx in range(self.steps_per_episode):
            step_result, debugger_out = self._run_step(task_obs, step_idx)
            last_step = step_result
            last_debugger_out = debugger_out

            total_reward += step_result.reward
            total_shaped += (
                debugger_out.shaped_reward if debugger_out else step_result.reward
            )
            best_pass_rate = max(best_pass_rate, step_result.pass_rate)
            best_hidden_rate = max(best_hidden_rate, step_result.hidden_pass_rate)

            # Merge result obs back — but do NOT overwrite top-level task fields
            # (description, function_signature, examples, task_id) that came from
            # the initial reset. Clobbering those breaks multi-step prompts.
            preserved = {
                k: task_obs[k]
                for k in ("task_id", "name", "description", "function_signature", "examples")
                if k in task_obs
            }
            task_obs.update(step_result.obs)
            task_obs.update(preserved)

        # Episode end bookkeeping
        self.planner.end_episode(best_pass_rate)
        self.coder.observe(task_obs, best_pass_rate)
        self._episode_count += 1

        metrics = EpisodeMetrics(
            episode=ep,
            task_id=task_obs.get("task_id", "?"),
            mode=self.mode,
            steps=self.steps_per_episode,
            total_reward=total_reward,
            shaped_reward=total_shaped,
            pass_rate=best_pass_rate,
            hidden_pass_rate=best_hidden_rate,
            # Per-episode counts (FIXED: use delta from snapshot)
            inference_count=self.coder.inference_count - infer_start,
            sleep_count=self.coder.sleep_count - sleep_start,
            time_s=time.perf_counter() - t0,
            planner_generator=self.planner.generator_mode,
            value_estimate=last_debugger_out.value_estimate if last_debugger_out else 0.0,
            bpta_delta_len=len(last_debugger_out.bpta_delta) if last_debugger_out else 0,
        )
        self._log_episode_summary(metrics)
        return metrics

    # ------------------------------------------------------------------
    # Single step
    # ------------------------------------------------------------------

    def _run_step(
        self,
        obs: dict[str, Any],
        step_idx: int,
    ) -> tuple[StepResult, DebuggerOutput | None]:
        """Execute one step: Planner → Coder → Gym → Debugger → BPTA."""
        import json

        # --- Planner (MARLIN) ---
        plan_json = ""
        if self.mode in ("multi_agent", "full"):
            plan_json = self.planner.act(obs)
            try:
                self.coder.current_plan = json.loads(plan_json)
            except Exception:
                self.coder.current_plan = None
            self._log("Planner", plan_json[:500])
        else:
            self.coder.current_plan = None

        # --- Coder (ETD-MAPPO) ---
        code_or_sleep = self.coder.act(obs)
        is_sleep = code_or_sleep.strip() == "sleep_token"

        if is_sleep:
            code_to_eval = self.coder.get_last_solution()
            self._log("Coder", "[ETD SLEEP] reusing last solution")
        else:
            code_to_eval = code_or_sleep
            self._log("Coder", code_or_sleep[:600])

        # --- Gym ---
        step_result = self.gym.step(
            code_to_eval if code_to_eval else "sleep_token",
            task_id=obs.get("task_id"),
        )
        self._log("Gym", step_result.summary())

        # --- Debugger (BPTA Critic) ---
        debugger_out: DebuggerOutput | None = None
        if self.mode in ("multi_agent", "full"):
            debugger_out = self.debugger.evaluate(
                step_result.obs,
                raw_reward=step_result.reward,
                code=code_to_eval,
            )
            self._log("Debugger", debugger_out.critique)

            if self.mode == "full":
                self._apply_bpta(debugger_out, step_result)

        return step_result, debugger_out

    # ------------------------------------------------------------------
    # BPTA backward pass
    # ------------------------------------------------------------------

    def _apply_bpta(
        self,
        debugger_out: DebuggerOutput,
        step_result: StepResult,
    ) -> None:
        """
        Hybrid BPTA:
        1. In-Context: inject Debugger's bpta_delta into Coder's context_delta.
        2. Stub PyTorch: forward+backward through StateAdapter (-V(s) loss).
        """
        # In-Context BPTA
        self.coder.inject_bpta_delta(debugger_out.bpta_delta)

        # Stub differentiable adapter
        state_vec = self._obs_to_state_vec(step_result.obs)
        context_offset = self.adapter(state_vec)

        # Build a gradient-connected pseudo-loss so backward() is always valid.
        value_scalar = float(debugger_out.value_estimate)
        value_tensor = torch.tensor(
            value_scalar,
            dtype=context_offset.dtype,
            device=context_offset.device,
        )
        loss = -(context_offset.mean() * value_tensor)

        self.adapter_optimizer.zero_grad()
        try:
            if torch.isfinite(loss):
                loss.backward()
                self.adapter_optimizer.step()
            else:
                logger.warning("[BPTA] Skipping adapter step due to non-finite loss.")
        except RuntimeError as exc:
            logger.warning("[BPTA] Adapter backward step skipped: %s", exc)

        offset_norm = context_offset.norm().item()
        logger.debug(
            "[BPTA] In-context delta injected (%d chars). "
            "offset_norm=%.4f value=%.3f pseudo_loss=%.4f",
            len(debugger_out.bpta_delta),
            offset_norm,
            debugger_out.value_estimate,
            loss.item(),
        )
        self._log(
            "BPTA",
            f"offset_norm={offset_norm:.4f} value={debugger_out.value_estimate:.3f} "
            f"| delta='{debugger_out.bpta_delta[:120]}'",
        )

    @staticmethod
    def _obs_to_state_vec(obs: dict[str, Any]) -> torch.Tensor:
        vec = [
            float(obs.get("pass_rate", 0.0)),
            float(obs.get("hidden_pass_rate", 0.0)),
            float(obs.get("reward", 0.0)),
            float(obs.get("was_sleep", False)),
            float(obs.get("syntax_error", False)),
            1.0 - float(obs.get("pass_rate", 0.0)),
        ]
        return torch.tensor(vec, dtype=torch.float32).unsqueeze(0)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, speaker: str, text: str) -> None:
        line = f"[Ep {self._episode_count:03d}][{speaker:10s}] {text}\n"
        self._chat_log.write(line)
        self._chat_log.flush()
        logger.debug(line.rstrip())

    def _log_episode_summary(self, metrics: EpisodeMetrics) -> None:
        efficiency = metrics.sleep_count / max(
            metrics.inference_count + metrics.sleep_count, 1
        )
        self._log(
            "SUMMARY",
            f"ep={metrics.episode} task={metrics.task_id} mode={metrics.mode} "
            f"reward={metrics.total_reward:.3f} shaped={metrics.shaped_reward:.3f} "
            f"pass={metrics.pass_rate:.2%} hidden={metrics.hidden_pass_rate:.2%} "
            f"infer={metrics.inference_count} sleep={metrics.sleep_count} "
            f"etd_eff={efficiency:.1%} "
            f"time={metrics.time_s:.1f}s planner_G={metrics.planner_generator}",
        )

    def close(self) -> None:
        try:
            self._chat_log.close()
        except Exception:
            pass

    def __del__(self) -> None:
        self.close()
