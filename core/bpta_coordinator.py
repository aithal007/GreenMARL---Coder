"""
BPTA Coordinator — orchestrates the Coder → Gym → Debugger → Coder loop.

Hybrid BPTA implementation:

  [In-Context BPTA]
  The Debugger's structured 'bpta_delta' is injected directly into the Coder's
  (and Planner's) prompt context for the next episode, acting as a textual
  gradient signal.

  [Stub PyTorch Adapter]
  A small MLP operates on a fixed-size "state embedding" derived from the
  observation dict (pass_rate, reward, entropy, etc.).  On each step it
  produces a context_offset vector.  This demonstrates the architectural
  concept of learned backpropagation through agent state — the offset is
  used to influence the context_delta magnitude rather than actual prompt
  embeddings (which would require model internals access).

  In a full BPTA implementation this adapter would be trained with:
      loss = -V(s_{t+1})    (maximize value)
  and gradients would flow backward through the agent graph.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
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
    Tiny MLP: (state_dim,) → (context_dim,)

    Input features (state_dim = 6):
        [pass_rate, hidden_pass_rate, reward, was_sleep, syntax_error, entropy_proxy]

    Output: a context_dim-dimensional offset that modulates the delta magnitude.
    In a full system this would project into the LLM's embedding space.
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
    inference_count: int
    sleep_count: int
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
    Runs one complete episode of the GreenMARL-Coder loop.

    Sequence per step:
      1. Planner.act(obs)         → plan JSON
      2. Coder.act(obs)           → code | sleep_token
      3. Gym.step(code)           → StepResult
      4. Debugger.evaluate(...)   → DebuggerOutput
      5. Inject BPTA delta        → Coder.inject_bpta_delta(delta)
      6. StateAdapter forward     → context_offset (logged, not used for training)
      7. Compute shaped reward, log, return metrics
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
    ) -> None:
        self.gym = gym
        self.planner = planner
        self.coder = coder
        self.debugger = debugger
        self.mode = mode              # "baseline" | "multi_agent" | "full"
        self.steps_per_episode = steps_per_episode

        self.adapter = StateAdapter()
        self.adapter_optimizer = optim.Adam(self.adapter.parameters(), lr=adapter_lr)

        LOGS_DIR.mkdir(exist_ok=True)
        self._chat_log = open(CHAT_LOG, "a", encoding="utf-8")  # noqa: SIM115
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

        p_plan = self.planner.past_performance_buffer[-1] if self.planner.past_performance_buffer else 0.0
        self.planner.start_episode(self.steps_per_episode, p_plan=p_plan)

        total_reward = 0.0
        total_shaped = 0.0
        best_pass_rate = 0.0
        best_hidden_rate = 0.0
        last_debugger_out: DebuggerOutput | None = None
        last_step: StepResult | None = None

        for step_idx in range(self.steps_per_episode):
            step_result, debugger_out = self._run_step(obs, step_idx)
            last_step = step_result
            last_debugger_out = debugger_out

            total_reward += step_result.reward
            if debugger_out:
                total_shaped += debugger_out.shaped_reward
            else:
                total_shaped += step_result.reward

            best_pass_rate = max(best_pass_rate, step_result.pass_rate)
            best_hidden_rate = max(best_hidden_rate, step_result.hidden_pass_rate)

            # Update obs with new test feedback for subsequent steps
            obs.update(step_result.obs)

        # Episode end
        self.planner.end_episode(best_pass_rate)
        self.coder.observe(obs, best_pass_rate)
        self._episode_count += 1

        metrics = EpisodeMetrics(
            episode=ep,
            task_id=obs.get("task_id", "?"),
            mode=self.mode,
            steps=self.steps_per_episode,
            total_reward=total_reward,
            shaped_reward=total_shaped,
            pass_rate=best_pass_rate,
            hidden_pass_rate=best_hidden_rate,
            inference_count=self.coder.inference_count,
            sleep_count=self.coder.sleep_count,
            time_s=time.perf_counter() - t0,
            planner_generator=self.planner.generator_mode,
            value_estimate=last_debugger_out.value_estimate if last_debugger_out else 0.0,
            bpta_delta_len=len(last_debugger_out.bpta_delta) if last_debugger_out else 0,
        )
        self._log_episode_summary(metrics, last_step, last_debugger_out)
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

        # --- Planner (MARLIN) ---
        plan_json = ""
        if self.mode in ("multi_agent", "full"):
            plan_json = self.planner.act(obs)
            try:
                import json
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
            self._log("Coder", "[SLEEP] reusing last solution")
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

            # --- In-context BPTA backward pass ---
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
        2. Stub PyTorch: forward pass through StateAdapter to compute
           context_offset; backward pass minimises -V(s) (maximize value).
           The L2 norm of the offset modulates how forcefully we inject the delta.
        """
        # -- In-Context BPTA --
        self.coder.inject_bpta_delta(debugger_out.bpta_delta)

        # -- Stub PyTorch adapter --
        state_vec = self._obs_to_state_vec(step_result.obs)
        context_offset = self.adapter(state_vec)

        # Pseudo-loss: maximize value estimate → minimize -V(s)
        value_tensor = torch.tensor(
            [debugger_out.value_estimate], dtype=torch.float32
        )
        loss = -value_tensor.mean()

        self.adapter_optimizer.zero_grad()
        # We detach context_offset since it is a concept demo (no real grad path)
        loss.backward()
        self.adapter_optimizer.step()

        offset_norm = context_offset.norm().item()
        logger.debug(
            "[BPTA] In-context delta injected (%d chars). "
            "Adapter offset_norm=%.4f, pseudo_loss=%.4f",
            len(debugger_out.bpta_delta),
            offset_norm,
            loss.item(),
        )
        self._log(
            "BPTA",
            f"offset_norm={offset_norm:.4f} | delta='{debugger_out.bpta_delta[:120]}'",
        )

    @staticmethod
    def _obs_to_state_vec(obs: dict[str, Any]) -> torch.Tensor:
        """Convert observation dict to a fixed-size state vector."""
        vec = [
            float(obs.get("pass_rate", 0.0)),
            float(obs.get("hidden_pass_rate", 0.0)),
            float(obs.get("reward", 0.0)),
            float(obs.get("was_sleep", False)),
            float(obs.get("syntax_error", False)),
            # entropy proxy: 1 - pass_rate as a rough stand-in when real entropy
            # is not in the obs (it is computed inside BaseAgent)
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

    def _log_episode_summary(
        self,
        metrics: EpisodeMetrics,
        last_step: StepResult | None,
        dbg: DebuggerOutput | None,
    ) -> None:
        self._log(
            "SUMMARY",
            f"ep={metrics.episode} task={metrics.task_id} mode={metrics.mode} "
            f"reward={metrics.total_reward:.3f} pass={metrics.pass_rate:.2%} "
            f"infer={metrics.inference_count} sleep={metrics.sleep_count} "
            f"time={metrics.time_s:.1f}s planner_G={metrics.planner_generator}",
        )

    def close(self) -> None:
        self._chat_log.close()

    def __del__(self) -> None:
        try:
            self._chat_log.close()
        except Exception:
            pass
