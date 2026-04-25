"""
MARLIN Planner Agent.

Implements the MARLIN (Multi-Agent Reinforcement Learning with LLM-Guided
Planning) algorithm from Algorithm 1:

    G ∈ {G_ADS, G_IAN}  — generator mode
    p = mean(pastPerformanceBuffer)

    if episode < m:              G ← G_ADS   (Adaptive Decision Search)
    elif episode < 2m:           G ← G_IAN   (Iterative Adaptive Negotiation)
    else:
        if p_plan == 1:          G ← G_IAN
        elif p < p_ILM:          G ← G_IAN
        else:                    G ← G_ADS

    if P == ∅ and G == G_IAN:   P ← makePlan()
    if step == step_max/2 and rand() <= 0.1:  G ← toggleGenerator(G)

The Planner emits a structured plan JSON that the Coder consumes as context.
"""

from __future__ import annotations

import json
import logging
import random
import re
from collections import deque
from typing import Any

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class PlannerAgent(BaseAgent):
    """MARLIN Planner: generates high-level strategy plans for the Coder."""

    SYSTEM_PROMPT_ADS = (
        "You are a senior software architect using Adaptive Decision Search. "
        "Given a coding task, output a JSON plan with keys: "
        "'strategy' (string), 'data_structures' (list), 'algorithm' (string), "
        "'edge_cases' (list), 'complexity' (dict with time and space). "
        "Be concise but precise. Output only valid JSON, no markdown fences."
    )

    SYSTEM_PROMPT_IAN = (
        "You are a senior software architect using Iterative Adaptive Negotiation. "
        "Given a coding task and optional feedback from a previous attempt, "
        "iteratively refine the plan. Output JSON with keys: "
        "'strategy' (string), 'data_structures' (list), 'algorithm' (string), "
        "'edge_cases' (list), 'complexity' (dict with time and space), "
        "'refinement_notes' (string — what changed from previous plan). "
        "Output only valid JSON, no markdown fences."
    )

    # MARLIN hyper-parameters
    M_THRESHOLD = 3          # episodes before switching from ADS to IAN
    P_ILM_THRESHOLD = 0.6    # performance threshold for G_ADS fallback
    TOGGLE_PROB = 0.1        # probability to toggle G at step_max/2
    PERF_BUFFER_SIZE = 10    # rolling window for performance buffer

    def __init__(self, model_name: str = BaseAgent.DEFAULT_MODEL, **kwargs: Any) -> None:
        super().__init__(name="Planner", model_name=model_name, **kwargs)

        self.episode: int = 0
        self.step: int = 0
        self.step_max: int = 1  # updated at episode start
        self.generator_mode: str = "ADS"  # "ADS" | "IAN"
        self.current_plan: dict | None = None
        self.past_performance_buffer: deque[float] = deque(maxlen=self.PERF_BUFFER_SIZE)
        self.last_obs: dict | None = None
        self.previous_plan: dict | None = None  # for IAN refinement

    # ------------------------------------------------------------------
    # MARLIN algorithm step
    # ------------------------------------------------------------------

    def marlin_select_generator(self, p_plan: float) -> str:
        """
        Implements the generator selection logic from Algorithm 1 line 8-13.

        p_plan: pass_rate of the current plan on the last eval (0.0–1.0).
        Returns: "ADS" or "IAN"
        """
        p = (
            sum(self.past_performance_buffer) / len(self.past_performance_buffer)
            if self.past_performance_buffer
            else 0.0
        )

        if self.episode < self.M_THRESHOLD:
            mode = "ADS"
        elif self.episode < 2 * self.M_THRESHOLD:
            mode = "IAN"
        else:
            if p_plan >= 1.0:
                mode = "IAN"
            elif p < self.P_ILM_THRESHOLD:
                mode = "IAN"
            else:
                mode = "ADS"

        return mode

    def maybe_toggle(self) -> None:
        """Algorithm 1 line 16-17: toggle generator at step_max/2 with prob 0.1."""
        if self.step == self.step_max // 2 and random.random() <= self.TOGGLE_PROB:
            old = self.generator_mode
            self.generator_mode = "IAN" if self.generator_mode == "ADS" else "ADS"
            logger.info(
                "[Planner] Toggled generator %s -> %s at step %d",
                old, self.generator_mode, self.step,
            )

    def start_episode(self, step_max: int, p_plan: float = 0.0) -> None:
        """Called at the beginning of each episode to update MARLIN state."""
        self.step = 0
        self.step_max = step_max
        self.generator_mode = self.marlin_select_generator(p_plan)
        self.current_plan = None
        logger.info(
            "[Planner] Episode %d | generator=%s | perf_mean=%.3f",
            self.episode,
            self.generator_mode,
            sum(self.past_performance_buffer) / max(len(self.past_performance_buffer), 1),
        )

    def end_episode(self, reward: float) -> None:
        """Called at episode end to update performance buffer."""
        self.past_performance_buffer.append(reward)
        self.episode += 1

    # ------------------------------------------------------------------
    # Plan generation
    # ------------------------------------------------------------------

    def make_plan(self, observation: dict[str, Any]) -> dict:
        """
        Generate a structured plan for the task in `observation`.
        Uses ADS or IAN prompt depending on current generator mode.
        """
        task_desc = observation.get("description", "")
        signature = observation.get("function_signature", "")
        examples = observation.get("examples", [])

        user_parts = [
            f"Task: {task_desc}",
            f"Function signature: {signature}",
        ]
        if examples:
            ex_str = "\n".join(
                f"  Input: {e['input']} -> Output: {e['output']}" for e in examples[:2]
            )
            user_parts.append(f"Examples:\n{ex_str}")

        if self.generator_mode == "IAN" and self.previous_plan:
            user_parts.append(
                f"Previous plan (refine it):\n{json.dumps(self.previous_plan, indent=2)}"
            )

        system = (
            self.SYSTEM_PROMPT_IAN
            if self.generator_mode == "IAN"
            else self.SYSTEM_PROMPT_ADS
        )
        prompt = self.build_chat_prompt(system, "\n".join(user_parts))
        raw, _ = self.generate(prompt, max_new_tokens=300)

        plan = self._parse_plan(raw)
        self.previous_plan = plan
        self.current_plan = plan
        return plan

    @staticmethod
    def _parse_plan(raw: str) -> dict:
        """Extract JSON from model output, falling back to a minimal default plan."""
        # strip markdown fences if present
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("[Planner] Could not parse plan JSON, using fallback.")
            return {
                "strategy": raw[:200] if raw else "iterative approach",
                "data_structures": ["dict", "list"],
                "algorithm": "brute-force",
                "edge_cases": ["empty input", "single element"],
                "complexity": {"time": "O(n)", "space": "O(n)"},
            }

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def act(self, observation: dict[str, Any]) -> str:
        """
        Main entry point. Returns a JSON string of the plan.
        If no plan exists and mode is IAN (or ADS first time), creates one.
        """
        self.last_obs = observation
        self.maybe_toggle()
        self.step += 1

        # Algorithm 1 line 14: if P == ∅ and G == G_IAN → makePlan()
        # Also make a plan on step 1 regardless.
        if self.current_plan is None or self.step == 1:
            plan = self.make_plan(observation)
        else:
            plan = self.current_plan

        return json.dumps(plan, indent=2)

    def observe(self, obs: dict[str, Any], reward: float) -> None:
        super().observe(obs, reward)
        logger.debug("[Planner] Observed reward=%.3f", reward)
