"""
ETD-MAPPO Coder Agent.

Implements Entropy-Threshold Decentralized (ETD) gating for Proximal Policy
Optimization in a multi-agent setting:

  Before decoding a full solution:
    1. Probe first ENTROPY_PROBE_TOKENS token logprobs.
    2. Compute Shannon entropy H over the token distribution.
    3. If H < ENTROPY_THRESHOLD AND past rewards are stable → sleep_token.
       (Agent "sleeps": reuses previous solution, saving an inference call.)
    4. Otherwise, generate a full Python solution guided by the Planner's plan.

The BPTA backward pass from the Debugger is injected into `context_buffer`
as a structured delta that shifts the prompt for the next episode.
"""

from __future__ import annotations

import logging
import re
from collections import deque
from typing import Any

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class CoderAgent(BaseAgent):
    """
    ETD-MAPPO Coder: generates Python solutions with entropy-gated execution.
    """

    SYSTEM_PROMPT = (
        "You are an expert Python programmer. "
        "Given a coding task and a high-level plan from the Planner, "
        "write a complete, correct Python function that solves the task. "
        "Output ONLY the Python function — no explanations, no markdown fences, "
        "no test code. The function must exactly match the given signature."
    )

    def __init__(
        self,
        model_name: str = BaseAgent.DEFAULT_MODEL,
        etd_enabled: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(name="Coder", model_name=model_name, **kwargs)
        self.etd_enabled = etd_enabled

        # Rolling reward history for ETD stability check
        self.past_rewards: deque[float] = deque(maxlen=8)
        # Previous solution for sleep reuse
        self._last_solution: str = ""
        # BPTA in-context delta: injected by BPTACoordinator after each episode
        self.context_delta: str = ""
        # Planner's current plan (injected by coordinator)
        self.current_plan: dict | None = None

        self.inference_count: int = 0
        self.sleep_count: int = 0

    # ------------------------------------------------------------------
    # ETD gating
    # ------------------------------------------------------------------

    def _build_prompt(self, observation: dict[str, Any]) -> str:
        task_desc = observation.get("description", "")
        signature = observation.get("function_signature", "")
        examples = observation.get("examples", [])

        user_parts = [f"Task: {task_desc}", f"Function signature: {signature}"]

        if examples:
            ex_str = "\n".join(
                f"  {e['input']} -> {e['output']}" for e in examples[:2]
            )
            user_parts.append(f"Examples:\n{ex_str}")

        if self.current_plan:
            import json
            plan_str = json.dumps(self.current_plan, indent=2)
            user_parts.append(f"Planner's strategy:\n{plan_str}")

        if self.context_delta:
            user_parts.append(
                f"[BPTA Feedback from Debugger — incorporate this]\n{self.context_delta}"
            )

        return self.build_chat_prompt(self.SYSTEM_PROMPT, "\n\n".join(user_parts))

    def act(self, observation: dict[str, Any]) -> str:
        """
        ETD-gated action.

        Returns either:
          - "sleep_token"  — reuse previous solution (entropy low + rewards stable)
          - Python source code for the current task
        """
        prompt = self._build_prompt(observation)
        past_rewards_list = list(self.past_rewards)

        if self.etd_enabled:
            entropy = self.calculate_entropy(prompt)
            stable = self._rewards_are_stable(past_rewards_list)
            logger.debug(
                "[Coder] ETD entropy=%.3f stable=%s past_rewards=%s",
                entropy, stable, past_rewards_list,
            )
            if entropy < self.ENTROPY_THRESHOLD and stable and self._last_solution:
                self.sleep_count += 1
                logger.info(
                    "[Coder] ETD SLEEP (entropy=%.3f, stable=%s). Reusing solution.",
                    entropy, stable,
                )
                return "sleep_token"

        # Full generation
        self.inference_count += 1
        text, _ = self.generate(prompt)
        solution = self._extract_code(text, observation.get("function_signature", ""))
        if solution:
            self._last_solution = solution
        return solution or self._last_solution or text

    def get_last_solution(self) -> str:
        """Return the previous solution (used when sleeping)."""
        return self._last_solution

    # ------------------------------------------------------------------
    # BPTA feedback injection
    # ------------------------------------------------------------------

    def inject_bpta_delta(self, delta: str) -> None:
        """
        Receive the in-context BPTA delta from the coordinator and store it
        for inclusion in the next act() prompt.
        """
        self.context_delta = delta
        logger.debug("[Coder] BPTA delta injected (%d chars)", len(delta))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_code(text: str, signature: str) -> str:
        """
        Pull the function definition out of the model's raw output.
        Strips markdown fences and any surrounding text.
        """
        # Remove markdown code fences
        text = re.sub(r"```(?:python)?", "", text).strip().rstrip("`").strip()

        # Try to extract just the def block
        func_match = re.search(r"(def \w+\(.*?)(?=\ndef |\Z)", text, re.DOTALL)
        if func_match:
            return func_match.group(1).strip()

        # Fallback: return everything if it starts with 'def'
        if text.lstrip().startswith("def "):
            return text.strip()

        return text.strip()

    def observe(self, obs: dict[str, Any], reward: float) -> None:
        super().observe(obs, reward)
        self.past_rewards.append(reward)

    @property
    def efficiency_ratio(self) -> float:
        """Fraction of steps saved by ETD gating."""
        total = self.inference_count + self.sleep_count
        if total == 0:
            return 0.0
        return self.sleep_count / total
