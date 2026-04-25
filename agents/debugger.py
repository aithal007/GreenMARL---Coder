"""
BPTA Debugger / Critic Agent.

Implements the Critic side of Bidirectional Policy Training with Augmented
rewards (BPTA):

  Forward pass:  Coder generates code → Gym evaluates → Debugger observes results.
  Backward pass: Debugger generates:
    1. A value estimate V(s) ∈ [-1, 1] for the current state.
    2. A reward shaping signal Δr added to the raw gym reward.
    3. A structured critique used by BPTACoordinator to form the in-context
       delta injected back into Coder and Planner prompts.

The "backward pass" is in-context (textual), not literal automatic
differentiation.  A stub PyTorch adapter in BPTACoordinator converts the
textual signal into a latent delta for architectural demonstration purposes.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


@dataclass
class DebuggerOutput:
    """Structured output from one Debugger forward pass."""
    value_estimate: float          # V(s) ∈ [-1, 1]
    reward_shaping: float          # Δr added to raw reward
    shaped_reward: float           # raw_reward + Δr
    critique: str                  # natural-language critique
    bpta_delta: str                # structured delta for in-context backprop
    raw_output: str                # full model response (for logging)


class DebuggerAgent(BaseAgent):
    """
    BPTA Critic: interprets sandbox results and generates value estimates +
    reward shaping signals + structured BPTA feedback.
    """

    SYSTEM_PROMPT = (
        "You are a senior code reviewer and RL critic agent. "
        "Given a coding task, the submitted Python code, and the test results, "
        "produce a JSON response with the following keys:\n"
        "  'value_estimate': float in [-1, 1] estimating how good the current state is.\n"
        "  'reward_shaping': float in [-0.3, 0.3] to add to the raw reward "
        "(positive if the code is partially correct or well-structured, "
        "negative if fundamentally flawed).\n"
        "  'critique': string — concise diagnosis of what went wrong or right.\n"
        "  'bpta_delta': string — specific actionable instructions for the Coder "
        "to fix the issues in the next attempt. Be concrete: name the exact bug, "
        "suggest the fix, mention any edge cases missed.\n"
        "Output only valid JSON, no markdown fences."
    )

    def __init__(
        self,
        model_name: str = BaseAgent.DEFAULT_MODEL,
        **kwargs: Any,
    ) -> None:
        super().__init__(name="Debugger", model_name=model_name, **kwargs)
        self.last_output: DebuggerOutput | None = None

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def act(self, observation: dict[str, Any]) -> str:
        """
        Produce a critique given the current observation (which includes
        code submitted, test results, and stdout/stderr from the gym).
        Returns the raw JSON string from the model.
        """
        output = self.evaluate(observation, observation.get("reward", 0.0))
        return output.bpta_delta

    # ------------------------------------------------------------------
    # Critic evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        observation: dict[str, Any],
        raw_reward: float,
        code: str = "",
    ) -> DebuggerOutput:
        """
        Full critic evaluation.  Called by BPTACoordinator after each gym step.

        Args:
            observation: gym observation dict (includes pass_rate, stderr, etc.)
            raw_reward: scalar reward from the gym step.
            code: the actual code submitted (for context in the prompt).

        Returns:
            DebuggerOutput with value estimate, shaped reward, and BPTA delta.
        """
        user_content = self._build_user_prompt(observation, raw_reward, code)
        prompt = self.build_chat_prompt(self.SYSTEM_PROMPT, user_content)

        raw, _ = self.generate(prompt, max_new_tokens=400)
        result = self._parse_output(raw, raw_reward)
        self.last_output = result

        logger.info(
            "[Debugger] V(s)=%.3f Δr=%.3f shaped_r=%.3f | %s",
            result.value_estimate,
            result.reward_shaping,
            result.shaped_reward,
            result.critique[:80],
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_user_prompt(
        obs: dict[str, Any],
        raw_reward: float,
        code: str,
    ) -> str:
        parts = [
            f"Task: {obs.get('task_description', 'unknown')}",
            f"Function signature: {obs.get('function_signature', '')}",
            f"Raw gym reward: {raw_reward:.4f}",
            f"Visible test pass rate: {obs.get('pass_rate', 0.0):.2%}",
            f"Hidden test pass rate: {obs.get('hidden_pass_rate', 0.0):.2%}",
        ]
        if obs.get("syntax_error"):
            parts.append("Syntax error detected in submitted code.")
        if obs.get("timed_out"):
            parts.append("Code execution timed out.")
        if obs.get("was_sleep"):
            parts.append("Coder used sleep_token (reused previous solution).")
        if obs.get("stderr_snippet"):
            parts.append(f"Stderr:\n{obs['stderr_snippet'][:300]}")
        if obs.get("stdout_snippet"):
            parts.append(f"Test output:\n{obs['stdout_snippet'][:300]}")
        if code and code != "sleep_token":
            parts.append(f"Submitted code:\n{code[:600]}")

        return "\n\n".join(parts)

    @staticmethod
    def _parse_output(raw: str, raw_reward: float) -> DebuggerOutput:
        """Parse model JSON output; fall back to heuristic estimates on failure."""
        clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        try:
            data = json.loads(clean)
            ve = float(data.get("value_estimate", 0.0))
            rs = float(data.get("reward_shaping", 0.0))
            critique = str(data.get("critique", ""))
            delta = str(data.get("bpta_delta", ""))
        except (json.JSONDecodeError, TypeError, ValueError):
            logger.warning("[Debugger] Could not parse output JSON; using heuristics.")
            ve = max(-1.0, min(1.0, raw_reward))
            rs = 0.0
            critique = raw[:200]
            delta = raw[:400]

        ve = max(-1.0, min(1.0, ve))
        rs = max(-0.3, min(0.3, rs))

        return DebuggerOutput(
            value_estimate=ve,
            reward_shaping=rs,
            shaped_reward=raw_reward + rs,
            critique=critique,
            bpta_delta=delta,
            raw_output=raw,
        )
