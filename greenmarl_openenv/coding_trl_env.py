"""
TRL/OpenEnv-style tool environment wrapping `env.coding_gym.CodingGym`.

Compatible with `trl.GRPOTrainer(..., environment_factory=CodingGymToolEnv)`.
The model calls `submit_python_solution` with a full function body; the gym
executes it and sets `self.reward` for the GRPO reward function.

Reserved names: do not use `reset`, `step`, `state`, `close` as tool names
(OpenEnv client/server convention).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Project root on `sys.path` (training scripts add parent)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


class CodingGymToolEnv:
    """
    One episode = one `CodingGym` task. The agent must call `submit_python_solution`
    with Python source. Reward is the gym's scalar (tests + hidden + penalties).
    """

    def __init__(self) -> None:
        from env.coding_gym import CodingGym  # local import

        self.gym = CodingGym()
        self.reward: float = 0.0
        self._last_result_summary: str = ""
        self._active_task_id: str | None = None

    def reset(self, **kwargs) -> str | None:
        """Start a new task episode; kwargs may include `task_id` to pin a task."""
        self.reward = 0.0
        self._last_result_summary = ""
        task_id = kwargs.get("task_id")
        if task_id:
            t = self.gym._resolve_task(str(task_id))
            self._active_task_id = t["id"]
            obs = self.gym._task_obs(t)
        else:
            self._active_task_id = None
            obs = self.gym.reset()
        lines = [
            f"You are in a code submission environment (task {obs.get('task_id', '?')}).",
            f"**Description:** {obs.get('description', '')[:4000]}",
            f"**Function signature (required):** {obs.get('function_signature', '')}",
            "Call the tool `submit_python_solution` with the full function implementation.",
        ]
        ex = obs.get("examples") or []
        if ex:
            lines.append("**Examples:**")
            for e in ex[:3]:
                lines.append(f"  {e.get('input', '')!s}  ->  {e.get('output', '')!s}")
        return "\n".join(lines)

    def submit_python_solution(self, python_code: str) -> str:
        """
        Submit a Python function body for the current task. The file should define
        the function whose signature was given in the task description.

        Args:
            python_code: Source code to execute in the coding sandbox (visible + hidden tests).

        Returns:
            A short string with pass rates, reward, and stderr head (for the model to read).
        """
        r = self.gym.step(
            python_code.strip() if python_code else "",
            task_id=self._active_task_id,
        )
        self.reward = float(r.reward)
        self._last_result_summary = r.summary()
        err = (r.stderr or "")[:800]
        return (
            f"{r.summary()}\n"
            f"visible_pass={r.passed_visible}/{r.total_visible} "
            f"hidden_pass={r.passed_hidden}/{r.total_hidden}\n"
            f"stderr_head:\n{err if err else '<empty>'}\n"
        )
