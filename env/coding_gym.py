"""
Coding Gym — sandboxed evaluation environment for GreenMARL-Coder.

Runs generated Python code in a subprocess with a timeout, executes
test cases against it, and returns a structured observation dict plus
a scalar reward.

Reward formula:
    reward = pass_rate * 1.0
             + hidden_pass_rate * 0.5   (bonus for edge cases)
             - time_penalty             (0.1 if near timeout)
             - sleep_cost               (small negative if sleep_token used)
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TASKS_PATH = Path(__file__).parent / "tasks.json"
SUBPROCESS_TIMEOUT = 5  # hard wall-clock limit per test run (seconds)


@dataclass
class StepResult:
    task_id: str
    code: str
    stdout: str
    stderr: str
    passed_visible: int
    total_visible: int
    passed_hidden: int
    total_hidden: int
    syntax_error: bool
    timed_out: bool
    was_sleep: bool
    elapsed_ms: float
    reward: float
    obs: dict[str, Any] = field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        if self.total_visible == 0:
            return 0.0
        return self.passed_visible / self.total_visible

    @property
    def hidden_pass_rate(self) -> float:
        if self.total_hidden == 0:
            return 0.0
        return self.passed_hidden / self.total_hidden

    def summary(self) -> str:
        return (
            f"[{self.task_id}] visible={self.passed_visible}/{self.total_visible} "
            f"hidden={self.passed_hidden}/{self.total_hidden} "
            f"reward={self.reward:.3f} sleep={self.was_sleep} "
            f"elapsed={self.elapsed_ms:.0f}ms"
        )


class CodingGym:
    """
    Environment that presents coding tasks, evaluates submitted code,
    and returns observations + rewards compatible with the MARL loop.
    """

    def __init__(self, tasks_path: Path = TASKS_PATH) -> None:
        with open(tasks_path) as f:
            data = json.load(f)
        self.tasks: list[dict] = data["tasks"]
        self._task_index: int = 0
        self._episode_rewards: list[float] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self) -> dict[str, Any]:
        """Start a fresh episode on the next task (cycles through tasks)."""
        self._task_index = self._task_index % len(self.tasks)
        task = self.tasks[self._task_index]
        return self._task_obs(task)

    def step(
        self,
        code_or_sleep: str,
        task_id: str | None = None,
    ) -> StepResult:
        """
        Evaluate submitted code (or "sleep_token") against the current task.

        Args:
            code_or_sleep: Python source OR the literal string "sleep_token".
            task_id: Override which task to evaluate against.
        """
        task = self._resolve_task(task_id)
        was_sleep = code_or_sleep.strip() == "sleep_token"

        if was_sleep:
            result = StepResult(
                task_id=task["id"],
                code="sleep_token",
                stdout="",
                stderr="",
                passed_visible=0,
                total_visible=len(task.get("test_cases", [])),
                passed_hidden=0,
                total_hidden=len(task.get("hidden_assertions", [])),
                syntax_error=False,
                timed_out=False,
                was_sleep=True,
                elapsed_ms=0.0,
                reward=-0.05,
            )
            result.obs = self._build_obs(result, task)
            return result

        return self._evaluate(task, code_or_sleep)

    def advance(self) -> None:
        """Move to the next task in the list."""
        self._task_index = (self._task_index + 1) % len(self.tasks)

    def current_task(self) -> dict:
        return self.tasks[self._task_index % len(self.tasks)]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_task(self, task_id: str | None) -> dict:
        if task_id is None:
            return self.tasks[self._task_index % len(self.tasks)]
        for t in self.tasks:
            if t["id"] == task_id:
                return t
        raise ValueError(f"Unknown task_id: {task_id}")

    def _task_obs(self, task: dict) -> dict[str, Any]:
        return {
            "task_id": task["id"],
            "name": task["name"],
            "difficulty": task["difficulty"],
            "description": task["description"],
            "function_signature": task["function_signature"],
            "examples": task.get("examples", []),
        }

    def _evaluate(self, task: dict, code: str) -> StepResult:
        test_cases = task.get("test_cases", [])
        hidden = task.get("hidden_assertions", [])
        time_limit_ms = task.get("time_limit_ms", 1000)

        # --- syntax check ------------------------------------------------
        try:
            compile(code, "<generated>", "exec")
        except SyntaxError as exc:
            result = StepResult(
                task_id=task["id"],
                code=code,
                stdout="",
                stderr=f"SyntaxError: {exc}",
                passed_visible=0,
                total_visible=len(test_cases),
                passed_hidden=0,
                total_hidden=len(hidden),
                syntax_error=True,
                timed_out=False,
                was_sleep=False,
                elapsed_ms=0.0,
                reward=-0.3,
            )
            result.obs = self._build_obs(result, task)
            return result

        # --- run visible tests -------------------------------------------
        t0 = time.perf_counter()
        vis_pass, vis_stdout, vis_stderr, timed_out = self._run_cases(
            code, test_cases, task, time_limit_ms
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # --- run hidden tests (only if visible passed) --------------------
        hid_pass = 0
        if vis_pass > 0 and not timed_out:
            hid_pass, _, _, _ = self._run_cases(code, hidden, task, time_limit_ms)

        # --- reward computation ------------------------------------------
        pass_rate = vis_pass / max(len(test_cases), 1)
        hidden_rate = hid_pass / max(len(hidden), 1)
        time_penalty = 0.1 if elapsed_ms > time_limit_ms * 0.8 else 0.0
        timeout_penalty = -0.4 if timed_out else 0.0
        reward = pass_rate * 1.0 + hidden_rate * 0.5 - time_penalty + timeout_penalty
        reward = max(-1.0, min(1.5, reward))

        result = StepResult(
            task_id=task["id"],
            code=code,
            stdout=vis_stdout,
            stderr=vis_stderr,
            passed_visible=vis_pass,
            total_visible=len(test_cases),
            passed_hidden=hid_pass,
            total_hidden=len(hidden),
            syntax_error=False,
            timed_out=timed_out,
            was_sleep=False,
            elapsed_ms=elapsed_ms,
            reward=reward,
        )
        result.obs = self._build_obs(result, task)
        return result

    def _run_cases(
        self,
        code: str,
        cases: list[dict],
        task: dict,
        time_limit_ms: int,
    ) -> tuple[int, str, str, bool]:
        """Execute test cases in a subprocess; return (passed, stdout, stderr, timed_out)."""
        if not cases:
            return 0, "", "", False

        harness = self._build_harness(code, cases, task)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(harness)
            tmp_path = f.name

        try:
            proc = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=SUBPROCESS_TIMEOUT,
            )
            stdout = proc.stdout.strip()
            stderr = proc.stderr.strip()
            timed_out = False
        except subprocess.TimeoutExpired:
            stdout = ""
            stderr = "TimeoutExpired"
            timed_out = True
        finally:
            os.unlink(tmp_path)

        passed = stdout.count("PASS")
        return passed, stdout, stderr, timed_out

    def _build_harness(self, code: str, cases: list[dict], task: dict) -> str:
        """Build a self-contained test harness string."""
        func_name = self._extract_func_name(task["function_signature"])
        lines = [code, ""]

        for i, case in enumerate(cases):
            args = self._case_to_args(case, task["id"])
            expected = case.get("expected")

            if task["id"] == "task_005":
                # group_anagrams: compare sorted group sizes
                check = (
                    f"_result_{i} = {func_name}({args})\n"
                    f"_sizes_{i} = sorted([len(g) for g in _result_{i}], reverse=True)\n"
                    f"_exp_{i} = sorted({case['expected_sizes']}, reverse=True)\n"
                    f"print('PASS' if _sizes_{i} == _exp_{i} else 'FAIL: got ' + str(_sizes_{i}))"
                )
            else:
                check = (
                    f"_result_{i} = {func_name}({args})\n"
                    f"print('PASS' if _result_{i} == {repr(expected)} else 'FAIL: got ' + repr(_result_{i}))"
                )

            lines.append(textwrap.dedent(f"""
try:
{textwrap.indent(check, '    ')}
except Exception as _e:
    print('FAIL: exception ' + str(_e))
"""))

        return "\n".join(lines)

    @staticmethod
    def _extract_func_name(signature: str) -> str:
        m = re.search(r"def (\w+)\(", signature)
        return m.group(1) if m else "solution"

    @staticmethod
    def _case_to_args(case: dict, task_id: str) -> str:
        """Convert a test case dict into a positional argument string."""
        skip_keys = {"expected", "expected_sizes"}
        parts = []
        for k, v in case.items():
            if k in skip_keys:
                continue
            parts.append(f"{k}={repr(v)}")
        return ", ".join(parts)

    @staticmethod
    def _build_obs(result: StepResult, task: dict) -> dict[str, Any]:
        return {
            "task_id": result.task_id,
            "pass_rate": result.pass_rate,
            "hidden_pass_rate": result.hidden_pass_rate,
            "reward": result.reward,
            "syntax_error": result.syntax_error,
            "timed_out": result.timed_out,
            "was_sleep": result.was_sleep,
            "elapsed_ms": result.elapsed_ms,
            "stdout_snippet": result.stdout[:500],
            "stderr_snippet": result.stderr[:500],
            "task_description": task["description"],
            "function_signature": task["function_signature"],
        }
