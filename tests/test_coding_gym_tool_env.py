"""Tests for TRL/OpenEnv tool wrapper `CodingGymToolEnv`."""

from __future__ import annotations

import pytest

from greenmarl_openenv.coding_trl_env import CodingGymToolEnv


def test_reset_returns_prompt_with_task() -> None:
    env = CodingGymToolEnv()
    text = env.reset()
    assert isinstance(text, str)
    assert "task" in text.lower() or "task_" in text
    assert "submit_python_solution" in text


def test_submit_python_runs_gym() -> None:
    env = CodingGymToolEnv()
    env.reset()
    # Minimal valid two_sum-style body for task_001
    code = """
def two_sum(nums: list[int], target: int) -> list[int]:
    seen = {}
    for i, x in enumerate(nums):
        t = target - x
        if t in seen:
            return [seen[t], i]
        seen[x] = i
    return []
"""
    out = env.submit_python_solution(code)
    assert "visible_pass" in out
    assert isinstance(env.reward, float)


def test_pinned_task_id_evaluates_same_task() -> None:
    env = CodingGymToolEnv()
    text = env.reset(task_id="task_001")
    assert "task_001" in text
    code = """
def two_sum(nums: list[int], target: int) -> list[int]:
    seen = {}
    for i, x in enumerate(nums):
        t = target - x
        if t in seen:
            return [seen[t], i]
        seen[x] = i
    return []
"""
    env.submit_python_solution(code)
    assert env.reward > 0.0
