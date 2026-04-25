"""
Tests for the Coding Gym sandbox.
Run with: pytest tests/test_gym.py -v
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from env.coding_gym import CodingGym, StepResult


@pytest.fixture
def gym():
    return CodingGym()


# ---------------------------------------------------------------------------
# Basic gym structure
# ---------------------------------------------------------------------------

def test_gym_loads_tasks(gym):
    assert len(gym.tasks) >= 3


def test_reset_returns_obs(gym):
    obs = gym.reset()
    assert "task_id" in obs
    assert "description" in obs
    assert "function_signature" in obs


def test_advance_cycles_tasks(gym):
    id1 = gym.reset()["task_id"]
    gym.advance()
    id2 = gym.reset()["task_id"]
    assert id1 != id2


# ---------------------------------------------------------------------------
# Sleep token
# ---------------------------------------------------------------------------

def test_sleep_token_returns_negative_reward(gym):
    gym.reset()
    result = gym.step("sleep_token")
    assert isinstance(result, StepResult)
    assert result.was_sleep is True
    assert result.reward < 0


# ---------------------------------------------------------------------------
# Code evaluation
# ---------------------------------------------------------------------------

def test_correct_two_sum(gym):
    gym._task_index = 0  # two_sum task
    result = gym.step(
        "def two_sum(nums, target):\n"
        "    seen = {}\n"
        "    for i, n in enumerate(nums):\n"
        "        if target - n in seen:\n"
        "            return [seen[target - n], i]\n"
        "        seen[n] = i\n",
        task_id="task_001",
    )
    assert result.pass_rate == 1.0
    assert result.reward > 0.9


def test_syntax_error_penalised(gym):
    result = gym.step("def two_sum(nums target:\n    pass", task_id="task_001")
    assert result.syntax_error is True
    assert result.reward < 0


def test_wrong_answer_low_reward(gym):
    result = gym.step(
        "def two_sum(nums, target):\n    return [0, 1]",  # always wrong
        task_id="task_001",
    )
    # might pass the very first test case by coincidence on [2,7,11,15], 9
    # but should fail others
    assert result.reward <= 1.0


def test_valid_parentheses_correct(gym):
    result = gym.step(
        "def is_valid(s):\n"
        "    stack = []\n"
        "    m = {')': '(', '}': '{', ']': '['}\n"
        "    for c in s:\n"
        "        if c in '([{':\n"
        "            stack.append(c)\n"
        "        elif stack and stack[-1] == m[c]:\n"
        "            stack.pop()\n"
        "        else:\n"
        "            return False\n"
        "    return not stack\n",
        task_id="task_002",
    )
    assert result.pass_rate == 1.0


# ---------------------------------------------------------------------------
# Reward bounds
# ---------------------------------------------------------------------------

def test_reward_clamped():
    gym = CodingGym()
    # A perfect solution should give reward <= 1.5
    result = gym.step(
        "def two_sum(nums, target):\n"
        "    seen = {}\n"
        "    for i, n in enumerate(nums):\n"
        "        if target - n in seen:\n"
        "            return [seen[target - n], i]\n"
        "        seen[n] = i\n",
        task_id="task_001",
    )
    assert -1.0 <= result.reward <= 1.5
