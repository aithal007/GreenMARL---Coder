"""
Tests for ETD (Entropy-Threshold Decentralized) gating in the Coder agent.

These tests mock the LLM generation so they run without a real model,
focusing on the gating logic itself.
Run with: pytest tests/test_etd_gating.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from agents.coder import CoderAgent


# ---------------------------------------------------------------------------
# Fixture: CoderAgent with mocked model
# ---------------------------------------------------------------------------

def _make_coder(etd_enabled: bool = True) -> CoderAgent:
    coder = CoderAgent.__new__(CoderAgent)
    # Manually initialise without calling __init__ to avoid HF model load
    coder.name = "Coder"
    coder.model_name = "mock"
    coder.device = "cpu"
    coder.max_new_tokens = 512
    coder.temperature = 0.0
    coder.etd_enabled = etd_enabled
    coder.ENTROPY_THRESHOLD = 1.5
    coder.ENTROPY_PROBE_TOKENS = 8
    coder.REWARD_STABILITY_WINDOW = 3
    coder._model = None
    coder._tokenizer = None
    coder._loaded = False
    coder.obs_history = []
    coder.reward_history = __import__("collections").deque(maxlen=10)
    from collections import deque
    coder.past_rewards = deque(maxlen=8)
    coder._last_solution = "def solution(): return 42"
    coder.context_delta = ""
    coder.current_plan = None
    coder.inference_count = 0
    coder.sleep_count = 0
    return coder


SAMPLE_OBS = {
    "task_id": "task_001",
    "description": "Two sum problem",
    "function_signature": "def two_sum(nums, target):",
    "examples": [],
}

GOOD_SOLUTION = (
    "def two_sum(nums, target):\n"
    "    seen = {}\n"
    "    for i, n in enumerate(nums):\n"
    "        if target - n in seen:\n"
    "            return [seen[target - n], i]\n"
    "        seen[n] = i\n"
)


# ---------------------------------------------------------------------------
# ETD gating: should sleep when entropy is low + rewards are stable
# ---------------------------------------------------------------------------

def test_etd_sleeps_on_low_entropy_stable_rewards():
    """When entropy < threshold and rewards are stable, agent returns sleep_token."""
    coder = _make_coder(etd_enabled=True)
    # Fill reward history with high, stable rewards
    for _ in range(5):
        coder.past_rewards.append(0.9)
    coder._last_solution = GOOD_SOLUTION

    # Patch entropy to return a sub-threshold value
    with patch.object(coder, "calculate_entropy", return_value=0.5):
        result = coder.act(SAMPLE_OBS)

    assert result == "sleep_token"
    assert coder.sleep_count == 1
    assert coder.inference_count == 0


def test_etd_acts_on_high_entropy():
    """When entropy >= threshold, agent generates even if rewards are stable."""
    coder = _make_coder(etd_enabled=True)
    for _ in range(5):
        coder.past_rewards.append(0.9)
    coder._last_solution = GOOD_SOLUTION

    # Mock generate to return a fixed solution
    with patch.object(coder, "calculate_entropy", return_value=3.5), \
         patch.object(coder, "generate", return_value=(GOOD_SOLUTION, None)):
        result = coder.act(SAMPLE_OBS)

    assert result != "sleep_token"
    assert coder.inference_count == 1
    assert coder.sleep_count == 0


def test_etd_acts_on_unstable_rewards():
    """When rewards are unstable (high variance), ETD forces a new generation."""
    coder = _make_coder(etd_enabled=True)
    # Alternating rewards = high variance
    for v in [0.1, 0.9, 0.1, 0.9, 0.1]:
        coder.past_rewards.append(v)
    coder._last_solution = GOOD_SOLUTION

    with patch.object(coder, "calculate_entropy", return_value=0.3), \
         patch.object(coder, "generate", return_value=(GOOD_SOLUTION, None)):
        result = coder.act(SAMPLE_OBS)

    assert result != "sleep_token"
    assert coder.inference_count == 1


def test_etd_disabled_always_generates():
    """With ETD disabled (baseline/multi-agent), agent always calls generate()."""
    coder = _make_coder(etd_enabled=False)
    for _ in range(5):
        coder.past_rewards.append(1.0)
    coder._last_solution = GOOD_SOLUTION

    with patch.object(coder, "generate", return_value=(GOOD_SOLUTION, None)) as mock_gen:
        result = coder.act(SAMPLE_OBS)

    mock_gen.assert_called_once()
    assert result != "sleep_token"
    assert coder.inference_count == 1


def test_etd_skips_sleep_when_no_previous_solution():
    """Even with low entropy + stable rewards, must act if no previous solution exists."""
    coder = _make_coder(etd_enabled=True)
    for _ in range(5):
        coder.past_rewards.append(0.9)
    coder._last_solution = ""  # no previous solution

    with patch.object(coder, "calculate_entropy", return_value=0.1), \
         patch.object(coder, "generate", return_value=(GOOD_SOLUTION, None)):
        result = coder.act(SAMPLE_OBS)

    assert result != "sleep_token"


# ---------------------------------------------------------------------------
# BPTA delta injection
# ---------------------------------------------------------------------------

def test_bpta_delta_injected_into_prompt():
    """BPTA delta should appear in the prompt built for the next act() call."""
    coder = _make_coder(etd_enabled=False)
    coder.inject_bpta_delta("Use a hash map to solve this in O(n).")

    # capture the prompt via generate mock
    captured_prompts = []

    def fake_generate(prompt, **kwargs):
        captured_prompts.append(prompt)
        return GOOD_SOLUTION, None

    with patch.object(coder, "generate", side_effect=fake_generate):
        coder.act(SAMPLE_OBS)

    assert captured_prompts, "generate() was never called"
    assert "hash map" in captured_prompts[0] or "BPTA" in captured_prompts[0]


# ---------------------------------------------------------------------------
# Efficiency ratio
# ---------------------------------------------------------------------------

def test_efficiency_ratio_computed_correctly():
    coder = _make_coder(etd_enabled=True)
    coder.inference_count = 7
    coder.sleep_count = 3
    assert abs(coder.efficiency_ratio - 0.3) < 1e-6


def test_efficiency_ratio_zero_when_no_steps():
    coder = _make_coder()
    assert coder.efficiency_ratio == 0.0
