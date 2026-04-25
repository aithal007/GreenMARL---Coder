"""
Lightweight Pydantic-style data containers for the OpenEnv manifest
(action/observation spec). The live TRL tool loop uses `CodingGymToolEnv` directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CodingAction:
    """Action: submit a Python program for the current task."""

    python_code: str
    tool_name: str = "submit_python_solution"


@dataclass
class CodingObservation:
    """Post-step observation and scalar reward (mirrors `StepResult` fields)."""

    task_id: str
    message: str
    reward: float
    pass_rate: float
    hidden_pass_rate: float
    syntax_error: bool
    was_sleep: bool
    extra: dict[str, Any] = field(default_factory=dict)
