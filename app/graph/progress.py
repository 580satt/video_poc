"""Per-step status for the UI: pending | in_progress | complete | error"""

from __future__ import annotations

from typing import Any


def step_status_patch(updates: dict[str, str]) -> dict[str, Any]:
    """Build a Mongo $set dict using dotted keys for nested step_status."""
    return {f"step_status.{k}": v for k, v in updates.items()}
