from __future__ import annotations

from typing import List, Optional

from app.models import InvoiceReward, InvoiceState


# reward weights — tweak these if you want to change how the agent is incentivised
_ISSUE_REWARDS = {
    "missing_gst_number":      0.20,
    "wrong_total_calculation": 0.20,
    "duplicate_invoice":       0.20,
    "missing_receipt":         0.10,
    "missing_vendor_name":     0.10,
}

_CORRECT_ACTION_BONUS = 0.20
_WRONG_ACTION_PENALTY = -0.50
_UNNECESSARY_STEP     = -0.05
_TOO_LONG_PENALTY     = -0.10
_GRACE_BUFFER         = 2


def compute_step_reward(
    state: InvoiceState,
    action_type: str,
    value: Optional[str],
    action_was_useful: bool,
) -> InvoiceReward:
    """
    Reward for a single step. Only add_issue actions give non-zero step reward.
    Final-decision actions (mark_invalid etc.) are scored at episode end.
    """
    breakdown: dict[str, float] = {}
    total = 0.0

    if action_type == "add_issue" and value:
        if not action_was_useful:
            # agent already recorded this issue
            breakdown["redundant_issue"] = _UNNECESSARY_STEP
            total += _UNNECESSARY_STEP
        elif value in state.expected_issues:
            amt = _ISSUE_REWARDS.get(value, 0.0)
            breakdown[f"detected_{value}"] = amt
            total += amt
        else:
            # flagged something that isn't actually wrong
            breakdown["false_positive"] = _UNNECESSARY_STEP
            total += _UNNECESSARY_STEP

    reason = "; ".join(f"{k}: {v:+.2f}" for k, v in breakdown.items()) or "no step reward"
    return InvoiceReward(
        total=round(max(-1.0, min(1.0, total)), 4),
        breakdown=breakdown,
        reason=reason,
    )


def compute_final_reward(state: InvoiceState, max_steps: int) -> InvoiceReward:
    """
    Terminal reward — scores only the final decision, not issue detection
    (issues were already rewarded step-by-step, so we don't double-count).
    """
    breakdown: dict[str, float] = {}
    total = 0.0

    status_to_action = {
        "approved": "mark_valid",
        "rejected":  "mark_invalid",
        "flagged":   "flag_duplicate",
        "on_hold":   "request_missing_info",
    }
    agent_action = status_to_action.get(str(state.approval_status.value), "")

    if state.expected_action:
        if agent_action == state.expected_action:
            breakdown["correct_final_action"] = _CORRECT_ACTION_BONUS
            total += _CORRECT_ACTION_BONUS
        elif agent_action:
            breakdown["wrong_final_action"] = _WRONG_ACTION_PENALTY
            total += _WRONG_ACTION_PENALTY

    # penalise if the agent took way more steps than needed
    min_steps = max(3, len(state.expected_issues) + 2)
    if state.steps_taken > min_steps + _GRACE_BUFFER:
        breakdown["too_many_steps"] = _TOO_LONG_PENALTY
        total += _TOO_LONG_PENALTY

    reason = "; ".join(f"{k}: {v:+.2f}" for k, v in breakdown.items()) or "episode ended"
    return InvoiceReward(total=round(total, 4), breakdown=breakdown, reason=reason)


def grade_submission(
    detected_issues: List[str],
    expected_issues: List[str],
    final_action: str,
    expected_action: str,
    steps_taken: int,
) -> float:
    """Standalone scorer used by the validation script. Returns 0.0–1.0."""
    total = sum(_ISSUE_REWARDS.get(i, 0.0) for i in expected_issues if i in detected_issues)
    total += _CORRECT_ACTION_BONUS if final_action == expected_action else _WRONG_ACTION_PENALTY
    return round(max(0.0, min(1.0, total)), 4)
