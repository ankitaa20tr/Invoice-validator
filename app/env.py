from __future__ import annotations

import copy
from typing import Optional

from app.models import (
    ActionType,
    ApprovalStatus,
    InvoiceAction,
    InvoiceObservation,
    InvoiceReward,
    InvoiceState,
    StepResult,
)
from app.tasks import load_task
from app.grader import compute_step_reward, compute_final_reward

MAX_STEPS = 20


class InvoiceValidationEnv:
    """
    The core RL environment. Wraps one invoice review episode.

    Call reset() to start, step() to take actions, state() if you need
    the raw ground-truth for grading/debugging.
    """

    def __init__(self) -> None:
        self._state: Optional[InvoiceState] = None
        self._cumulative_step_reward: float = 0.0

    def reset(self, difficulty: str = "easy") -> InvoiceObservation:
        self._state = load_task(difficulty)
        self._cumulative_step_reward = 0.0
        return self._build_observation("Episode started. Begin review.")

    def step(self, action: InvoiceAction) -> StepResult:
        if self._state is None:
            raise RuntimeError("Call reset() before step().")

        state = self._state
        state.steps_taken += 1

        # hard cap on steps
        if state.steps_taken >= MAX_STEPS and not state.done:
            state.done = True
            terminal = compute_final_reward(state, MAX_STEPS)
            terminal.reason = "step limit reached; " + terminal.reason
            return StepResult(
                observation=self._build_observation("Step limit reached. Episode ended."),
                reward=terminal,
                done=True,
                info={"reason": "max_steps_exceeded"},
            )

        action_was_useful, msg = self._apply_action(action)
        reward = compute_step_reward(state, action.action_type, action.value, action_was_useful)
        self._cumulative_step_reward += reward.total

        # episode ends on finalize_review or on any terminal-decision action
        if action.action_type == ActionType.finalize_review or state.done:
            state.done = True
            terminal = compute_final_reward(state, MAX_STEPS)
            final_total = round(max(0.0, min(1.0, self._cumulative_step_reward + terminal.total)), 4)
            final_reward = InvoiceReward(
                total=final_total,
                breakdown={**terminal.breakdown, **reward.breakdown},
                reason=terminal.reason,
            )
            return StepResult(
                observation=self._build_observation(msg),
                reward=final_reward,
                done=True,
                info={"final": True, "cumulative_step_reward": self._cumulative_step_reward},
            )

        return StepResult(
            observation=self._build_observation(msg),
            reward=reward,
            done=False,
            info={},
        )

    def state(self) -> InvoiceState:
        if self._state is None:
            raise RuntimeError("Call reset() first.")
        return copy.deepcopy(self._state)

    # -------------------------------------------------------------------------

    def _apply_action(self, action: InvoiceAction) -> tuple[bool, str]:
        """Returns (changed_anything, message)."""
        state = self._state
        atype = action.action_type

        if atype == ActionType.add_issue:
            issue = (action.value or "").strip()
            if not issue:
                return False, "add_issue needs a value."
            if issue in state.detected_issues:
                return False, f"'{issue}' already noted."
            state.detected_issues.append(issue)
            return True, f"Recorded issue: '{issue}'."

        elif atype == ActionType.mark_valid:
            if state.approval_status == ApprovalStatus.approved:
                return False, "Already marked valid."
            state.approval_status = ApprovalStatus.approved
            state.done = True
            return True, "Invoice approved."

        elif atype == ActionType.mark_invalid:
            if state.approval_status == ApprovalStatus.rejected:
                return False, "Already marked invalid."
            state.approval_status = ApprovalStatus.rejected
            state.done = True
            return True, "Invoice rejected."

        elif atype == ActionType.flag_duplicate:
            if state.approval_status == ApprovalStatus.flagged:
                return False, "Already flagged."
            state.approval_status = ApprovalStatus.flagged
            return True, "Flagged as duplicate — needs manual check."

        elif atype == ActionType.request_missing_info:
            if state.approval_status == ApprovalStatus.on_hold:
                return False, "Already on hold."
            state.approval_status = ApprovalStatus.on_hold
            state.done = True
            return True, "On hold — requested missing info from vendor."

        elif atype == ActionType.finalize_review:
            return True, "Review finalised."

        return False, f"Unknown action: {atype}"

    def _build_observation(self, last_action_result: str) -> InvoiceObservation:
        state = self._state
        remaining = max(0, MAX_STEPS - state.steps_taken)
        task_desc = (
            f"[{state.task_difficulty.upper()} TASK] Review invoice {state.invoice_id}. "
            f"Use add_issue for each problem you find, then pick a final action "
            f"(mark_valid / mark_invalid / flag_duplicate / request_missing_info) "
            f"and call finalize_review."
        )
        return InvoiceObservation(
            invoice_id=state.invoice_id,
            vendor_name=state.vendor_name,
            subtotal=state.subtotal,
            tax=state.tax,
            total=state.total,
            gst_number=state.gst_number,
            invoice_date=state.invoice_date,
            duplicate_flag=state.duplicate_invoice,
            receipt_attached=state.receipt_attached,
            approval_status=str(state.approval_status.value),
            detected_issues=list(state.detected_issues),
            last_action_result=last_action_result,
            remaining_steps=remaining,
            done=state.done,
            task_description=task_desc,
        )
