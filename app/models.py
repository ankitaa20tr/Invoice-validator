from __future__ import annotations

from typing import Any, Dict, List, Optional
from enum import Enum

from pydantic import BaseModel, Field


class ApprovalStatus(str, Enum):
    pending  = "pending"
    approved = "approved"
    rejected = "rejected"
    flagged  = "flagged"   # duplicate suspected
    on_hold  = "on_hold"   # waiting on vendor to send missing info


class ActionType(str, Enum):
    mark_valid           = "mark_valid"
    mark_invalid         = "mark_invalid"
    flag_duplicate       = "flag_duplicate"
    request_missing_info = "request_missing_info"
    add_issue            = "add_issue"
    finalize_review      = "finalize_review"


class InvoiceState(BaseModel):
    # what's on the actual invoice
    invoice_id:        str
    vendor_name:       Optional[str] = None
    subtotal:          float = 0.0
    tax:               float = 0.0
    total:             float = 0.0
    gst_number:        Optional[str] = None
    invoice_date:      Optional[str] = None
    duplicate_invoice: bool = False
    receipt_attached:  bool = True

    # tracks where we are in the review
    approval_status: ApprovalStatus = ApprovalStatus.pending
    detected_issues: List[str] = Field(default_factory=list)
    steps_taken:     int  = 0
    done:            bool = False

    # hidden ground truth — agent never sees these
    expected_issues: List[str]    = Field(default_factory=list)
    expected_action: Optional[str] = None
    task_difficulty: str           = "easy"


class InvoiceAction(BaseModel):
    """
    What the agent wants to do next.

    Examples:
        {"action_type": "add_issue", "value": "missing_gst_number"}
        {"action_type": "mark_invalid"}
    """
    action_type: ActionType
    value: Optional[str] = None

    class Config:
        use_enum_values = True


class InvoiceObservation(BaseModel):
    """What the agent sees after every step — no ground truth exposed."""
    invoice_id:        str
    vendor_name:       Optional[str]
    subtotal:          float
    tax:               float
    total:             float
    gst_number:        Optional[str]
    invoice_date:      Optional[str]
    duplicate_flag:    bool
    receipt_attached:  bool

    approval_status:   str
    detected_issues:   List[str]
    last_action_result: str = "none"
    remaining_steps:   int  = 10
    done:              bool = False
    task_description:  str  = ""


class InvoiceReward(BaseModel):
    total:     float = 0.0
    breakdown: Dict[str, float] = Field(default_factory=dict)
    reason:    str = ""


class StepResult(BaseModel):
    observation: InvoiceObservation
    reward:      InvoiceReward
    done:        bool
    info:        Dict[str, Any] = Field(default_factory=dict)
