from __future__ import annotations

import math
from typing import List

from app.models import InvoiceState

# floats on invoices are messy — anything within ±1 rupee is fine
_TOTAL_TOLERANCE = 1.0


def check_missing_gst(state: InvoiceState) -> List[str]:
    if not state.gst_number or not state.gst_number.strip():
        return ["missing_gst_number"]
    return []


def check_missing_vendor(state: InvoiceState) -> List[str]:
    if not state.vendor_name or not state.vendor_name.strip():
        return ["missing_vendor_name"]
    return []


def check_wrong_total(state: InvoiceState) -> List[str]:
    expected = round(state.subtotal + state.tax, 2)
    if math.fabs(state.total - expected) > _TOTAL_TOLERANCE:
        return ["wrong_total_calculation"]
    return []


def check_duplicate(state: InvoiceState) -> List[str]:
    return ["duplicate_invoice"] if state.duplicate_invoice else []


def check_missing_receipt(state: InvoiceState) -> List[str]:
    return ["missing_receipt"] if not state.receipt_attached else []


_RULES = [
    check_missing_gst,
    check_missing_vendor,
    check_wrong_total,
    check_duplicate,
    check_missing_receipt,
]


def validate_invoice(state: InvoiceState) -> List[str]:
    found: List[str] = []
    for rule in _RULES:
        found.extend(rule(state))
    return sorted(set(found))


def compute_expected_total(state: InvoiceState) -> float:
    return round(state.subtotal + state.tax, 2)
