from app.models import InvoiceState, ApprovalStatus


def get_easy_task() -> InvoiceState:
    """
    Everything looks normal except the GST number is missing.
    Agent should: add_issue → request_missing_info → finalize.
    """
    return InvoiceState(
        invoice_id="INV-2024-0042",
        vendor_name="Acme Corp",
        subtotal=5000.00,
        tax=900.00,
        total=5900.00,        # correct (5000+900)
        gst_number=None,      # <-- the one problem
        invoice_date="2024-03-15",
        duplicate_invoice=False,
        receipt_attached=True,
        expected_issues=["missing_gst_number"],
        expected_action="request_missing_info",
        task_difficulty="easy",
    )


def get_medium_task() -> InvoiceState:
    """
    Vendor added up the numbers wrong.
    2000 + 360 = 2360, but invoice says 2200.
    Agent should: add_issue(wrong_total_calculation) → mark_invalid → finalize.
    """
    return InvoiceState(
        invoice_id="INV-2024-0117",
        vendor_name="BuildRight Supplies",
        subtotal=2000.00,
        tax=360.00,
        total=2200.00,         # wrong — should be 2360
        gst_number="29AABCU9603R1ZX",
        invoice_date="2024-04-02",
        duplicate_invoice=False,
        receipt_attached=True,
        expected_issues=["wrong_total_calculation"],
        expected_action="mark_invalid",
        task_difficulty="medium",
    )


def get_hard_task() -> InvoiceState:
    """
    Everything wrong at once — classic fraud pattern.
    No vendor name, no GST, wrong total, already submitted before, no receipt.
    Agent needs to catch all five issues before making a final call.
    """
    return InvoiceState(
        invoice_id="INV-2024-0088",
        vendor_name=None,           # missing
        subtotal=8000.00,
        tax=1440.00,
        total=9800.00,              # wrong — should be 9440
        gst_number=None,            # missing
        invoice_date="2024-02-28",
        duplicate_invoice=True,     # seen before
        receipt_attached=False,     # no receipt
        expected_issues=[
            "missing_vendor_name",
            "missing_gst_number",
            "wrong_total_calculation",
            "duplicate_invoice",
            "missing_receipt",
        ],
        expected_action="mark_invalid",
        task_difficulty="hard",
    )


TASKS = {
    "easy":   get_easy_task,
    "medium": get_medium_task,
    "hard":   get_hard_task,
}


def load_task(difficulty: str) -> InvoiceState:
    if difficulty not in TASKS:
        raise ValueError(f"Unknown difficulty '{difficulty}'. Pick from: {list(TASKS)}")
    return TASKS[difficulty]()
