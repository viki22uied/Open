"""
Task registry for the Invoice Review environment.

Each task defines a difficulty level, description, step budget, and a
deterministic data generator so results are fully reproducible.
"""

from __future__ import annotations

from typing import Dict, List

from src.models import TaskInfo


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

TASK_REGISTRY: Dict[str, Dict] = {
    "easy": {
        "task_id": "easy",
        "name": "Single Invoice Review",
        "difficulty": "easy",
        "description": (
            "Review a single invoice from a trusted vendor. The invoice "
            "contains obvious arithmetic errors in line item pricing. "
            "Identify the errors, flag them, and decide whether to approve "
            "or reject the invoice. A straightforward warm-up task."
        ),
        "max_steps": 15,
        "num_invoices": 1,
        "seed": 42,
    },
    "medium": {
        "task_id": "medium",
        "name": "Multi-Invoice Batch Review",
        "difficulty": "medium",
        "description": (
            "Review a batch of 3 invoices from different vendors. One invoice "
            "is clean and should be approved. The other two contain errors "
            "including vendor status issues, PO mismatches, missing purchase "
            "orders, and overcharges. You must perform appropriate checks "
            "(vendor verification, PO matching) and correctly disposition "
            "each invoice."
        ),
        "max_steps": 30,
        "num_invoices": 3,
        "seed": 42,
    },
    "hard": {
        "task_id": "hard",
        "name": "Complex Procurement Audit",
        "difficulty": "hard",
        "description": (
            "Perform a comprehensive audit of 5 invoices requiring careful "
            "analysis. Challenges include: a blacklisted vendor, a duplicate "
            "invoice, subtle math errors ($50 overcharge), policy violations "
            "(dual-approval threshold), unauthorized line items not on the PO, "
            "and correct escalation routing. You must check vendors, verify "
            "POs, scan for duplicates, catch math errors, enforce policy, and "
            "correctly approve/reject/escalate each invoice."
        ),
        "max_steps": 50,
        "num_invoices": 5,
        "seed": 42,
    },
}


def get_task(task_id: str) -> TaskInfo:
    """Return TaskInfo for a given task ID."""
    if task_id not in TASK_REGISTRY:
        raise ValueError(
            f"Unknown task_id: {task_id!r}. Available: {list(TASK_REGISTRY)}"
        )
    t = TASK_REGISTRY[task_id]
    return TaskInfo(
        task_id=t["task_id"],
        name=t["name"],
        difficulty=t["difficulty"],
        description=t["description"],
        max_steps=t["max_steps"],
        num_invoices=t["num_invoices"],
    )


def list_tasks() -> List[TaskInfo]:
    """Return metadata for all registered tasks."""
    return [get_task(tid) for tid in TASK_REGISTRY]
