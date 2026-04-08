"""
Deterministic data generator for Invoice Review tasks.

Each task difficulty level produces a fixed set of invoices with specific
injected errors. Using seed-based RNG ensures full reproducibility.
"""

from __future__ import annotations

import random
from copy import deepcopy
from typing import Any, Dict, List, Tuple

from src.models import (
    CompanyPolicy,
    ErrorCategory,
    Invoice,
    LineItem,
    PurchaseOrder,
    Severity,
    VendorRecord,
)


# ---------------------------------------------------------------------------
# Vendor master data (shared across tasks)
# ---------------------------------------------------------------------------

_VENDORS: List[Dict[str, Any]] = [
    {
        "vendor_id": "V-1001",
        "vendor_name": "Acme Office Supplies",
        "status": "active",
        "payment_terms": "Net 30",
        "contract_expiry": "2027-06-30",
        "risk_rating": "low",
    },
    {
        "vendor_id": "V-1002",
        "vendor_name": "GlobalTech Solutions",
        "status": "active",
        "payment_terms": "Net 45",
        "contract_expiry": "2026-12-31",
        "risk_rating": "medium",
    },
    {
        "vendor_id": "V-1003",
        "vendor_name": "QuickShip Logistics",
        "status": "active",
        "payment_terms": "Net 15",
        "contract_expiry": "2027-03-31",
        "risk_rating": "low",
    },
    {
        "vendor_id": "V-1004",
        "vendor_name": "ShadowCorp Industries",
        "status": "suspended",
        "payment_terms": "Net 60",
        "contract_expiry": "2025-01-15",
        "risk_rating": "high",
    },
    {
        "vendor_id": "V-1005",
        "vendor_name": "Premium Catering Co",
        "status": "active",
        "payment_terms": "Net 30",
        "contract_expiry": "2027-09-30",
        "risk_rating": "low",
    },
    {
        "vendor_id": "V-1006",
        "vendor_name": "TechNova Consulting",
        "status": "blacklisted",
        "payment_terms": "Net 30",
        "contract_expiry": "2024-06-30",
        "risk_rating": "high",
    },
    {
        "vendor_id": "V-1007",
        "vendor_name": "ClearView Analytics",
        "status": "active",
        "payment_terms": "Net 30",
        "contract_expiry": "2027-12-31",
        "risk_rating": "medium",
    },
]


def _build_vendors(vendor_ids: List[str]) -> List[VendorRecord]:
    """Return VendorRecord objects for the requested vendor IDs."""
    lookup = {v["vendor_id"]: v for v in _VENDORS}
    return [VendorRecord(**lookup[vid]) for vid in vendor_ids if vid in lookup]


# ---------------------------------------------------------------------------
# Easy task — single invoice, obvious math error
# ---------------------------------------------------------------------------

def _generate_easy(seed: int = 42) -> Dict[str, Any]:
    rng = random.Random(seed)
    _ = rng  # deterministic but simple enough we hardcode

    invoices = [
        Invoice(
            invoice_id="INV-2025-0001",
            vendor_name="Acme Office Supplies",
            vendor_id="V-1001",
            invoice_date="2025-10-15",
            due_date="2025-11-14",
            po_number="PO-5001",
            currency="USD",
            line_items=[
                LineItem(description="A4 Copy Paper (Case)", quantity=50, unit_price=24.99, extended_price=1249.50),
                LineItem(description="Ballpoint Pens (Box of 12)", quantity=20, unit_price=8.50, extended_price=170.00),
                LineItem(description="Desk Organizer", quantity=10, unit_price=15.00, extended_price=200.00),
                # Math error: 10 * 15.00 = 150.00, not 200.00
            ],
            subtotal=1619.50,
            tax_rate=0.08,
            tax_amount=129.56,
            total_amount=1749.06,
            notes="Rush delivery requested.",
            _injected_errors=[],
        ),
    ]
    # Manually set injected errors (Pydantic private field workaround)
    invoices[0].__dict__["_injected_errors"] = [
        {
            "category": ErrorCategory.MATH_ERROR.value,
            "description": "Desk Organizer extended_price is 200.00 but should be 150.00 (10 × 15.00)",
            "severity": Severity.MEDIUM.value,
            "line_item_index": 2,
        },
        {
            "category": ErrorCategory.MATH_ERROR.value,
            "description": "Subtotal should be 1569.50 not 1619.50 (cascading from line item error)",
            "severity": Severity.MEDIUM.value,
        },
    ]

    purchase_orders = [
        PurchaseOrder(
            po_number="PO-5001",
            vendor_id="V-1001",
            approved_amount=2000.00,
            approved_items=["A4 Copy Paper", "Ballpoint Pens", "Desk Organizer"],
            status="open",
            created_date="2025-10-01",
        ),
    ]

    vendors = _build_vendors(["V-1001"])

    policy = CompanyPolicy(
        max_auto_approve=5000.0,
        require_po_above=1000.0,
        require_dual_approval_above=10000.0,
        blocked_vendors=[],
        max_line_item_variance_pct=10.0,
    )

    ground_truth_errors = {
        "INV-2025-0001": [
            {
                "category": ErrorCategory.MATH_ERROR.value,
                "description": "Desk Organizer extended_price is $200.00 but should be $150.00 (10 × $15.00 = $150.00). Overcharge of $50.00.",
                "severity": Severity.MEDIUM.value,
            },
            {
                "category": ErrorCategory.MATH_ERROR.value,
                "description": "Subtotal is $1,619.50 but correct sum is $1,569.50 ($1,249.50 + $170.00 + $150.00). Cascading error from line item.",
                "severity": Severity.MEDIUM.value,
            },
        ],
    }

    ground_truth_dispositions = {
        "INV-2025-0001": "rejected",  # Has errors, should be rejected or sent back
    }

    return {
        "invoices": invoices,
        "purchase_orders": purchase_orders,
        "vendors": vendors,
        "policy": policy,
        "ground_truth_errors": ground_truth_errors,
        "ground_truth_dispositions": ground_truth_dispositions,
    }


# ---------------------------------------------------------------------------
# Medium task — 3 invoices, mix of errors across categories
# ---------------------------------------------------------------------------

def _generate_medium(seed: int = 42) -> Dict[str, Any]:
    rng = random.Random(seed)
    _ = rng

    invoices = [
        # Invoice 1: Clean — should be approved
        Invoice(
            invoice_id="INV-2025-0010",
            vendor_name="QuickShip Logistics",
            vendor_id="V-1003",
            invoice_date="2025-11-01",
            due_date="2025-11-16",
            po_number="PO-5010",
            currency="USD",
            line_items=[
                LineItem(description="Standard Freight — Zone A", quantity=5, unit_price=120.00, extended_price=600.00),
                LineItem(description="Express Handling Fee", quantity=2, unit_price=45.00, extended_price=90.00),
            ],
            subtotal=690.00,
            tax_rate=0.07,
            tax_amount=48.30,
            total_amount=738.30,
            notes=None,
        ),
        # Invoice 2: Suspended vendor + PO mismatch
        Invoice(
            invoice_id="INV-2025-0011",
            vendor_name="ShadowCorp Industries",
            vendor_id="V-1004",
            invoice_date="2025-11-05",
            due_date="2026-01-04",
            po_number="PO-5011",
            currency="USD",
            line_items=[
                LineItem(description="Industrial Bearings (Set)", quantity=100, unit_price=35.00, extended_price=3500.00),
                LineItem(description="Lubricant (Gallon)", quantity=20, unit_price=22.50, extended_price=450.00),
            ],
            subtotal=3950.00,
            tax_rate=0.07,
            tax_amount=276.50,
            total_amount=4226.50,
            notes="Urgent — production line waiting.",
        ),
        # Invoice 3: Overcharge + missing PO for high-value
        Invoice(
            invoice_id="INV-2025-0012",
            vendor_name="GlobalTech Solutions",
            vendor_id="V-1002",
            invoice_date="2025-11-10",
            due_date="2025-12-25",
            po_number=None,  # Missing PO for >$1000
            currency="USD",
            line_items=[
                LineItem(description="Cloud Server Hosting (Monthly)", quantity=1, unit_price=2400.00, extended_price=2400.00),
                LineItem(description="Premium Support Package", quantity=1, unit_price=800.00, extended_price=1200.00),
                # Overcharge: 1 * 800 = 800, not 1200
                LineItem(description="SSL Certificate (Annual)", quantity=3, unit_price=150.00, extended_price=450.00),
            ],
            subtotal=4050.00,
            tax_rate=0.08,
            tax_amount=324.00,
            total_amount=4374.00,
            notes="Auto-renewal contract.",
        ),
    ]

    # Set injected errors
    invoices[0].__dict__["_injected_errors"] = []  # clean
    invoices[1].__dict__["_injected_errors"] = [
        {"category": ErrorCategory.VENDOR_ISSUE.value, "description": "Vendor is suspended", "severity": Severity.CRITICAL.value},
        {"category": ErrorCategory.PO_MISMATCH.value, "description": "PO-5011 amount exceeded", "severity": Severity.HIGH.value},
    ]
    invoices[2].__dict__["_injected_errors"] = [
        {"category": ErrorCategory.OVERCHARGE.value, "description": "Premium Support Package overcharged by $400", "severity": Severity.HIGH.value},
        {"category": ErrorCategory.MISSING_FIELD.value, "description": "No PO for invoice over $1000", "severity": Severity.HIGH.value},
    ]

    purchase_orders = [
        PurchaseOrder(
            po_number="PO-5010",
            vendor_id="V-1003",
            approved_amount=800.00,
            approved_items=["Freight", "Handling"],
            status="open",
            created_date="2025-10-20",
        ),
        PurchaseOrder(
            po_number="PO-5011",
            vendor_id="V-1004",
            approved_amount=3000.00,  # Invoice exceeds this
            approved_items=["Industrial Bearings"],
            status="open",
            created_date="2025-10-25",
        ),
    ]

    vendors = _build_vendors(["V-1002", "V-1003", "V-1004"])

    policy = CompanyPolicy(
        max_auto_approve=5000.0,
        require_po_above=1000.0,
        require_dual_approval_above=10000.0,
        blocked_vendors=["V-1006"],
        max_line_item_variance_pct=10.0,
    )

    ground_truth_errors = {
        "INV-2025-0010": [],  # Clean invoice
        "INV-2025-0011": [
            {
                "category": ErrorCategory.VENDOR_ISSUE.value,
                "description": "Vendor ShadowCorp Industries (V-1004) is suspended. Cannot process invoices from suspended vendors.",
                "severity": Severity.CRITICAL.value,
            },
            {
                "category": ErrorCategory.PO_MISMATCH.value,
                "description": "Invoice total $4,226.50 exceeds PO-5011 approved amount of $3,000.00 by $1,226.50.",
                "severity": Severity.HIGH.value,
            },
        ],
        "INV-2025-0012": [
            {
                "category": ErrorCategory.OVERCHARGE.value,
                "description": "Premium Support Package: extended_price is $1,200.00 but should be $800.00 (1 × $800.00). Overcharge of $400.00.",
                "severity": Severity.HIGH.value,
            },
            {
                "category": ErrorCategory.MISSING_FIELD.value,
                "description": "Invoice total exceeds $1,000 policy threshold but has no purchase order number.",
                "severity": Severity.HIGH.value,
            },
        ],
    }

    ground_truth_dispositions = {
        "INV-2025-0010": "approved",
        "INV-2025-0011": "rejected",
        "INV-2025-0012": "rejected",
    }

    return {
        "invoices": invoices,
        "purchase_orders": purchase_orders,
        "vendors": vendors,
        "policy": policy,
        "ground_truth_errors": ground_truth_errors,
        "ground_truth_dispositions": ground_truth_dispositions,
    }


# ---------------------------------------------------------------------------
# Hard task — 5 invoices, subtle errors, policy violations, duplicates
# ---------------------------------------------------------------------------

def _generate_hard(seed: int = 42) -> Dict[str, Any]:
    rng = random.Random(seed)
    _ = rng

    invoices = [
        # Invoice 1: Clean, small — should be auto-approved
        Invoice(
            invoice_id="INV-2025-0100",
            vendor_name="Premium Catering Co",
            vendor_id="V-1005",
            invoice_date="2025-12-01",
            due_date="2025-12-31",
            po_number="PO-6001",
            currency="USD",
            line_items=[
                LineItem(description="Corporate Lunch Package", quantity=30, unit_price=18.50, extended_price=555.00),
                LineItem(description="Beverage Service", quantity=30, unit_price=4.75, extended_price=142.50),
            ],
            subtotal=697.50,
            tax_rate=0.06,
            tax_amount=41.85,
            total_amount=739.35,
        ),
        # Invoice 2: Blacklisted vendor
        Invoice(
            invoice_id="INV-2025-0101",
            vendor_name="TechNova Consulting",
            vendor_id="V-1006",
            invoice_date="2025-12-03",
            due_date="2026-01-02",
            po_number="PO-6002",
            currency="USD",
            line_items=[
                LineItem(description="IT Strategy Consulting (Hours)", quantity=40, unit_price=250.00, extended_price=10000.00),
                LineItem(description="Documentation & Deliverables", quantity=1, unit_price=1500.00, extended_price=1500.00),
            ],
            subtotal=11500.00,
            tax_rate=0.08,
            tax_amount=920.00,
            total_amount=12420.00,
            notes="Phase 2 of digital transformation project.",
        ),
        # Invoice 3: Duplicate of Invoice 1 (same vendor, same amounts, different ID)
        Invoice(
            invoice_id="INV-2025-0102",
            vendor_name="Premium Catering Co",
            vendor_id="V-1005",
            invoice_date="2025-12-01",
            due_date="2025-12-31",
            po_number="PO-6001",
            currency="USD",
            line_items=[
                LineItem(description="Corporate Lunch Package", quantity=30, unit_price=18.50, extended_price=555.00),
                LineItem(description="Beverage Service", quantity=30, unit_price=4.75, extended_price=142.50),
            ],
            subtotal=697.50,
            tax_rate=0.06,
            tax_amount=41.85,
            total_amount=739.35,
        ),
        # Invoice 4: Subtle math error + exceeds dual-approval threshold
        Invoice(
            invoice_id="INV-2025-0103",
            vendor_name="ClearView Analytics",
            vendor_id="V-1007",
            invoice_date="2025-12-05",
            due_date="2026-01-04",
            po_number="PO-6003",
            currency="USD",
            line_items=[
                LineItem(description="Data Analytics Platform License", quantity=1, unit_price=8500.00, extended_price=8500.00),
                LineItem(description="Implementation Support", quantity=24, unit_price=175.00, extended_price=4200.00),
                # Correct: 24 * 175 = 4200 ✓
                LineItem(description="Training Sessions", quantity=6, unit_price=450.00, extended_price=2750.00),
                # Error: 6 * 450 = 2700, not 2750 — subtle $50 overcharge
            ],
            subtotal=15450.00,
            tax_rate=0.08,
            tax_amount=1236.00,
            total_amount=16686.00,
            notes="Annual renewal with expanded scope.",
        ),
        # Invoice 5: Policy violation — unauthorized items on PO
        Invoice(
            invoice_id="INV-2025-0104",
            vendor_name="Acme Office Supplies",
            vendor_id="V-1001",
            invoice_date="2025-12-07",
            due_date="2026-01-06",
            po_number="PO-6004",
            currency="USD",
            line_items=[
                LineItem(description="Ergonomic Office Chair", quantity=5, unit_price=450.00, extended_price=2250.00),
                LineItem(description="Standing Desk Converter", quantity=5, unit_price=320.00, extended_price=1600.00),
                LineItem(description="Monitor Arms", quantity=10, unit_price=89.00, extended_price=890.00),
                LineItem(description="Personal Mini Fridge", quantity=5, unit_price=199.00, extended_price=995.00),
                # Unauthorized item — not on PO
            ],
            subtotal=5735.00,
            tax_rate=0.08,
            tax_amount=458.80,
            total_amount=6193.80,
            notes="Office renovation project Phase 1.",
        ),
    ]

    # Set injected errors
    invoices[0].__dict__["_injected_errors"] = []
    invoices[1].__dict__["_injected_errors"] = [
        {"category": ErrorCategory.VENDOR_ISSUE.value, "description": "Vendor is blacklisted", "severity": Severity.CRITICAL.value},
        {"category": ErrorCategory.POLICY_VIOLATION.value, "description": "Exceeds dual-approval threshold without two approvers", "severity": Severity.HIGH.value},
    ]
    invoices[2].__dict__["_injected_errors"] = [
        {"category": ErrorCategory.DUPLICATE.value, "description": "Duplicate of INV-2025-0100", "severity": Severity.HIGH.value},
    ]
    invoices[3].__dict__["_injected_errors"] = [
        {"category": ErrorCategory.MATH_ERROR.value, "description": "Training Sessions overcharged by $50", "severity": Severity.LOW.value},
        {"category": ErrorCategory.POLICY_VIOLATION.value, "description": "Exceeds dual-approval threshold", "severity": Severity.HIGH.value},
    ]
    invoices[4].__dict__["_injected_errors"] = [
        {"category": ErrorCategory.UNAUTHORIZED.value, "description": "Personal Mini Fridge not on PO", "severity": Severity.MEDIUM.value},
        {"category": ErrorCategory.PO_MISMATCH.value, "description": "Invoice total exceeds PO approved amount", "severity": Severity.HIGH.value},
    ]

    purchase_orders = [
        PurchaseOrder(
            po_number="PO-6001",
            vendor_id="V-1005",
            approved_amount=800.00,
            approved_items=["Corporate Lunch Package", "Beverage Service"],
            status="open",
            created_date="2025-11-25",
        ),
        PurchaseOrder(
            po_number="PO-6002",
            vendor_id="V-1006",
            approved_amount=15000.00,
            approved_items=["IT Strategy Consulting", "Documentation"],
            status="open",
            created_date="2025-11-20",
        ),
        PurchaseOrder(
            po_number="PO-6003",
            vendor_id="V-1007",
            approved_amount=16000.00,
            approved_items=["Data Analytics Platform License", "Implementation Support", "Training Sessions"],
            status="open",
            created_date="2025-11-28",
        ),
        PurchaseOrder(
            po_number="PO-6004",
            vendor_id="V-1001",
            approved_amount=5000.00,
            approved_items=["Ergonomic Office Chair", "Standing Desk Converter", "Monitor Arms"],
            status="open",
            created_date="2025-12-01",
        ),
    ]

    vendors = _build_vendors(["V-1001", "V-1005", "V-1006", "V-1007"])

    policy = CompanyPolicy(
        max_auto_approve=5000.0,
        require_po_above=1000.0,
        require_dual_approval_above=10000.0,
        blocked_vendors=["V-1006"],
        max_line_item_variance_pct=10.0,
    )

    ground_truth_errors = {
        "INV-2025-0100": [],  # Clean
        "INV-2025-0101": [
            {
                "category": ErrorCategory.VENDOR_ISSUE.value,
                "description": "Vendor TechNova Consulting (V-1006) is blacklisted. All invoices must be rejected.",
                "severity": Severity.CRITICAL.value,
            },
            {
                "category": ErrorCategory.POLICY_VIOLATION.value,
                "description": "Invoice total $12,420.00 exceeds dual-approval threshold of $10,000. Requires two separate approvers.",
                "severity": Severity.HIGH.value,
            },
        ],
        "INV-2025-0102": [
            {
                "category": ErrorCategory.DUPLICATE.value,
                "description": "Invoice INV-2025-0102 appears to be a duplicate of INV-2025-0100. Same vendor, date, PO, amounts, and line items.",
                "severity": Severity.HIGH.value,
            },
        ],
        "INV-2025-0103": [
            {
                "category": ErrorCategory.MATH_ERROR.value,
                "description": "Training Sessions: extended_price is $2,750.00 but should be $2,700.00 (6 × $450.00). Overcharge of $50.00.",
                "severity": Severity.LOW.value,
            },
            {
                "category": ErrorCategory.POLICY_VIOLATION.value,
                "description": "Invoice total $16,686.00 exceeds dual-approval threshold of $10,000. Must be escalated for dual approval.",
                "severity": Severity.HIGH.value,
            },
        ],
        "INV-2025-0104": [
            {
                "category": ErrorCategory.UNAUTHORIZED.value,
                "description": "Line item 'Personal Mini Fridge' ($995.00) is not listed on PO-6004 approved items.",
                "severity": Severity.MEDIUM.value,
            },
            {
                "category": ErrorCategory.PO_MISMATCH.value,
                "description": "Invoice total $6,193.80 exceeds PO-6004 approved amount of $5,000.00 by $1,193.80.",
                "severity": Severity.HIGH.value,
            },
        ],
    }

    ground_truth_dispositions = {
        "INV-2025-0100": "approved",
        "INV-2025-0101": "rejected",
        "INV-2025-0102": "rejected",
        "INV-2025-0103": "escalated",  # Needs dual approval + has math error
        "INV-2025-0104": "rejected",
    }

    return {
        "invoices": invoices,
        "purchase_orders": purchase_orders,
        "vendors": vendors,
        "policy": policy,
        "ground_truth_errors": ground_truth_errors,
        "ground_truth_dispositions": ground_truth_dispositions,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_GENERATORS = {
    "easy": _generate_easy,
    "medium": _generate_medium,
    "hard": _generate_hard,
}


def generate_task_data(difficulty: str, seed: int = 42) -> Dict[str, Any]:
    """Generate all data for a task at the given difficulty level.

    Returns a dict with keys:
        invoices, purchase_orders, vendors, policy,
        ground_truth_errors, ground_truth_dispositions
    """
    if difficulty not in _GENERATORS:
        raise ValueError(f"Unknown difficulty: {difficulty!r}. Choose from {list(_GENERATORS)}")
    return _GENERATORS[difficulty](seed=seed)
