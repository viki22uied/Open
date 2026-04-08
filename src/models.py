"""
Typed Pydantic models for the Invoice Review OpenEnv environment.

Defines Observation, Action, Reward, and internal State schemas used
throughout the environment, server, and grader layers.
"""

from __future__ import annotations

import enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ActionType(str, enum.Enum):
    """Available actions an agent can take."""
    INSPECT_INVOICE = "inspect_invoice"
    FLAG_ERROR = "flag_error"
    CHECK_PO_MATCH = "check_po_match"
    CHECK_VENDOR = "check_vendor"
    CHECK_DUPLICATE = "check_duplicate"
    APPROVE = "approve"
    REJECT = "reject"
    ESCALATE = "escalate"
    SUBMIT_REVIEW = "submit_review"


class Severity(str, enum.Enum):
    """Severity of a flagged error."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(str, enum.Enum):
    """Categories of invoice errors."""
    MATH_ERROR = "math_error"
    MISSING_FIELD = "missing_field"
    PO_MISMATCH = "po_mismatch"
    DUPLICATE = "duplicate"
    VENDOR_ISSUE = "vendor_issue"
    POLICY_VIOLATION = "policy_violation"
    OVERCHARGE = "overcharge"
    UNAUTHORIZED = "unauthorized"


class InvoiceStatus(str, enum.Enum):
    """Final disposition for an invoice."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    ESCALATED = "escalated"


# ---------------------------------------------------------------------------
# Data models — invoices, line items, purchase orders, vendors
# ---------------------------------------------------------------------------

class LineItem(BaseModel):
    """A single line item on an invoice."""
    description: str
    quantity: int
    unit_price: float
    extended_price: float  # may contain intentional math errors


class Invoice(BaseModel):
    """An invoice document the agent must review."""
    invoice_id: str
    vendor_name: str
    vendor_id: str
    invoice_date: str
    due_date: str
    po_number: Optional[str] = None
    currency: str = "USD"
    line_items: List[LineItem]
    subtotal: float
    tax_rate: float
    tax_amount: float
    total_amount: float
    notes: Optional[str] = None
    # Hidden ground-truth (not exposed in observation directly)
    _injected_errors: List[Dict[str, Any]] = []


class PurchaseOrder(BaseModel):
    """A purchase order for cross-referencing."""
    po_number: str
    vendor_id: str
    approved_amount: float
    approved_items: List[str]
    status: str  # open, fulfilled, cancelled
    created_date: str


class VendorRecord(BaseModel):
    """Vendor master data."""
    vendor_id: str
    vendor_name: str
    status: str  # active, suspended, blacklisted
    payment_terms: str
    contract_expiry: Optional[str] = None
    risk_rating: str  # low, medium, high


class CompanyPolicy(BaseModel):
    """Procurement policy rules."""
    max_auto_approve: float = 5000.0
    require_po_above: float = 1000.0
    require_dual_approval_above: float = 10000.0
    blocked_vendors: List[str] = Field(default_factory=list)
    max_line_item_variance_pct: float = 10.0


# ---------------------------------------------------------------------------
# OpenEnv interface models
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """What the agent sees at each step."""
    task_id: str
    task_description: str
    step_number: int
    max_steps: int
    invoices: List[Invoice]
    purchase_orders: List[PurchaseOrder]
    vendors: List[VendorRecord]
    policy: CompanyPolicy
    review_status: Dict[str, InvoiceStatus]  # invoice_id -> status
    flagged_errors: Dict[str, List[Dict[str, str]]]  # invoice_id -> errors
    checks_performed: Dict[str, List[str]]  # invoice_id -> list of checks
    done: bool = False
    message: str = ""


class Action(BaseModel):
    """An action the agent sends to the environment."""
    action_type: ActionType
    invoice_id: Optional[str] = None
    error_category: Optional[ErrorCategory] = None
    error_description: Optional[str] = None
    severity: Optional[Severity] = None
    reason: Optional[str] = None


class Reward(BaseModel):
    """Reward signal returned after each step."""
    value: float = Field(ge=-1.0, le=1.0)
    breakdown: Dict[str, float] = Field(default_factory=dict)
    message: str = ""


class EnvState(BaseModel):
    """Full internal state of the environment (for state() endpoint)."""
    task_id: str
    step_number: int
    max_steps: int
    done: bool
    invoices: List[Invoice]
    purchase_orders: List[PurchaseOrder]
    vendors: List[VendorRecord]
    policy: CompanyPolicy
    review_status: Dict[str, InvoiceStatus]
    flagged_errors: Dict[str, List[Dict[str, str]]]
    checks_performed: Dict[str, List[str]]
    ground_truth_errors: Dict[str, List[Dict[str, Any]]]
    ground_truth_dispositions: Dict[str, str]
    cumulative_reward: float
    action_history: List[Dict[str, Any]]


class StepResult(BaseModel):
    """Return value of step()."""
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class TaskInfo(BaseModel):
    """Metadata about a registered task."""
    task_id: str
    name: str
    difficulty: str
    description: str
    max_steps: int
    num_invoices: int
