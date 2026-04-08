"""
Core Invoice Review environment implementing the OpenEnv interface.

Manages state transitions, action processing, reward computation,
and grading for all registered tasks.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from src.data.generator import generate_task_data
from src.graders.grader import InvoiceReviewGrader
from src.models import (
    Action,
    ActionType,
    CompanyPolicy,
    EnvState,
    ErrorCategory,
    Invoice,
    InvoiceStatus,
    Observation,
    PurchaseOrder,
    Reward,
    Severity,
    StepResult,
    VendorRecord,
)
from src.rewards.reward import RewardCalculator
from src.tasks.registry import TASK_REGISTRY, get_task


class InvoiceReviewEnv:
    """OpenEnv-compliant Invoice Review environment.

    Implements:
        reset(task_id) -> Observation
        step(action)   -> StepResult (observation, reward, done, info)
        state()        -> EnvState
    """

    def __init__(self):
        self._task_id: Optional[str] = None
        self._step_number: int = 0
        self._max_steps: int = 0
        self._done: bool = True

        # Data
        self._invoices: List[Invoice] = []
        self._purchase_orders: List[PurchaseOrder] = []
        self._vendors: List[VendorRecord] = []
        self._policy: CompanyPolicy = CompanyPolicy()

        # Agent state
        self._review_status: Dict[str, InvoiceStatus] = {}
        self._flagged_errors: Dict[str, List[Dict[str, str]]] = {}
        self._checks_performed: Dict[str, List[str]] = {}

        # Ground truth
        self._ground_truth_errors: Dict[str, List[Dict[str, Any]]] = {}
        self._ground_truth_dispositions: Dict[str, str] = {}

        # History & rewards
        self._action_history: List[Dict[str, Any]] = []
        self._reward_calc: Optional[RewardCalculator] = None
        self._cumulative_reward: float = 0.0
        self._message: str = ""

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(self, task_id: str = "easy") -> Observation:
        """Initialize a new episode for the given task.

        Args:
            task_id: One of 'easy', 'medium', 'hard'.

        Returns:
            Initial observation.
        """
        task_info = get_task(task_id)
        task_def = TASK_REGISTRY[task_id]
        data = generate_task_data(task_def["difficulty"], seed=task_def["seed"])

        self._task_id = task_id
        self._step_number = 0
        self._max_steps = task_info.max_steps
        self._done = False

        self._invoices = data["invoices"]
        self._purchase_orders = data["purchase_orders"]
        self._vendors = data["vendors"]
        self._policy = data["policy"]

        # Initialize tracking
        self._review_status = {
            inv.invoice_id: InvoiceStatus.PENDING for inv in self._invoices
        }
        self._flagged_errors = {inv.invoice_id: [] for inv in self._invoices}
        self._checks_performed = {inv.invoice_id: [] for inv in self._invoices}

        self._ground_truth_errors = data["ground_truth_errors"]
        self._ground_truth_dispositions = data["ground_truth_dispositions"]

        self._action_history = []
        self._cumulative_reward = 0.0
        self._message = (
            f"Episode started. Task: {task_info.name} ({task_info.difficulty}). "
            f"Review {task_info.num_invoices} invoice(s) within {task_info.max_steps} steps. "
            f"Use inspect_invoice, check_vendor, check_po_match, check_duplicate "
            f"to investigate, flag_error to report issues, then approve/reject/escalate "
            f"each invoice, and finally submit_review when done."
        )

        self._reward_calc = RewardCalculator(
            ground_truth_errors=self._ground_truth_errors,
            ground_truth_dispositions=self._ground_truth_dispositions,
            num_invoices=task_info.num_invoices,
        )

        return self._build_observation()

    def step(self, action: Action) -> StepResult:
        """Process an agent action and advance the environment.

        Args:
            action: The action to take.

        Returns:
            StepResult with observation, reward, done flag, and info dict.
        """
        if self._done:
            return StepResult(
                observation=self._build_observation(),
                reward=Reward(value=0.0, message="Episode already ended."),
                done=True,
                info={"error": "Episode already ended. Call reset() first."},
            )

        self._step_number += 1

        # Process the action
        action_valid, result_msg = self._process_action(action)

        # Compute reward
        reward_data = self._reward_calc.compute_reward(
            action_type=action.action_type,
            invoice_id=action.invoice_id,
            error_category=action.error_category.value if action.error_category else None,
            action_valid=action_valid,
            action_result=result_msg,
        )
        reward_value = reward_data.pop("value")
        self._cumulative_reward += reward_value

        self._message = result_msg

        # Record action
        self._action_history.append({
            "step": self._step_number,
            "action": action.model_dump(),
            "valid": action_valid,
            "result": result_msg,
            "reward": reward_value,
        })

        # Check termination
        info: Dict[str, Any] = {"action_valid": action_valid}

        if action.action_type == ActionType.SUBMIT_REVIEW and action_valid:
            self._done = True
            final_score, breakdown = self._run_grader()
            info["final_score"] = final_score
            info["grader_breakdown"] = breakdown
            info["cumulative_reward"] = self._cumulative_reward
            self._message = (
                f"Review submitted. Final score: {final_score:.3f}. "
                f"Cumulative reward: {self._cumulative_reward:.3f}."
            )

        elif self._step_number >= self._max_steps:
            self._done = True
            final_score, breakdown = self._run_grader()
            # Penalty for running out of steps
            final_score = max(0.0, final_score * 0.7)
            info["final_score"] = final_score
            info["grader_breakdown"] = breakdown
            info["timeout"] = True
            info["cumulative_reward"] = self._cumulative_reward
            self._message = (
                f"Step limit reached. Auto-submitting. "
                f"Final score: {final_score:.3f} (includes timeout penalty). "
                f"Cumulative reward: {self._cumulative_reward:.3f}."
            )

        reward = Reward(
            value=max(-1.0, min(1.0, reward_value)),
            breakdown=reward_data,
            message=self._message,
        )

        return StepResult(
            observation=self._build_observation(),
            reward=reward,
            done=self._done,
            info=info,
        )

    def state(self) -> EnvState:
        """Return the full internal state (for debugging / inspection)."""
        return EnvState(
            task_id=self._task_id or "",
            step_number=self._step_number,
            max_steps=self._max_steps,
            done=self._done,
            invoices=self._invoices,
            purchase_orders=self._purchase_orders,
            vendors=self._vendors,
            policy=self._policy,
            review_status={k: v.value for k, v in self._review_status.items()},
            flagged_errors=self._flagged_errors,
            checks_performed=self._checks_performed,
            ground_truth_errors=self._ground_truth_errors,
            ground_truth_dispositions=self._ground_truth_dispositions,
            cumulative_reward=self._cumulative_reward,
            action_history=self._action_history,
        )

    # ------------------------------------------------------------------
    # Action processing
    # ------------------------------------------------------------------

    def _process_action(self, action: Action) -> Tuple[bool, str]:
        """Execute an action and return (valid, result_message)."""
        at = action.action_type
        inv_id = action.invoice_id

        # Validate invoice target for actions that need it
        needs_target = at not in (ActionType.SUBMIT_REVIEW,)
        if needs_target:
            if not inv_id:
                return False, "Action requires an invoice_id."
            valid_ids = {inv.invoice_id for inv in self._invoices}
            if inv_id not in valid_ids:
                return False, f"Unknown invoice_id: {inv_id!r}."

        # Process by type
        if at == ActionType.INSPECT_INVOICE:
            return self._action_inspect(inv_id)
        elif at == ActionType.FLAG_ERROR:
            return self._action_flag_error(action)
        elif at == ActionType.CHECK_PO_MATCH:
            return self._action_check_po(inv_id)
        elif at == ActionType.CHECK_VENDOR:
            return self._action_check_vendor(inv_id)
        elif at == ActionType.CHECK_DUPLICATE:
            return self._action_check_duplicate(inv_id)
        elif at == ActionType.APPROVE:
            return self._action_disposition(inv_id, InvoiceStatus.APPROVED, action.reason)
        elif at == ActionType.REJECT:
            return self._action_disposition(inv_id, InvoiceStatus.REJECTED, action.reason)
        elif at == ActionType.ESCALATE:
            return self._action_disposition(inv_id, InvoiceStatus.ESCALATED, action.reason)
        elif at == ActionType.SUBMIT_REVIEW:
            return self._action_submit()
        else:
            return False, f"Unknown action type: {at!r}."

    def _action_inspect(self, inv_id: str) -> Tuple[bool, str]:
        if "inspect_invoice" not in self._checks_performed[inv_id]:
            self._checks_performed[inv_id].append("inspect_invoice")

        inv = self._get_invoice(inv_id)
        items_text = "; ".join(
            f"{li.description}: qty={li.quantity}, unit=${li.unit_price:.2f}, ext=${li.extended_price:.2f}"
            for li in inv.line_items
        )
        return True, (
            f"Invoice {inv_id} from {inv.vendor_name} ({inv.vendor_id}). "
            f"Date: {inv.invoice_date}, Due: {inv.due_date}, PO: {inv.po_number or 'N/A'}. "
            f"Line items: [{items_text}]. "
            f"Subtotal: ${inv.subtotal:.2f}, Tax ({inv.tax_rate*100:.0f}%): ${inv.tax_amount:.2f}, "
            f"Total: ${inv.total_amount:.2f}. Notes: {inv.notes or 'None'}."
        )

    def _action_flag_error(self, action: Action) -> Tuple[bool, str]:
        inv_id = action.invoice_id
        if not action.error_category:
            return False, "flag_error requires error_category."
        if not action.error_description:
            return False, "flag_error requires error_description."

        severity = action.severity.value if action.severity else Severity.MEDIUM.value

        self._flagged_errors[inv_id].append({
            "category": action.error_category.value,
            "description": action.error_description,
            "severity": severity,
        })

        return True, (
            f"Flagged {action.error_category.value} on {inv_id} "
            f"(severity: {severity}): {action.error_description}"
        )

    def _action_check_po(self, inv_id: str) -> Tuple[bool, str]:
        if "check_po_match" not in self._checks_performed[inv_id]:
            self._checks_performed[inv_id].append("check_po_match")

        inv = self._get_invoice(inv_id)
        if not inv.po_number:
            return True, (
                f"Invoice {inv_id}: No purchase order number provided. "
                f"Policy requires PO for invoices above ${self._policy.require_po_above:,.2f}. "
                f"Invoice total: ${inv.total_amount:,.2f}."
            )

        # Find matching PO
        matching_po = None
        for po in self._purchase_orders:
            if po.po_number == inv.po_number:
                matching_po = po
                break

        if not matching_po:
            return True, (
                f"Invoice {inv_id}: PO {inv.po_number} not found in system."
            )

        # Check vendor match
        vendor_match = matching_po.vendor_id == inv.vendor_id
        # Check amount
        amount_ok = inv.total_amount <= matching_po.approved_amount
        # Check items
        inv_items = {li.description for li in inv.line_items}
        po_items = set(matching_po.approved_items)
        # Fuzzy match: check if PO items are substrings of invoice items
        unmatched_items = []
        for inv_item in inv_items:
            matched = any(
                po_item.lower() in inv_item.lower() or inv_item.lower() in po_item.lower()
                for po_item in po_items
            )
            if not matched:
                unmatched_items.append(inv_item)

        return True, (
            f"PO Match for {inv_id} (PO: {inv.po_number}): "
            f"Vendor match: {'✓' if vendor_match else '✗ (PO vendor: ' + matching_po.vendor_id + ')'}. "
            f"Amount: invoice ${inv.total_amount:,.2f} vs PO approved ${matching_po.approved_amount:,.2f} "
            f"({'✓ within limit' if amount_ok else '✗ EXCEEDS by $' + f'{inv.total_amount - matching_po.approved_amount:,.2f}'}). "
            f"PO Status: {matching_po.status}. "
            f"Items not on PO: {unmatched_items if unmatched_items else 'none (all matched)'}."
        )

    def _action_check_vendor(self, inv_id: str) -> Tuple[bool, str]:
        if "check_vendor" not in self._checks_performed[inv_id]:
            self._checks_performed[inv_id].append("check_vendor")

        inv = self._get_invoice(inv_id)
        vendor = None
        for v in self._vendors:
            if v.vendor_id == inv.vendor_id:
                vendor = v
                break

        if not vendor:
            return True, (
                f"Vendor check for {inv_id}: Vendor {inv.vendor_id} "
                f"({inv.vendor_name}) NOT FOUND in vendor master data."
            )

        is_blocked = inv.vendor_id in self._policy.blocked_vendors
        return True, (
            f"Vendor check for {inv_id}: {vendor.vendor_name} ({vendor.vendor_id}). "
            f"Status: {vendor.status}. Risk: {vendor.risk_rating}. "
            f"Payment terms: {vendor.payment_terms}. "
            f"Contract expiry: {vendor.contract_expiry or 'N/A'}. "
            f"On blocked list: {'YES ⚠' if is_blocked else 'No'}."
        )

    def _action_check_duplicate(self, inv_id: str) -> Tuple[bool, str]:
        if "check_duplicate" not in self._checks_performed[inv_id]:
            self._checks_performed[inv_id].append("check_duplicate")

        inv = self._get_invoice(inv_id)
        duplicates = []
        for other in self._invoices:
            if other.invoice_id == inv_id:
                continue
            # Check for duplicate indicators
            same_vendor = other.vendor_id == inv.vendor_id
            same_amount = abs(other.total_amount - inv.total_amount) < 0.01
            same_date = other.invoice_date == inv.invoice_date
            same_po = other.po_number and other.po_number == inv.po_number

            if same_vendor and same_amount and (same_date or same_po):
                duplicates.append(other.invoice_id)

        if duplicates:
            return True, (
                f"Duplicate check for {inv_id}: POTENTIAL DUPLICATES FOUND — "
                f"{duplicates}. Same vendor, amount, and date/PO."
            )
        return True, f"Duplicate check for {inv_id}: No duplicates detected."

    def _action_disposition(
        self, inv_id: str, status: InvoiceStatus, reason: Optional[str]
    ) -> Tuple[bool, str]:
        if self._review_status[inv_id] != InvoiceStatus.PENDING:
            return True, (
                f"Invoice {inv_id} already dispositioned as "
                f"{self._review_status[inv_id].value}. Overwriting to {status.value}."
            )
        self._review_status[inv_id] = status
        return True, (
            f"Invoice {inv_id} marked as {status.value}. "
            f"Reason: {reason or 'No reason provided'}."
        )

    def _action_submit(self) -> Tuple[bool, str]:
        pending = [
            inv_id for inv_id, s in self._review_status.items()
            if s == InvoiceStatus.PENDING
        ]
        if pending:
            return True, (
                f"Review submitted with {len(pending)} invoice(s) still pending: "
                f"{pending}. These will default to 'pending' in grading."
            )
        return True, "Review submitted. All invoices have been dispositioned."

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_invoice(self, inv_id: str) -> Invoice:
        for inv in self._invoices:
            if inv.invoice_id == inv_id:
                return inv
        raise ValueError(f"Invoice {inv_id} not found.")

    def _build_observation(self) -> Observation:
        return Observation(
            task_id=self._task_id or "",
            task_description=get_task(self._task_id).description if self._task_id else "",
            step_number=self._step_number,
            max_steps=self._max_steps,
            invoices=self._invoices,
            purchase_orders=self._purchase_orders,
            vendors=self._vendors,
            policy=self._policy,
            review_status={k: v.value for k, v in self._review_status.items()},
            flagged_errors=self._flagged_errors,
            checks_performed=self._checks_performed,
            done=self._done,
            message=self._message,
        )

    def _run_grader(self) -> Tuple[float, Dict[str, float]]:
        grader = InvoiceReviewGrader(
            difficulty=TASK_REGISTRY[self._task_id]["difficulty"]
        )
        return grader.grade(
            ground_truth_errors=self._ground_truth_errors,
            ground_truth_dispositions=self._ground_truth_dispositions,
            flagged_errors=self._flagged_errors,
            review_status={k: v.value for k, v in self._review_status.items()},
            checks_performed=self._checks_performed,
            steps_used=self._step_number,
            max_steps=self._max_steps,
            num_invoices=len(self._invoices),
        )
