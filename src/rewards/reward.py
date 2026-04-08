"""
Dense reward calculator for the Invoice Review environment.

Provides step-by-step reward signal that:
  - Rewards meaningful investigation actions (+)
  - Rewards correct error flagging (+)
  - Rewards correct dispositions (+)
  - Penalizes repeated/redundant actions (-)
  - Penalizes invalid actions (-)
  - Penalizes premature submission (-)
  - Penalizes step waste / doing nothing useful (-)

All per-step rewards are in [-1.0, 1.0].
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from src.models import ActionType, ErrorCategory, InvoiceStatus


class RewardCalculator:
    """Computes shaped per-step rewards."""

    # Base rewards for different action types
    _ACTION_BASE_REWARDS = {
        ActionType.INSPECT_INVOICE: 0.05,
        ActionType.FLAG_ERROR: 0.0,       # Depends on correctness
        ActionType.CHECK_PO_MATCH: 0.04,
        ActionType.CHECK_VENDOR: 0.04,
        ActionType.CHECK_DUPLICATE: 0.04,
        ActionType.APPROVE: 0.0,          # Depends on correctness
        ActionType.REJECT: 0.0,           # Depends on correctness
        ActionType.ESCALATE: 0.0,         # Depends on correctness
        ActionType.SUBMIT_REVIEW: 0.0,    # Terminal, scored by grader
    }

    def __init__(
        self,
        ground_truth_errors: Dict[str, List[Dict[str, Any]]],
        ground_truth_dispositions: Dict[str, str],
        num_invoices: int,
    ):
        self.ground_truth_errors = ground_truth_errors
        self.ground_truth_dispositions = ground_truth_dispositions
        self.num_invoices = num_invoices

        # Track what has been done to detect repetition
        self._inspected: Set[str] = set()
        self._checks_done: Set[str] = set()  # "inv_id:check_type"
        self._flagged: Dict[str, Set[str]] = {}  # inv_id -> set of categories
        self._dispositioned: Set[str] = set()
        self._total_reward = 0.0
        self._step_count = 0
        self._consecutive_invalid = 0

    def compute_reward(
        self,
        action_type: ActionType,
        invoice_id: Optional[str],
        error_category: Optional[str],
        action_valid: bool,
        action_result: str,
    ) -> Dict[str, float]:
        """Compute reward for a single step.

        Args:
            action_type: The action taken.
            invoice_id: Target invoice (if applicable).
            error_category: Error category flagged (if applicable).
            action_valid: Whether the action was structurally valid.
            action_result: Result message from environment.

        Returns:
            Dict with 'value' (total step reward) and component breakdown.
        """
        self._step_count += 1
        breakdown: Dict[str, float] = {}

        # ------------------------------------------------------------------
        # Invalid action penalty
        # ------------------------------------------------------------------
        if not action_valid:
            self._consecutive_invalid += 1
            penalty = -0.10 * self._consecutive_invalid  # Escalating penalty
            breakdown["invalid_action_penalty"] = max(-0.5, penalty)
            total = sum(breakdown.values())
            total = max(-1.0, min(1.0, total))
            self._total_reward += total
            return {"value": total, **breakdown}

        self._consecutive_invalid = 0  # Reset on valid action

        # ------------------------------------------------------------------
        # Base reward for the action type
        # ------------------------------------------------------------------
        base = self._ACTION_BASE_REWARDS.get(action_type, 0.0)

        # ------------------------------------------------------------------
        # Action-specific rewards
        # ------------------------------------------------------------------

        if action_type == ActionType.INSPECT_INVOICE:
            if invoice_id and invoice_id not in self._inspected:
                self._inspected.add(invoice_id)
                breakdown["inspection_reward"] = base
            elif invoice_id and invoice_id in self._inspected:
                breakdown["redundant_action_penalty"] = -0.03
            else:
                breakdown["missing_target_penalty"] = -0.05

        elif action_type in (
            ActionType.CHECK_PO_MATCH,
            ActionType.CHECK_VENDOR,
            ActionType.CHECK_DUPLICATE,
        ):
            check_key = f"{invoice_id}:{action_type.value}"
            if invoice_id and check_key not in self._checks_done:
                self._checks_done.add(check_key)
                # Bonus if this check is relevant to the invoice's errors
                gt_errors = self.ground_truth_errors.get(invoice_id, [])
                gt_cats = {e["category"] for e in gt_errors}
                relevant_map = {
                    ActionType.CHECK_PO_MATCH: {"po_mismatch", "missing_field"},
                    ActionType.CHECK_VENDOR: {"vendor_issue"},
                    ActionType.CHECK_DUPLICATE: {"duplicate"},
                }
                relevant_cats = relevant_map.get(action_type, set())
                if gt_cats & relevant_cats:
                    breakdown["relevant_check_reward"] = base + 0.06
                else:
                    breakdown["check_reward"] = base
            elif invoice_id and check_key in self._checks_done:
                breakdown["redundant_action_penalty"] = -0.03
            else:
                breakdown["missing_target_penalty"] = -0.05

        elif action_type == ActionType.FLAG_ERROR:
            if invoice_id and error_category:
                inv_flags = self._flagged.setdefault(invoice_id, set())
                if error_category in inv_flags:
                    breakdown["redundant_flag_penalty"] = -0.05
                else:
                    inv_flags.add(error_category)
                    # Check if this is a true positive
                    gt_errors = self.ground_truth_errors.get(invoice_id, [])
                    gt_cats = {e["category"] for e in gt_errors}
                    if error_category in gt_cats:
                        breakdown["correct_flag_reward"] = 0.15
                    else:
                        # False positive
                        breakdown["false_flag_penalty"] = -0.08
            else:
                breakdown["incomplete_flag_penalty"] = -0.05

        elif action_type in (
            ActionType.APPROVE,
            ActionType.REJECT,
            ActionType.ESCALATE,
        ):
            if invoice_id:
                if invoice_id in self._dispositioned:
                    breakdown["redundant_disposition_penalty"] = -0.05
                else:
                    self._dispositioned.add(invoice_id)
                    expected = self.ground_truth_dispositions.get(invoice_id)
                    actual = action_type.value.replace("approve", "approved").replace(
                        "reject", "rejected"
                    ).replace("escalate", "escalated")
                    # Map action to disposition
                    action_to_disp = {
                        ActionType.APPROVE: "approved",
                        ActionType.REJECT: "rejected",
                        ActionType.ESCALATE: "escalated",
                    }
                    actual = action_to_disp[action_type]

                    if actual == expected:
                        breakdown["correct_disposition_reward"] = 0.20
                    elif actual == "escalated" and expected == "rejected":
                        breakdown["cautious_disposition_reward"] = 0.05
                    elif actual == "rejected" and expected == "escalated":
                        breakdown["harsh_disposition_penalty"] = -0.05
                    elif actual == "approved" and expected in ("rejected", "escalated"):
                        breakdown["wrong_approval_penalty"] = -0.20
                    else:
                        breakdown["wrong_disposition_penalty"] = -0.10

                    # Bonus for performing checks before disposition
                    checks_for_inv = sum(
                        1 for k in self._checks_done if k.startswith(f"{invoice_id}:")
                    )
                    inspected = invoice_id in self._inspected
                    if inspected and checks_for_inv >= 1:
                        breakdown["informed_decision_bonus"] = 0.05
                    elif not inspected:
                        breakdown["uninformed_decision_penalty"] = -0.08
            else:
                breakdown["missing_target_penalty"] = -0.05

        elif action_type == ActionType.SUBMIT_REVIEW:
            # Check completeness — penalize if not all invoices dispositioned
            undispositioned = set(self.ground_truth_dispositions.keys()) - self._dispositioned
            if undispositioned:
                pct_incomplete = len(undispositioned) / max(1, len(self.ground_truth_dispositions))
                breakdown["premature_submission_penalty"] = -0.15 * pct_incomplete
            else:
                breakdown["complete_submission_reward"] = 0.10

        # ------------------------------------------------------------------
        # Aggregate
        # ------------------------------------------------------------------
        total = sum(breakdown.values())
        total = max(-1.0, min(1.0, total))
        self._total_reward += total

        return {"value": total, **breakdown}

    @property
    def cumulative_reward(self) -> float:
        return self._total_reward
