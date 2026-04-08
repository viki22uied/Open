"""
Deterministic grader for the Invoice Review environment.

Computes a final score in [0.0, 1.0] based on:
  1. Error detection accuracy (precision + recall)
  2. Disposition correctness (approve/reject/escalate)
  3. Due-diligence completeness (checks performed)
  4. Efficiency (step usage relative to budget)

Weights are tuned per difficulty level.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple


class InvoiceReviewGrader:
    """Deterministic grader that scores agent performance on a task."""

    # Scoring weight profiles by difficulty
    _WEIGHTS = {
        "easy": {
            "error_detection": 0.40,
            "disposition": 0.35,
            "due_diligence": 0.15,
            "efficiency": 0.10,
        },
        "medium": {
            "error_detection": 0.30,
            "disposition": 0.30,
            "due_diligence": 0.25,
            "efficiency": 0.15,
        },
        "hard": {
            "error_detection": 0.25,
            "disposition": 0.30,
            "due_diligence": 0.25,
            "efficiency": 0.20,
        },
    }

    def __init__(self, difficulty: str = "easy"):
        if difficulty not in self._WEIGHTS:
            raise ValueError(f"Unknown difficulty: {difficulty!r}")
        self.difficulty = difficulty
        self.weights = self._WEIGHTS[difficulty]

    def grade(
        self,
        ground_truth_errors: Dict[str, List[Dict[str, Any]]],
        ground_truth_dispositions: Dict[str, str],
        flagged_errors: Dict[str, List[Dict[str, str]]],
        review_status: Dict[str, str],
        checks_performed: Dict[str, List[str]],
        steps_used: int,
        max_steps: int,
        num_invoices: int,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute final score and component breakdown.

        Returns:
            (score, breakdown) where score is in [0.0, 1.0] and breakdown
            maps component names to their weighted contributions.
        """
        error_score = self._score_error_detection(
            ground_truth_errors, flagged_errors
        )
        disp_score = self._score_dispositions(
            ground_truth_dispositions, review_status
        )
        diligence_score = self._score_due_diligence(
            ground_truth_errors, checks_performed, num_invoices
        )
        efficiency_score = self._score_efficiency(steps_used, max_steps)

        breakdown = {
            "error_detection": error_score,
            "disposition": disp_score,
            "due_diligence": diligence_score,
            "efficiency": efficiency_score,
        }

        final = sum(
            self.weights[k] * breakdown[k] for k in self.weights
        )
        final = max(0.0, min(1.0, final))

        weighted_breakdown = {
            f"{k}_raw": v for k, v in breakdown.items()
        }
        weighted_breakdown.update({
            f"{k}_weighted": self.weights[k] * breakdown[k]
            for k in self.weights
        })
        weighted_breakdown["final_score"] = final

        return final, weighted_breakdown

    # ------------------------------------------------------------------
    # Component scorers
    # ------------------------------------------------------------------

    def _score_error_detection(
        self,
        ground_truth: Dict[str, List[Dict[str, Any]]],
        flagged: Dict[str, List[Dict[str, str]]],
    ) -> float:
        """Score based on precision and recall of flagged errors.

        For each invoice, we check how many ground-truth error categories
        were correctly detected and whether any false positives were flagged.
        """
        if not ground_truth:
            return 1.0  # No errors to find

        total_gt_errors = 0
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for inv_id, gt_errors in ground_truth.items():
            gt_categories: Set[str] = {e["category"] for e in gt_errors}
            total_gt_errors += len(gt_categories)

            flagged_items = flagged.get(inv_id, [])
            flagged_categories: Set[str] = set()
            for f in flagged_items:
                cat = f.get("category", f.get("error_category", ""))
                if cat:
                    flagged_categories.add(cat)

            # True positives: categories in both GT and flagged
            tp = gt_categories & flagged_categories
            true_positives += len(tp)

            # False positives: flagged but not in GT
            fp = flagged_categories - gt_categories
            false_positives += len(fp)

            # False negatives: in GT but not flagged
            fn = gt_categories - flagged_categories
            false_negatives += len(fn)

        # Also handle invoices the agent flagged errors on that have no GT errors
        for inv_id, flagged_items in flagged.items():
            if inv_id not in ground_truth or not ground_truth[inv_id]:
                false_positives += len(flagged_items)

        if total_gt_errors == 0 and false_positives == 0:
            return 1.0
        if total_gt_errors == 0 and false_positives > 0:
            return max(0.0, 1.0 - 0.2 * false_positives)

        precision = true_positives / max(1, true_positives + false_positives)
        recall = true_positives / max(1, total_gt_errors)
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        # Penalize false positives slightly less than missed errors
        fp_penalty = 0.05 * false_positives
        return max(0.0, min(1.0, f1 - fp_penalty))

    def _score_dispositions(
        self,
        ground_truth: Dict[str, str],
        review_status: Dict[str, str],
    ) -> float:
        """Score based on correct final disposition of each invoice."""
        if not ground_truth:
            return 1.0

        correct = 0
        total = len(ground_truth)

        for inv_id, expected in ground_truth.items():
            actual = review_status.get(inv_id, "pending")
            if actual == expected:
                correct += 1
            elif actual == "pending":
                # Didn't make a decision — partial penalty
                correct += 0.0
            elif actual == "escalated" and expected == "rejected":
                # Escalating what should be rejected is cautious — partial credit
                correct += 0.3
            elif actual == "rejected" and expected == "escalated":
                # Rejecting what should be escalated — some credit
                correct += 0.2

        return correct / total if total > 0 else 1.0

    def _score_due_diligence(
        self,
        ground_truth_errors: Dict[str, List[Dict[str, Any]]],
        checks_performed: Dict[str, List[str]],
        num_invoices: int,
    ) -> float:
        """Score based on completeness of verification checks.

        We expect agents to:
        - Inspect every invoice
        - Check vendors for invoices with vendor issues
        - Check PO matching for invoices with POs
        - Check duplicates when relevant
        """
        if num_invoices == 0:
            return 1.0

        expected_checks_per_invoice = 1.0  # At least inspect
        completed_checks = 0.0
        total_expected = 0.0

        all_invoice_ids = set(ground_truth_errors.keys()) | set(checks_performed.keys())

        for inv_id in all_invoice_ids:
            total_expected += 1.0  # inspect_invoice
            checks = set(checks_performed.get(inv_id, []))

            if "inspect_invoice" in checks:
                completed_checks += 1.0

            # If there are vendor-related errors, vendor check is expected
            gt_errors = ground_truth_errors.get(inv_id, [])
            gt_categories = {e["category"] for e in gt_errors}

            if "vendor_issue" in gt_categories:
                total_expected += 1.0
                if "check_vendor" in checks:
                    completed_checks += 1.0
            else:
                # Still good practice to check vendor
                total_expected += 0.5
                if "check_vendor" in checks:
                    completed_checks += 0.5

            if "po_mismatch" in gt_categories or "missing_field" in gt_categories:
                total_expected += 1.0
                if "check_po_match" in checks:
                    completed_checks += 1.0
            else:
                total_expected += 0.3
                if "check_po_match" in checks:
                    completed_checks += 0.3

            if "duplicate" in gt_categories:
                total_expected += 1.0
                if "check_duplicate" in checks:
                    completed_checks += 1.0

        return completed_checks / total_expected if total_expected > 0 else 1.0

    def _score_efficiency(self, steps_used: int, max_steps: int) -> float:
        """Score based on efficient use of step budget.

        Perfect: use ~40-70% of budget
        Good: use <40% or 70-85%
        Poor: use >85% (running out of time)
        """
        if max_steps <= 0:
            return 0.5

        ratio = steps_used / max_steps

        if ratio <= 0:
            return 0.0  # Didn't do anything
        elif ratio <= 0.3:
            # Very fast — might have skipped things, but efficient
            return 0.7
        elif ratio <= 0.5:
            return 0.9
        elif ratio <= 0.7:
            return 1.0  # Sweet spot
        elif ratio <= 0.85:
            return 0.8
        elif ratio <= 0.95:
            return 0.5
        else:
            return 0.3  # Nearly exhausted budget
