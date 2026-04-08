"""
Comprehensive tests for the Invoice Review OpenEnv environment.

Tests cover:
  - Environment reset/step/state lifecycle
  - All action types
  - Reward signal correctness
  - Grader determinism
  - Edge cases and error handling
  - All three task difficulty levels
"""

from __future__ import annotations

import pytest

from src.environment import InvoiceReviewEnv
from src.graders.grader import InvoiceReviewGrader
from src.models import Action, ActionType, ErrorCategory, Severity
from src.tasks.registry import TASK_REGISTRY, get_task, list_tasks


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env():
    return InvoiceReviewEnv()


# ---------------------------------------------------------------------------
# Task Registry Tests
# ---------------------------------------------------------------------------

class TestTaskRegistry:
    def test_list_tasks_returns_three(self):
        tasks = list_tasks()
        assert len(tasks) == 3

    def test_task_ids(self):
        assert set(TASK_REGISTRY.keys()) == {"easy", "medium", "hard"}

    def test_get_task_easy(self):
        t = get_task("easy")
        assert t.task_id == "easy"
        assert t.difficulty == "easy"
        assert t.num_invoices == 1
        assert t.max_steps == 15

    def test_get_task_medium(self):
        t = get_task("medium")
        assert t.num_invoices == 3
        assert t.max_steps == 30

    def test_get_task_hard(self):
        t = get_task("hard")
        assert t.num_invoices == 5
        assert t.max_steps == 50

    def test_get_task_invalid(self):
        with pytest.raises(ValueError):
            get_task("impossible")


# ---------------------------------------------------------------------------
# Environment Reset Tests
# ---------------------------------------------------------------------------

class TestEnvReset:
    def test_reset_easy(self, env):
        obs = env.reset("easy")
        assert obs.task_id == "easy"
        assert obs.step_number == 0
        assert obs.max_steps == 15
        assert len(obs.invoices) == 1
        assert not obs.done

    def test_reset_medium(self, env):
        obs = env.reset("medium")
        assert len(obs.invoices) == 3
        assert obs.max_steps == 30

    def test_reset_hard(self, env):
        obs = env.reset("hard")
        assert len(obs.invoices) == 5
        assert obs.max_steps == 50

    def test_reset_clears_state(self, env):
        obs = env.reset("easy")
        # Take a step
        action = Action(
            action_type=ActionType.INSPECT_INVOICE,
            invoice_id=obs.invoices[0].invoice_id,
        )
        env.step(action)

        # Reset should clear everything
        obs2 = env.reset("easy")
        assert obs2.step_number == 0
        assert all(
            s == "pending" for s in obs2.review_status.values()
        )

    def test_reset_invalid_task(self, env):
        with pytest.raises(ValueError):
            env.reset("nonexistent")

    def test_reset_review_status_all_pending(self, env):
        obs = env.reset("medium")
        for inv_id, status in obs.review_status.items():
            assert status == "pending"


# ---------------------------------------------------------------------------
# Environment Step Tests
# ---------------------------------------------------------------------------

class TestEnvStep:
    def test_step_increments(self, env):
        obs = env.reset("easy")
        inv_id = obs.invoices[0].invoice_id
        result = env.step(Action(
            action_type=ActionType.INSPECT_INVOICE,
            invoice_id=inv_id,
        ))
        assert result.observation.step_number == 1
        assert not result.done

    def test_step_after_done(self, env):
        obs = env.reset("easy")
        inv_id = obs.invoices[0].invoice_id
        # Approve and submit
        env.step(Action(action_type=ActionType.APPROVE, invoice_id=inv_id, reason="test"))
        result = env.step(Action(action_type=ActionType.SUBMIT_REVIEW))
        assert result.done

        # Step after done should return zero reward
        result2 = env.step(Action(action_type=ActionType.INSPECT_INVOICE, invoice_id=inv_id))
        assert result2.done
        assert result2.reward.value == 0.0

    def test_inspect_returns_details(self, env):
        obs = env.reset("easy")
        inv_id = obs.invoices[0].invoice_id
        result = env.step(Action(
            action_type=ActionType.INSPECT_INVOICE,
            invoice_id=inv_id,
        ))
        assert inv_id in result.reward.message or inv_id in result.observation.message

    def test_invalid_invoice_id(self, env):
        env.reset("easy")
        result = env.step(Action(
            action_type=ActionType.INSPECT_INVOICE,
            invoice_id="FAKE-ID",
        ))
        assert result.reward.value < 0  # Should be penalized

    def test_flag_error_valid(self, env):
        obs = env.reset("easy")
        inv_id = obs.invoices[0].invoice_id
        result = env.step(Action(
            action_type=ActionType.FLAG_ERROR,
            invoice_id=inv_id,
            error_category=ErrorCategory.MATH_ERROR,
            error_description="Extended price is wrong",
            severity=Severity.MEDIUM,
        ))
        assert result.reward.value > 0  # Correct flag should be rewarded
        assert len(result.observation.flagged_errors[inv_id]) == 1

    def test_flag_error_false_positive(self, env):
        obs = env.reset("easy")
        inv_id = obs.invoices[0].invoice_id
        result = env.step(Action(
            action_type=ActionType.FLAG_ERROR,
            invoice_id=inv_id,
            error_category=ErrorCategory.DUPLICATE,  # Not actually a duplicate
            error_description="Looks like a duplicate",
            severity=Severity.HIGH,
        ))
        assert result.reward.value < 0  # False positive should be penalized

    def test_approve_correct(self, env):
        obs = env.reset("medium")
        # INV-2025-0010 should be approved
        result = env.step(Action(
            action_type=ActionType.INSPECT_INVOICE,
            invoice_id="INV-2025-0010",
        ))
        result = env.step(Action(
            action_type=ActionType.CHECK_VENDOR,
            invoice_id="INV-2025-0010",
        ))
        result = env.step(Action(
            action_type=ActionType.APPROVE,
            invoice_id="INV-2025-0010",
            reason="Clean invoice",
        ))
        # Should get positive reward for correct disposition + informed decision
        assert result.reward.value > 0

    def test_reject_correct(self, env):
        obs = env.reset("easy")
        inv_id = obs.invoices[0].invoice_id
        env.step(Action(action_type=ActionType.INSPECT_INVOICE, invoice_id=inv_id))
        result = env.step(Action(
            action_type=ActionType.REJECT,
            invoice_id=inv_id,
            reason="Math errors found",
        ))
        # INV-2025-0001 should be rejected — positive reward
        assert result.reward.value > 0

    def test_submit_completes_episode(self, env):
        obs = env.reset("easy")
        inv_id = obs.invoices[0].invoice_id
        env.step(Action(action_type=ActionType.REJECT, invoice_id=inv_id, reason="errors"))
        result = env.step(Action(action_type=ActionType.SUBMIT_REVIEW))
        assert result.done
        assert "final_score" in result.info
        assert 0.0 <= result.info["final_score"] <= 1.0

    def test_step_limit_triggers_done(self, env):
        obs = env.reset("easy")
        inv_id = obs.invoices[0].invoice_id
        # Exhaust all steps
        result = None
        for _ in range(15):
            result = env.step(Action(
                action_type=ActionType.INSPECT_INVOICE,
                invoice_id=inv_id,
            ))
            if result.done:
                break
        assert result.done
        assert "final_score" in result.info


# ---------------------------------------------------------------------------
# Check Actions Tests
# ---------------------------------------------------------------------------

class TestCheckActions:
    def test_check_vendor(self, env):
        obs = env.reset("medium")
        result = env.step(Action(
            action_type=ActionType.CHECK_VENDOR,
            invoice_id="INV-2025-0011",
        ))
        assert "suspended" in result.observation.message.lower() or \
               "ShadowCorp" in result.observation.message

    def test_check_po_match(self, env):
        obs = env.reset("medium")
        result = env.step(Action(
            action_type=ActionType.CHECK_PO_MATCH,
            invoice_id="INV-2025-0011",
        ))
        assert "PO" in result.observation.message

    def test_check_po_missing(self, env):
        obs = env.reset("medium")
        result = env.step(Action(
            action_type=ActionType.CHECK_PO_MATCH,
            invoice_id="INV-2025-0012",
        ))
        assert "No purchase order" in result.observation.message

    def test_check_duplicate(self, env):
        obs = env.reset("hard")
        result = env.step(Action(
            action_type=ActionType.CHECK_DUPLICATE,
            invoice_id="INV-2025-0102",
        ))
        assert "duplicate" in result.observation.message.lower() or \
               "INV-2025-0100" in result.observation.message

    def test_check_no_duplicate(self, env):
        obs = env.reset("easy")
        inv_id = obs.invoices[0].invoice_id
        result = env.step(Action(
            action_type=ActionType.CHECK_DUPLICATE,
            invoice_id=inv_id,
        ))
        assert "No duplicates" in result.observation.message


# ---------------------------------------------------------------------------
# Reward Tests
# ---------------------------------------------------------------------------

class TestRewards:
    def test_redundant_inspection_penalized(self, env):
        obs = env.reset("easy")
        inv_id = obs.invoices[0].invoice_id
        # First inspection
        r1 = env.step(Action(action_type=ActionType.INSPECT_INVOICE, invoice_id=inv_id))
        # Second inspection (redundant)
        r2 = env.step(Action(action_type=ActionType.INSPECT_INVOICE, invoice_id=inv_id))
        assert r2.reward.value < r1.reward.value

    def test_correct_flag_rewarded(self, env):
        obs = env.reset("easy")
        inv_id = obs.invoices[0].invoice_id
        result = env.step(Action(
            action_type=ActionType.FLAG_ERROR,
            invoice_id=inv_id,
            error_category=ErrorCategory.MATH_ERROR,
            error_description="Line item calculation error",
            severity=Severity.MEDIUM,
        ))
        assert result.reward.value > 0

    def test_uninformed_decision_penalized(self, env):
        obs = env.reset("easy")
        inv_id = obs.invoices[0].invoice_id
        # Approve without inspecting first
        result = env.step(Action(
            action_type=ActionType.APPROVE,
            invoice_id=inv_id,
            reason="looks fine",
        ))
        # Should have uninformed penalty
        assert "uninformed_decision_penalty" in result.reward.breakdown

    def test_premature_submit_penalized(self, env):
        obs = env.reset("medium")
        # Submit without doing anything
        result = env.step(Action(action_type=ActionType.SUBMIT_REVIEW))
        assert result.reward.value < 0


# ---------------------------------------------------------------------------
# Grader Tests
# ---------------------------------------------------------------------------

class TestGrader:
    def test_perfect_easy(self):
        grader = InvoiceReviewGrader("easy")
        score, breakdown = grader.grade(
            ground_truth_errors={"INV-001": [{"category": "math_error"}]},
            ground_truth_dispositions={"INV-001": "rejected"},
            flagged_errors={"INV-001": [{"category": "math_error"}]},
            review_status={"INV-001": "rejected"},
            checks_performed={"INV-001": ["inspect_invoice", "check_vendor", "check_po_match"]},
            steps_used=8,
            max_steps=15,
            num_invoices=1,
        )
        assert score > 0.8

    def test_zero_effort(self):
        grader = InvoiceReviewGrader("easy")
        score, breakdown = grader.grade(
            ground_truth_errors={"INV-001": [{"category": "math_error"}]},
            ground_truth_dispositions={"INV-001": "rejected"},
            flagged_errors={},
            review_status={},
            checks_performed={},
            steps_used=1,
            max_steps=15,
            num_invoices=1,
        )
        assert score < 0.3

    def test_grader_determinism(self):
        grader = InvoiceReviewGrader("medium")
        args = dict(
            ground_truth_errors={"A": [{"category": "math_error"}], "B": []},
            ground_truth_dispositions={"A": "rejected", "B": "approved"},
            flagged_errors={"A": [{"category": "math_error"}]},
            review_status={"A": "rejected", "B": "approved"},
            checks_performed={"A": ["inspect_invoice"], "B": ["inspect_invoice"]},
            steps_used=15,
            max_steps=30,
            num_invoices=2,
        )
        s1, _ = grader.grade(**args)
        s2, _ = grader.grade(**args)
        assert s1 == s2  # Deterministic

    def test_score_in_range(self):
        for diff in ["easy", "medium", "hard"]:
            grader = InvoiceReviewGrader(diff)
            score, _ = grader.grade(
                ground_truth_errors={"A": [{"category": "math_error"}]},
                ground_truth_dispositions={"A": "rejected"},
                flagged_errors={"A": [{"category": "math_error"}]},
                review_status={"A": "rejected"},
                checks_performed={"A": ["inspect_invoice"]},
                steps_used=5,
                max_steps=15,
                num_invoices=1,
            )
            assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# State Tests
# ---------------------------------------------------------------------------

class TestState:
    def test_state_reflects_actions(self, env):
        obs = env.reset("easy")
        inv_id = obs.invoices[0].invoice_id
        env.step(Action(action_type=ActionType.INSPECT_INVOICE, invoice_id=inv_id))
        state = env.state()
        assert state.step_number == 1
        assert "inspect_invoice" in state.checks_performed.get(inv_id, [])
        assert len(state.action_history) == 1

    def test_state_has_ground_truth(self, env):
        env.reset("easy")
        state = env.state()
        assert state.ground_truth_errors
        assert state.ground_truth_dispositions


# ---------------------------------------------------------------------------
# Full Episode Tests
# ---------------------------------------------------------------------------

class TestFullEpisode:
    def test_easy_optimal_run(self, env):
        """Run optimal sequence on easy task and verify high score."""
        obs = env.reset("easy")
        inv_id = "INV-2025-0001"

        # Inspect
        env.step(Action(action_type=ActionType.INSPECT_INVOICE, invoice_id=inv_id))
        # Check vendor
        env.step(Action(action_type=ActionType.CHECK_VENDOR, invoice_id=inv_id))
        # Check PO
        env.step(Action(action_type=ActionType.CHECK_PO_MATCH, invoice_id=inv_id))
        # Flag math errors
        env.step(Action(
            action_type=ActionType.FLAG_ERROR,
            invoice_id=inv_id,
            error_category=ErrorCategory.MATH_ERROR,
            error_description="Desk Organizer overcharged",
            severity=Severity.MEDIUM,
        ))
        # Reject
        env.step(Action(
            action_type=ActionType.REJECT,
            invoice_id=inv_id,
            reason="Math errors in line items",
        ))
        # Submit
        result = env.step(Action(action_type=ActionType.SUBMIT_REVIEW))

        assert result.done
        assert result.info["final_score"] > 0.6

    def test_hard_episode_completes(self, env):
        """Verify hard task can be completed without errors."""
        obs = env.reset("hard")
        assert len(obs.invoices) == 5

        for inv in obs.invoices:
            env.step(Action(
                action_type=ActionType.INSPECT_INVOICE,
                invoice_id=inv.invoice_id,
            ))
            env.step(Action(
                action_type=ActionType.APPROVE,
                invoice_id=inv.invoice_id,
                reason="Approving",
            ))

        result = env.step(Action(action_type=ActionType.SUBMIT_REVIEW))
        assert result.done
        assert 0.0 <= result.info["final_score"] <= 1.0
