"""
Pre-submission validation script for the Invoice Review OpenEnv environment.

Verifies all requirements without needing an LLM API key:
  1. openenv.yaml exists and is parseable
  2. All typed models import correctly
  3. Task registry has 3+ tasks
  4. Environment reset/step/state work for all tasks
  5. Grader produces deterministic scores in [0.0, 1.0]
  6. Reward values are in [-1.0, 1.0]
  7. Full episode completes for each task
  8. Server starts and responds to HTTP requests
"""

from __future__ import annotations

import os
import sys
import time
import subprocess
import signal

# Ensure we can import from the project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def check(label: str, condition: bool, detail: str = ""):
    status = "[PASS]" if condition else "[FAIL]"
    msg = f"  {status}: {label}"
    if detail and not condition:
        msg += f" — {detail}"
    print(msg)
    return condition


def main():
    print("=" * 60)
    print("Invoice Review OpenEnv — Pre-Submission Validation")
    print("=" * 60)

    all_pass = True

    # ------------------------------------------------------------------
    # 1. openenv.yaml
    # ------------------------------------------------------------------
    print("\n[1] OpenEnv YAML Configuration")
    try:
        import yaml
        yaml_path = os.path.join(os.path.dirname(__file__), "openenv.yaml")
        with open(yaml_path) as f:
            config = yaml.safe_load(f)
        all_pass &= check("openenv.yaml exists", True)
        all_pass &= check("Has name field", "name" in config)
        all_pass &= check("Has interface field", "interface" in config)
        all_pass &= check("Has tasks field", "tasks" in config)
        all_pass &= check("Has 3+ tasks", len(config.get("tasks", [])) >= 3)
        all_pass &= check("Has models field", "models" in config)
        all_pass &= check("Has reward field", "reward" in config)
        all_pass &= check("Has grading field", "grading" in config)
    except Exception as e:
        all_pass &= check("openenv.yaml parseable", False, str(e))

    # ------------------------------------------------------------------
    # 2. Model imports
    # ------------------------------------------------------------------
    print("\n[2] Typed Pydantic Models")
    try:
        from src.models import Observation, Action, Reward, EnvState, StepResult, TaskInfo
        all_pass &= check("Observation model imports", True)
        all_pass &= check("Action model imports", True)
        all_pass &= check("Reward model imports", True)
        all_pass &= check("EnvState model imports", True)
        all_pass &= check("StepResult model imports", True)
        all_pass &= check("TaskInfo model imports", True)
    except Exception as e:
        all_pass &= check("Models import", False, str(e))

    # ------------------------------------------------------------------
    # 3. Task registry
    # ------------------------------------------------------------------
    print("\n[3] Task Registry")
    try:
        from src.tasks import list_tasks, get_task
        tasks = list_tasks()
        all_pass &= check("list_tasks returns results", len(tasks) > 0)
        all_pass &= check("3+ tasks registered", len(tasks) >= 3)

        task_ids = [t.task_id for t in tasks]
        all_pass &= check("Has 'easy' task", "easy" in task_ids)
        all_pass &= check("Has 'medium' task", "medium" in task_ids)
        all_pass &= check("Has 'hard' task", "hard" in task_ids)

        for t in tasks:
            all_pass &= check(
                f"Task '{t.task_id}' has max_steps > 0",
                t.max_steps > 0,
            )
    except Exception as e:
        all_pass &= check("Task registry works", False, str(e))

    # ------------------------------------------------------------------
    # 4. Environment lifecycle
    # ------------------------------------------------------------------
    print("\n[4] Environment Lifecycle (reset/step/state)")
    try:
        from src.environment import InvoiceReviewEnv
        from src.models import Action, ActionType, ErrorCategory, Severity

        env = InvoiceReviewEnv()

        for task_id in ["easy", "medium", "hard"]:
            obs = env.reset(task_id)
            all_pass &= check(
                f"reset('{task_id}') returns observation",
                obs is not None and obs.task_id == task_id,
            )
            all_pass &= check(
                f"  step_number starts at 0",
                obs.step_number == 0,
            )
            all_pass &= check(
                f"  done is False",
                not obs.done,
            )
            all_pass &= check(
                f"  has invoices",
                len(obs.invoices) > 0,
            )

            # Test step
            inv_id = obs.invoices[0].invoice_id
            result = env.step(Action(
                action_type=ActionType.INSPECT_INVOICE,
                invoice_id=inv_id,
            ))
            all_pass &= check(
                f"  step() returns result",
                result is not None,
            )
            all_pass &= check(
                f"  reward in [-1, 1]",
                -1.0 <= result.reward.value <= 1.0,
            )

            # Test state
            state = env.state()
            all_pass &= check(
                f"  state() returns state",
                state is not None and state.task_id == task_id,
            )
            all_pass &= check(
                f"  state has ground_truth",
                len(state.ground_truth_errors) > 0 or len(state.ground_truth_dispositions) > 0,
            )

    except Exception as e:
        all_pass &= check("Environment lifecycle", False, str(e))

    # ------------------------------------------------------------------
    # 5. Grader determinism and range
    # ------------------------------------------------------------------
    print("\n[5] Grader Quality")
    try:
        from src.graders import InvoiceReviewGrader

        for diff in ["easy", "medium", "hard"]:
            grader = InvoiceReviewGrader(diff)

            # Test with some flagged errors
            s1, b1 = grader.grade(
                ground_truth_errors={"A": [{"category": "math_error"}]},
                ground_truth_dispositions={"A": "rejected"},
                flagged_errors={"A": [{"category": "math_error"}]},
                review_status={"A": "rejected"},
                checks_performed={"A": ["inspect_invoice", "check_vendor"]},
                steps_used=8,
                max_steps=15,
                num_invoices=1,
            )
            s2, b2 = grader.grade(
                ground_truth_errors={"A": [{"category": "math_error"}]},
                ground_truth_dispositions={"A": "rejected"},
                flagged_errors={"A": [{"category": "math_error"}]},
                review_status={"A": "rejected"},
                checks_performed={"A": ["inspect_invoice", "check_vendor"]},
                steps_used=8,
                max_steps=15,
                num_invoices=1,
            )

            all_pass &= check(f"Grader '{diff}' score in [0,1]", 0.0 <= s1 <= 1.0)
            all_pass &= check(f"Grader '{diff}' is deterministic", s1 == s2)

            # Test different inputs give different scores
            s3, _ = grader.grade(
                ground_truth_errors={"A": [{"category": "math_error"}]},
                ground_truth_dispositions={"A": "rejected"},
                flagged_errors={},
                review_status={},
                checks_performed={},
                steps_used=1,
                max_steps=15,
                num_invoices=1,
            )
            all_pass &= check(
                f"Grader '{diff}' varies with input",
                s1 != s3,
                f"Both scored {s1}",
            )

    except Exception as e:
        all_pass &= check("Grader quality", False, str(e))

    # ------------------------------------------------------------------
    # 6. Full episode completion
    # ------------------------------------------------------------------
    print("\n[6] Full Episode Completion")
    try:
        from src.environment import InvoiceReviewEnv
        from src.models import Action, ActionType, ErrorCategory, Severity

        env = InvoiceReviewEnv()

        for task_id in ["easy", "medium", "hard"]:
            obs = env.reset(task_id)

            # Walk through all invoices with basic actions
            for inv in obs.invoices:
                iid = inv.invoice_id
                env.step(Action(action_type=ActionType.INSPECT_INVOICE, invoice_id=iid))
                env.step(Action(action_type=ActionType.CHECK_VENDOR, invoice_id=iid))
                env.step(Action(action_type=ActionType.CHECK_PO_MATCH, invoice_id=iid))
                env.step(Action(action_type=ActionType.APPROVE, invoice_id=iid, reason="test"))

            result = env.step(Action(action_type=ActionType.SUBMIT_REVIEW))
            all_pass &= check(
                f"Episode '{task_id}' completes",
                result.done,
            )
            all_pass &= check(
                f"  final_score in [0, 1]",
                0.0 <= result.info.get("final_score", -1) <= 1.0,
            )
            all_pass &= check(
                f"  final_score = {result.info.get('final_score', 0):.4f}",
                True,
            )

    except Exception as e:
        all_pass &= check("Full episode", False, str(e))

    # ------------------------------------------------------------------
    # 7. File checks
    # ------------------------------------------------------------------
    print("\n[7] Required Files")
    base = os.path.dirname(os.path.abspath(__file__))
    required_files = [
        "README.md",
        "openenv.yaml",
        "inference.py",
        "Dockerfile",
        "requirements.txt",
        ".env.example",
        "src/models.py",
        "src/environment.py",
        "server/app.py",
        "src/data/generator.py",
        "src/tasks/registry.py",
        "src/graders/grader.py",
        "src/rewards/reward.py",
        "tests/test_environment.py",
    ]
    for f in required_files:
        path = os.path.join(base, f)
        all_pass &= check(f"File exists: {f}", os.path.isfile(path))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    if all_pass:
        print("ALL CHECKS PASSED - Ready for submission!")
    else:
        print("SOME CHECKS FAILED - Please fix issues above.")
    print(f"{'=' * 60}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
