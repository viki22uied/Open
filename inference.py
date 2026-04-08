"""
Baseline inference script for the Invoice Review OpenEnv environment.

Uses the OpenAI Python client (compatible with HF router and any
OpenAI-compatible endpoint) to run a baseline agent across all tasks.

Usage:
    python inference.py

Environment variables:
    API_BASE_URL  — LLM API endpoint (default: https://router.huggingface.co/v1)
    MODEL_NAME    — model identifier (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN      — Hugging Face API token
    OPENAI_API_KEY— fallback API key
    ENV_BASE_URL  — environment server URL (default: http://localhost:7860)
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN", "")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

TASKS = ["easy", "medium", "hard"]
MAX_RETRIES = 3

# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# ---------------------------------------------------------------------------
# Environment client
# ---------------------------------------------------------------------------


class EnvClient:
    """HTTP client for the Invoice Review OpenEnv server."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def health(self) -> bool:
        try:
            r = self.session.get(f"{self.base_url}/health", timeout=10)
            return r.status_code == 200
        except Exception:
            return False

    def reset(self, task_id: str) -> Dict[str, Any]:
        r = self.session.post(
            f"{self.base_url}/reset",
            json={"task_id": task_id},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        r = self.session.post(
            f"{self.base_url}/step",
            json=action,
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    def get_tasks(self) -> List[Dict[str, Any]]:
        r = self.session.get(f"{self.base_url}/tasks", timeout=10)
        r.raise_for_status()
        return r.json()["tasks"]


# ---------------------------------------------------------------------------
# System prompt for the baseline agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert invoice review agent. You review invoices for a company's procurement department.

Your job is to:
1. Inspect each invoice carefully
2. Check vendor status
3. Verify PO matching
4. Check for duplicates
5. Flag any errors found (math errors, overcharges, policy violations, etc.)
6. Make a disposition decision (approve, reject, or escalate) for each invoice
7. Submit your final review

Available actions (respond with JSON):
- {"action_type": "inspect_invoice", "invoice_id": "<id>"}
- {"action_type": "check_vendor", "invoice_id": "<id>"}
- {"action_type": "check_po_match", "invoice_id": "<id>"}
- {"action_type": "check_duplicate", "invoice_id": "<id>"}
- {"action_type": "flag_error", "invoice_id": "<id>", "error_category": "<category>", "error_description": "<desc>", "severity": "<sev>"}
- {"action_type": "approve", "invoice_id": "<id>", "reason": "<reason>"}
- {"action_type": "reject", "invoice_id": "<id>", "reason": "<reason>"}
- {"action_type": "escalate", "invoice_id": "<id>", "reason": "<reason>"}
- {"action_type": "submit_review"}

Error categories: math_error, missing_field, po_mismatch, duplicate, vendor_issue, policy_violation, overcharge, unauthorized
Severity levels: low, medium, high, critical

IMPORTANT RULES:
- Always inspect an invoice before making decisions about it
- Check vendor status and PO match for every invoice
- Flag ALL errors you find before making a disposition
- Reject invoices with critical errors (vendor issues, large overcharges)
- Escalate invoices that need dual approval (>$10,000) or have policy violations
- Approve clean invoices
- Submit review only after all invoices are dispositioned

Respond with ONLY a valid JSON action object, nothing else."""


# ---------------------------------------------------------------------------
# Agent logic
# ---------------------------------------------------------------------------


def build_user_message(observation: Dict[str, Any], step_num: int) -> str:
    """Build a user message from the observation for the LLM."""
    obs = observation
    invoices = obs.get("invoices", [])
    invoice_ids = [inv["invoice_id"] for inv in invoices]
    review_status = obs.get("review_status", {})
    flagged = obs.get("flagged_errors", {})
    checks = obs.get("checks_performed", {})

    msg_parts = [
        f"Step {obs['step_number']}/{obs['max_steps']}.",
        f"Message: {obs.get('message', '')}",
        f"Invoice IDs: {invoice_ids}",
        f"Review status: {json.dumps(review_status)}",
        f"Checks performed: {json.dumps(checks)}",
        f"Errors flagged: {json.dumps({k: len(v) for k, v in flagged.items()})}",
    ]

    # Include policy info
    policy = obs.get("policy", {})
    if policy:
        msg_parts.append(
            f"Policy: auto-approve below ${policy.get('max_auto_approve', 5000)}, "
            f"PO required above ${policy.get('require_po_above', 1000)}, "
            f"dual approval above ${policy.get('require_dual_approval_above', 10000)}, "
            f"blocked vendors: {policy.get('blocked_vendors', [])}"
        )

    return "\n".join(msg_parts)


def parse_action(response_text: str) -> Optional[Dict[str, Any]]:
    """Parse the LLM response into an action dict."""
    text = response_text.strip()

    # Try to extract JSON from the response
    # Handle markdown code blocks
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    # Try direct JSON parse
    try:
        action = json.loads(text)
        if isinstance(action, dict) and "action_type" in action:
            return action
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in the text
    for i, ch in enumerate(text):
        if ch == "{":
            depth = 0
            for j in range(i, len(text)):
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                if depth == 0:
                    try:
                        action = json.loads(text[i : j + 1])
                        if isinstance(action, dict) and "action_type" in action:
                            return action
                    except json.JSONDecodeError:
                        break
            break

    return None


def get_llm_action(
    messages: List[Dict[str, str]],
    retry: int = 0,
) -> Dict[str, Any]:
    """Call the LLM and parse the response into an action."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.1,
            max_tokens=512,
        )
        text = response.choices[0].message.content or ""
        action = parse_action(text)
        if action:
            return action

        # Failed to parse — retry
        if retry < MAX_RETRIES:
            messages.append({"role": "assistant", "content": text})
            messages.append({
                "role": "user",
                "content": "Your response was not valid JSON. Please respond with ONLY a JSON action object.",
            })
            return get_llm_action(messages, retry + 1)

        # Fallback: submit review
        return {"action_type": "submit_review"}

    except Exception as e:
        print(f"  LLM error: {e}", file=sys.stderr)
        if retry < MAX_RETRIES:
            time.sleep(2 ** retry)
            return get_llm_action(messages, retry + 1)
        return {"action_type": "submit_review"}


def run_task(env_client: EnvClient, task_id: str) -> Dict[str, Any]:
    """Run a single task and return results."""
    start_time = time.time()

    # Reset
    reset_data = env_client.reset(task_id)
    observation = reset_data["observation"]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(observation, 0)},
    ]

    total_reward = 0.0
    steps = 0
    done = False
    final_score = 0.0
    info = {}
    rewards_list = []

    while not done:
        # Get action from LLM
        action = get_llm_action(messages)
        steps += 1

        # Execute action
        step_result = env_client.step(action)
        observation = step_result["observation"]
        reward = step_result["reward"]
        done = step_result["done"]
        info = step_result.get("info", {})

        reward_value = float(reward["value"])
        rewards_list.append(reward_value)
        total_reward += reward_value
        
        error_val = info.get("error", "null")
        done_val = str(done).lower()

        # Update conversation with the result
        result_msg = reward.get("message", observation.get("message", ""))
        messages.append({"role": "assistant", "content": json.dumps(action)})
        messages.append({"role": "user", "content": build_user_message(observation, steps)})

        if done:
            final_score = info.get("final_score", 0.0)
            break

    elapsed = time.time() - start_time
    success_bool = str(final_score >= 0.5).lower()
    rewards_str = ",".join(f"{r:.2f}" for r in rewards_list)

    return {
        "task_id": task_id,
        "steps": steps,
        "final_score": final_score,
        "cumulative_reward": total_reward,
        "elapsed_seconds": elapsed,
        "info": info,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """Run baseline inference on all tasks."""
    print("START")
    print("STEP: Initializing model")
    print("=" * 60)
    print("Invoice Review OpenEnv — Baseline Inference")
    print("=" * 60)
    print(f"API: {API_BASE_URL}")
    print(f"Model: {MODEL_NAME}")
    print(f"Env: {ENV_BASE_URL}")
    print("=" * 60)

    env_client = EnvClient(ENV_BASE_URL)

    # Health check
    if not env_client.health():
        print("ERROR: Environment server not reachable at", ENV_BASE_URL, file=sys.stderr)
        print("Start the server first: python -m uvicorn server.app:app --host 0.0.0.0 --port 7860", file=sys.stderr)
        sys.exit(1)

    results = []
    print("STEP: Running inference")
    for task_id in TASKS:
        print(f"\n{'—' * 40}")
        print(f"Running task: {task_id}")
        print(f"{'—' * 40}")
        try:
            result = run_task(env_client, task_id)
            results.append(result)
        except Exception as e:
            print(f"ERROR on task {task_id}: {e}", file=sys.stderr)
            traceback.print_exc()
            results.append({
                "task_id": task_id,
                "steps": 0,
                "final_score": 0.0,
                "cumulative_reward": 0.0,
                "elapsed_seconds": 0.0,
                "error": str(e),
            })

    print("STEP: Returning output")
    # Summary
    print(f"\n{'=' * 60}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 60}")
    for r in results:
        print(
            f"  {r['task_id']:8s} | "
            f"score={r['final_score']:.4f} | "
            f"reward={r['cumulative_reward']:.4f} | "
            f"steps={r['steps']:3d} | "
            f"time={r.get('elapsed_seconds', 0):.1f}s"
        )

    avg_score = sum(r["final_score"] for r in results) / len(results) if results else 0
    print(f"\n  Average score: {avg_score:.4f}")
    print(f"{'=' * 60}")
    print("END")


if __name__ == "__main__":
    main()
