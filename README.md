---
title: Invoice Review OpenEnv
emoji: 📑
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
---
# Invoice Review OpenEnv

A production-quality OpenEnv benchmark environment that simulates **real-world invoice and procurement review** workflows for training and evaluating AI agents.

## Overview

Accounts payable teams process thousands of invoices daily, checking for math errors, vendor issues, purchase order mismatches, policy violations, duplicates, and unauthorized charges. This environment encapsulates that workflow as a structured multi-step decision problem suitable for training tool-using AI agents.

### Why Invoice Review?

| Criterion | Strength |
|-----------|----------|
| **Real-world utility** | Every company with procurement needs AP review — this is a genuine $B workflow |
| **Deterministic grading** | Math errors, PO amounts, vendor status are all objectively verifiable |
| **Rich reward shaping** | Many intermediate investigation actions provide natural dense signal |
| **Difficulty scaling** | From single-invoice arithmetic to complex multi-invoice audit with subtle errors |
| **Practical agent value** | An agent that masters this could genuinely automate invoice processing |

## Quick Start

### Local Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Start the environment server
python -m uvicorn src.server:app --host 0.0.0.0 --port 7860

# In another terminal, run the baseline
python inference.py
```

### Docker

```bash
# Build
docker build -t invoice-review-openenv .

# Run
docker run -p 7860:7860 invoice-review-openenv

# Test health
curl http://localhost:7860/health
```

### Run Tests

```bash
pytest tests/ -v
```

---

## Environment API

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check / welcome |
| `GET` | `/health` | Container health probe |
| `GET` | `/tasks` | List available tasks |
| `POST` | `/reset` | Reset environment with `{"task_id": "easy"}` |
| `POST` | `/step` | Execute action, returns observation + reward |
| `GET` | `/state` | Full internal state (debugging) |

### OpenEnv Interface

```python
reset(task_id) -> Observation        # Start new episode
step(action)   -> StepResult         # (observation, reward, done, info)
state()        -> EnvState           # Full internal state
```

---

## Action Space

| Action | Parameters | Description |
|--------|-----------|-------------|
| `inspect_invoice` | `invoice_id` | View invoice details (line items, amounts, dates) |
| `check_vendor` | `invoice_id` | Look up vendor status, risk rating, contract info |
| `check_po_match` | `invoice_id` | Cross-reference invoice against purchase order |
| `check_duplicate` | `invoice_id` | Scan for duplicate invoices |
| `flag_error` | `invoice_id`, `error_category`, `error_description`, `severity` | Report a found error |
| `approve` | `invoice_id`, `reason` | Approve the invoice |
| `reject` | `invoice_id`, `reason` | Reject the invoice |
| `escalate` | `invoice_id`, `reason` | Escalate for dual approval |
| `submit_review` | — | Finalize the review (ends episode) |

### Error Categories

`math_error` · `missing_field` · `po_mismatch` · `duplicate` · `vendor_issue` · `policy_violation` · `overcharge` · `unauthorized`

### Severity Levels

`low` · `medium` · `high` · `critical`

---

## Observation Space

Each observation includes:

- **Invoices** — Full invoice data with line items, amounts, dates, vendor info
- **Purchase Orders** — PO reference data for cross-checking
- **Vendors** — Vendor master records with status and risk ratings
- **Company Policy** — Auto-approve limits, PO thresholds, blocked vendors
- **Review Status** — Current disposition of each invoice
- **Flagged Errors** — Errors the agent has reported so far
- **Checks Performed** — Which verification steps have been done
- **Step Counter** — Current step / max steps

---

## Tasks

### Easy — Single Invoice Review
- **Invoices:** 1
- **Max Steps:** 15
- **Challenge:** Obvious math errors in line item pricing
- **Expected Score:** 0.7–0.9 for competent agents

### Medium — Multi-Invoice Batch Review
- **Invoices:** 3
- **Max Steps:** 30
- **Challenges:** Suspended vendor, PO amount exceeded, overcharge, missing PO
- **Expected Score:** 0.5–0.8 for competent agents

### Hard — Complex Procurement Audit
- **Invoices:** 5
- **Max Steps:** 50
- **Challenges:** Blacklisted vendor, duplicate detection, subtle $50 overcharge, dual-approval threshold, unauthorized line items
- **Expected Score:** 0.3–0.7 for competent agents

---

## Reward Design

The environment provides **dense per-step rewards** in `[-1.0, 1.0]`:

### Positive Signals
- ✅ First inspection of an invoice: `+0.05`
- ✅ Relevant verification check (e.g., vendor check when vendor is suspicious): `+0.10`
- ✅ Correctly flagging a true error: `+0.15`
- ✅ Correct disposition (approve/reject/escalate): `+0.20`
- ✅ Making informed decisions (checked before deciding): `+0.05`
- ✅ Complete submission (all invoices dispositioned): `+0.10`

### Negative Signals
- ❌ Invalid/malformed action: `-0.10` (escalating)
- ❌ Redundant action (re-inspecting same invoice): `-0.03`
- ❌ False positive error flag: `-0.08`
- ❌ Wrong approval (approving bad invoice): `-0.20`
- ❌ Uninformed decision (no inspection first): `-0.08`
- ❌ Premature submission (invoices still pending): `-0.15`

---

## Grading

Final scores are computed deterministically in `[0.0, 1.0]` with four weighted components:

| Component | Easy | Medium | Hard | Description |
|-----------|------|--------|------|-------------|
| Error Detection | 40% | 30% | 25% | F1 of flagged vs ground-truth error categories |
| Disposition | 35% | 30% | 30% | Correct approve/reject/escalate decisions |
| Due Diligence | 15% | 25% | 25% | Completeness of verification checks |
| Efficiency | 10% | 15% | 20% | Step budget utilization |

---

## Baseline Inference

```bash
# Set environment variables
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export HF_TOKEN=hf_your_token
export ENV_BASE_URL=http://localhost:7860

# Run baseline
python inference.py
```

### Structured Output Format

```
[START] task_id=easy num_invoices=1 max_steps=15
[STEP] task_id=easy step=1 action=inspect_invoice invoice=INV-2025-0001 reward=0.0500 cumulative_reward=0.0500 done=False
[STEP] task_id=easy step=2 action=check_vendor invoice=INV-2025-0001 reward=0.0400 cumulative_reward=0.0900 done=False
...
[END] task_id=easy steps=6 final_score=0.8234 cumulative_reward=0.5100 elapsed_seconds=12.3
```

### Expected Baseline Scores

| Task | Score Range | Steps |
|------|------------|-------|
| Easy | 0.60–0.85 | 5–10 |
| Medium | 0.45–0.70 | 12–25 |
| Hard | 0.30–0.60 | 20–40 |

---

## Hugging Face Space Deployment

1. Create a new **Docker** Space on Hugging Face
2. Push this repository to the Space
3. The Dockerfile is pre-configured for port 7860
4. The Space will automatically build and serve the environment

```bash
# Example: push to HF Space
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/invoice-review-openenv
git push space main
```

---

## Project Structure

```
├── README.md                 # This file
├── openenv.yaml              # OpenEnv metadata and configuration
├── inference.py              # Baseline inference script
├── Dockerfile                # Container for deployment
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variable template
├── validate.py               # Pre-submission validation script
├── src/
│   ├── __init__.py           # Package exports
│   ├── models.py             # Pydantic typed models (Observation, Action, Reward, etc.)
│   ├── environment.py        # Core environment engine (reset/step/state)
│   ├── server.py             # FastAPI HTTP server
│   ├── data/
│   │   ├── __init__.py
│   │   └── generator.py      # Deterministic invoice data generation
│   ├── tasks/
│   │   ├── __init__.py
│   │   └── registry.py       # Task definitions and registry
│   ├── graders/
│   │   ├── __init__.py
│   │   └── grader.py         # Deterministic scoring logic
│   └── rewards/
│       ├── __init__.py
│       └── reward.py         # Dense reward shaping
└── tests/
    ├── __init__.py
    └── test_environment.py   # Comprehensive test suite
```

---

## Validation

Run pre-submission validation:

```bash
python validate.py
```

This checks:
- ✅ Server starts and responds to health check
- ✅ All 3 tasks are listed
- ✅ Reset works for all tasks
- ✅ Step works with valid actions
- ✅ Grader produces scores in [0.0, 1.0]
- ✅ Full episode completes successfully
- ✅ openenv.yaml exists and is valid

---

## License

MIT
