---
title: Invoice Validation Env
emoji: 📄
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Invoice Validation Environment

An **OpenEnv-compatible reinforcement learning environment** that puts an AI agent in the role of a finance team reviewer. The agent receives invoice data, detects fraud signals and arithmetic errors, and decides whether to approve, reject, flag, or hold the invoice for more information.

---

## Why Invoice Validation?

Finance teams process thousands of invoices monthly. Manual review is slow, expensive, and error-prone. An agent that can reliably catch:

- **Missing legal identifiers** (GST numbers, vendor names)
- **Arithmetic fraud** (inflated totals that don't match line items)
- **Duplicate submissions** (same invoice re-submitted hoping to get paid twice)
- **Missing documentation** (no payment receipts)

…can save companies significant money and catch fraud before it happens.

This environment gives researchers a realistic, deterministic, and scoreable benchmark for training and evaluating such agents.

---

## Project Structure

```
Invoice_Validator/
│
├── app/
│   ├── __init__.py       # package marker
│   ├── env.py            # InvoiceValidationEnv — the core state machine
│   ├── models.py         # Pydantic models (State, Action, Observation, Reward)
│   ├── grader.py         # deterministic reward & scoring logic
│   ├── tasks.py          # three task definitions (easy / medium / hard)
│   ├── validator.py      # pure business-logic validation rules
│   └── main.py           # FastAPI server wrapping the environment
│
├── inference.py          # LLM agent loop (run this to evaluate a model)
├── Dockerfile            # multi-stage Python 3.11-slim image
├── requirements.txt
├── openenv.yaml          # OpenEnv specification
├── .env.example          # environment variable template
└── scripts/
    └── validate-submission.sh   # smoke-test against a live server
```

---

## Environment Design

### State (`InvoiceState`)

The full internal state of one review episode. The agent never sees this directly.

| Field | Type | Description |
|---|---|---|
| `invoice_id` | str | Unique invoice reference |
| `vendor_name` | Optional[str] | Name of the supplier (can be None/blank) |
| `subtotal` | float | Pre-tax subtotal |
| `tax` | float | Tax amount |
| `total` | float | Total as listed on invoice |
| `gst_number` | Optional[str] | GST registration number |
| `invoice_date` | Optional[str] | Date on the invoice |
| `duplicate_invoice` | bool | Whether this was already submitted |
| `receipt_attached` | bool | Whether a payment receipt is attached |
| `approval_status` | ApprovalStatus | pending / approved / rejected / flagged / on_hold |
| `detected_issues` | List[str] | Issues the agent has called `add_issue` for |
| `steps_taken` | int | Number of steps consumed so far |
| `done` | bool | Whether the episode is over |
| `expected_issues` | List[str] | Ground truth (used by grader, hidden from agent) |
| `expected_action` | Optional[str] | Ground truth final action (hidden from agent) |
| `task_difficulty` | str | easy / medium / hard |

### Action Space (`InvoiceAction`)

```json
{"action_type": "add_issue", "value": "missing_gst_number"}
{"action_type": "mark_valid"}
{"action_type": "mark_invalid"}
{"action_type": "flag_duplicate"}
{"action_type": "request_missing_info"}
{"action_type": "finalize_review"}
```

| Action | Effect |
|---|---|
| `add_issue` | Records an issue string into `detected_issues` |
| `mark_valid` | Sets status → approved, closes episode |
| `mark_invalid` | Sets status → rejected, closes episode |
| `flag_duplicate` | Sets status → flagged (episode continues) |
| `request_missing_info` | Sets status → on_hold, closes episode |
| `finalize_review` | Closes the episode with current status |

### Observation Space (`InvoiceObservation`)

What the agent sees after every step. Derived from `InvoiceState` but with ground-truth fields stripped out.

```json
{
  "invoice_id": "INV-2024-0042",
  "vendor_name": "Acme Corp",
  "subtotal": 5000.0,
  "tax": 900.0,
  "total": 5900.0,
  "gst_number": null,
  "invoice_date": "2024-03-15",
  "duplicate_flag": false,
  "receipt_attached": true,
  "approval_status": "pending",
  "detected_issues": [],
  "last_action_result": "Episode started. Begin review.",
  "remaining_steps": 20,
  "done": false,
  "task_description": "[EASY TASK] Review invoice INV-2024-0042 ..."
}
```

---

## Reward Function

Rewards are shaped to encourage **complete, precise detection** followed by an **appropriate final decision**.

| Event | Reward |
|---|---|
| Detect `missing_gst_number` | +0.20 |
| Detect `wrong_total_calculation` | +0.20 |
| Detect `duplicate_invoice` | +0.20 |
| Detect `missing_receipt` | +0.10 |
| Detect `missing_vendor_name` | +0.10 |
| Correct final action | +0.20 |
| Wrong final action | −0.50 |
| Redundant / false-positive action | −0.05 |
| Too many steps (beyond grace period) | −0.10 |
| **Total** | **clamped to [0.0, 1.0]** |

The reward is **split**: small instantaneous rewards come back on each `add_issue` step so the agent gets a learning signal during the episode; the big terminal reward fires when `finalize_review` is called.

---

## Tasks

### Easy — single missing field

```
Invoice: INV-2024-0042 | Vendor: Acme Corp
Subtotal: 5000 | Tax: 900 | Total: 5900 ✓
GST Number: MISSING
```

**Expected agent behaviour:**
1. `add_issue` → `missing_gst_number`
2. `request_missing_info`
3. `finalize_review`

**Target score:** ≥ 0.80

---

### Medium — arithmetic error

```
Invoice: INV-2024-0117 | Vendor: BuildRight Supplies
Subtotal: 2000 | Tax: 360 | Total: 2200 ✗ (should be 2360)
```

**Expected agent behaviour:**
1. `add_issue` → `wrong_total_calculation`
2. `mark_invalid`
3. `finalize_review`

**Target score:** ≥ 0.75

---

### Hard — five overlapping issues

```
Invoice: INV-2024-0088 | Vendor: MISSING
Subtotal: 8000 | Tax: 1440 | Total: 9800 ✗ (should be 9440)
GST Number: MISSING | Duplicate: YES | Receipt: MISSING
```

**Expected agent behaviour:**
1. `add_issue` × 5 (all issues)
2. `flag_duplicate`
3. `request_missing_info` or `mark_invalid`
4. `finalize_review`

**Target score:** ≥ 0.70

---

## Baseline Scores

Tested with `llama-3.3-70b-versatile` via Groq:

| Task | Score |
|---|---|
| Easy | 0.80 |
| Medium | 0.60 |
| Hard | 0.50 |

Scores will vary by model and prompt tuning.

---

## Running Locally

### Prerequisites

- Python 3.11+
- A free [Groq API key](https://console.groq.com) (or any OpenAI-compatible provider)

### Step 1 — Clone and install

```bash
# if you haven't cloned yet
git clone https://github.com/your-username/invoice-validation-env.git
cd invoice-validation-env

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2 — Configure environment variables

```bash
cp .env.example .env
# open .env and fill in OPENAI_API_KEY with your Groq key
```

### Step 3 — Start the server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 7860 --reload
```

The API docs will be live at: http://localhost:7860/docs

### Step 4 — Run the inference script

```bash
# Easy task (no server needed — runs in-process)
python inference.py --task easy

# Medium task against a running server
python inference.py --task medium --env-url http://localhost:7860

# Hard task
python inference.py --task hard
```

### Step 5 — Smoke-test the server

```bash
bash scripts/validate-submission.sh http://localhost:7860
```

---

## Manual API Testing (curl)

```bash
# Reset to easy task
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"difficulty": "easy"}'

# Add an issue
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "add_issue", "value": "missing_gst_number"}'

# Request missing info and close the episode
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "request_missing_info"}'

curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "finalize_review"}'

# Peek at the full state (debug only)
curl http://localhost:7860/state
```

---

## Building the Docker Image

```bash
docker build -t invoice-validation-env .

# Run the container
docker run --rm \
  -p 7860:7860 \
  --env-file .env \
  invoice-validation-env
```

The server will be available at http://localhost:7860.

To run inference against the containerised server:

```bash
python inference.py --task hard --env-url http://localhost:7860
```

---

## Deploying to Hugging Face Spaces

This project is designed as a **Docker Space** on Hugging Face.

### Step 1 — Create the Space

1. Go to https://huggingface.co/spaces
2. Click **New Space**
3. Choose **Docker** as the SDK
4. Give it a name, e.g. `invoice-validation-env`
5. Set visibility to **Public** (or Private if you prefer)

### Step 2 — Add secrets

In your Space settings → **Secrets**, add:

| Secret name | Value |
|---|---|
| `OPENAI_API_KEY` | Your Groq or HF API key |
| `MODEL_NAME` | `llama-3.3-70b-versatile` |
| `API_BASE_URL` | `https://api.groq.com/openai/v1` |
| `HF_TOKEN` | Your Hugging Face token |

### Step 3 — Push the code

```bash
# Add the HF remote
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/invoice-validation-env

# Push — HF will build and deploy automatically
git push hf main
```

### Step 4 — Verify deployment

Once the build finishes (usually 2–4 minutes), your Space will be live at:

```
https://YOUR_USERNAME-invoice-validation-env.hf.space
```

Run the smoke-test against it:

```bash
bash scripts/validate-submission.sh \
  https://YOUR_USERNAME-invoice-validation-env.hf.space
```

Run inference against the deployed server:

```bash
API_BASE_URL=https://api.groq.com/openai/v1 \
MODEL_NAME=llama-3.3-70b-versatile \
OPENAI_API_KEY=your_key \
python inference.py --task hard \
  --env-url https://YOUR_USERNAME-invoice-validation-env.hf.space
```

---

## Example Output

```
[START] task=hard env=in-process model=llama-3.3-70b-versatile
[STEP] step=1 action=add_issue value=missing_vendor_name reward=0.1000 done=False
[STEP] step=2 action=add_issue value=missing_gst_number reward=0.2000 done=False
[STEP] step=3 action=add_issue value=wrong_total_calculation reward=0.2000 done=False
[STEP] step=4 action=add_issue value=duplicate_invoice reward=0.2000 done=False
[STEP] step=5 action=add_issue value=missing_receipt reward=0.1000 done=False
[STEP] step=6 action=flag_duplicate value=N/A reward=0.0000 done=False
[STEP] step=7 action=mark_invalid value=N/A reward=0.0000 done=False
[STEP] step=8 action=finalize_review value=N/A reward=0.7000 done=True
[END] success=True steps=8 score=0.8
```

---

## Environment Variables Reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `API_BASE_URL` | Yes | — | LLM provider base URL |
| `MODEL_NAME` | Yes | — | Model identifier |
| `OPENAI_API_KEY` | Yes | — | API key for the provider |
| `HF_TOKEN` | No | — | Hugging Face token |

---

## Free Model Providers

### Groq (recommended — fastest)

```
API_BASE_URL=https://api.groq.com/openai/v1
MODEL_NAME=llama-3.3-70b-versatile
```

Get a free key at: https://console.groq.com

### Hugging Face Inference API

```
API_BASE_URL=https://api-inference.huggingface.co/v1
MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.3
OPENAI_API_KEY=hf_your_token_here
```

---

## Resource Requirements

| Resource | Limit |
|---|---|
| vCPU | 2 |
| Memory | 8 GB |
| Max episode duration | < 20 minutes |
| Docker image size | ~250 MB |

---

## License

MIT — free to use, fork, and build on.
