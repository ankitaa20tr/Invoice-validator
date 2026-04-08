"""
inference.py — OpenEnv-compatible inference script for Invoice Validation Environment.

Follows the exact [START] / [STEP] / [END] structured log format required by the
OpenEnv hackathon checker.

Environment variables
---------------------
  API_BASE_URL      — LLM provider base URL  (default: Groq endpoint)
  MODEL_NAME        — model identifier       (default: llama-3.3-70b-versatile)
  OPENAI_API_KEY    — API key (primary)
  HF_TOKEN          — Hugging Face API key   (used when OPENAI_API_KEY is not set)
  LOCAL_IMAGE_NAME  — Docker image name      (optional)
  ENV_URL           — Override env server URL (optional)

Usage
-----
  python inference.py
  python inference.py --task easy
  python inference.py --task hard --env-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# Config — all from environment variables with sensible defaults
# ---------------------------------------------------------------------------

API_BASE_URL     = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME       = os.getenv("MODEL_NAME",   "llama-3.3-70b-versatile")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
HF_TOKEN         = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# Resolve API key: prefer OPENAI_API_KEY, fall back to HF_TOKEN
API_KEY = OPENAI_API_KEY or HF_TOKEN or ""

ENV_URL_DEFAULT = os.getenv(
    "ENV_URL",
    "https://ankitaatr-invoice-validation-env.hf.space",
)

TASK_NAME      = "easy"
BENCHMARK      = "invoice-validation"
MAX_STEPS      = 20
MAX_TOTAL_REWARD = 1.0          # env returns normalised terminal reward in [0, 1]
SUCCESS_SCORE_THRESHOLD = 0.5  # episode considered successful above this
TEMPERATURE    = 0.0
MAX_TOKENS     = 256

# ---------------------------------------------------------------------------
# OpenAI-compatible client (used for all LLM calls per hackathon rules)
# ---------------------------------------------------------------------------

client = OpenAI(
    api_key=API_KEY if API_KEY else "not-set",
    base_url=API_BASE_URL,
)

USE_LLM = bool(API_KEY)

if not USE_LLM:
    print(
        "[INFO] Neither OPENAI_API_KEY nor HF_TOKEN is set — "
        "using deterministic rule-based agent.",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Structured log helpers  (must match [START] / [STEP] / [END] spec exactly)
# ---------------------------------------------------------------------------

def log_start(*, task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    *,
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str] = None,
) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:+.4f} "
        f"done={done} error={error}",
        flush=True,
    )


def log_end(
    *,
    success: bool,
    steps: int,
    score: float,
    rewards: List[float],
) -> None:
    print(
        f"[END] success={success} steps={steps} "
        f"score={score:.4f} rewards={rewards}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert finance auditor. Your job is to review an invoice and detect any problems.

You must respond with a single JSON object representing your next action. No extra text.

Valid actions:
  {"action_type": "add_issue", "value": "<issue_name>"}
  {"action_type": "mark_valid"}
  {"action_type": "mark_invalid"}
  {"action_type": "flag_duplicate"}
  {"action_type": "request_missing_info"}
  {"action_type": "finalize_review"}

Known issue names:
  missing_gst_number
  missing_vendor_name
  wrong_total_calculation
  duplicate_invoice
  missing_receipt

Rules:
- First use add_issue for every problem you find.
- Then set a final status (mark_valid / mark_invalid / flag_duplicate / request_missing_info).
- Then call finalize_review to end the episode.
- Do not repeat an action you have already taken.
- If total != subtotal + tax, that is wrong_total_calculation.
"""


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_user_prompt(obs: Dict[str, Any], step: int) -> str:
    issues_found  = obs.get("detected_issues", [])
    issues_str    = ", ".join(issues_found) if issues_found else "none yet"
    expected_total = obs["subtotal"] + obs["tax"]

    return (
        f"Step {step} — Invoice Review\n\n"
        f"Invoice ID    : {obs['invoice_id']}\n"
        f"Vendor        : {obs.get('vendor_name') or 'NOT PROVIDED'}\n"
        f"Subtotal      : {obs['subtotal']}\n"
        f"Tax           : {obs['tax']}\n"
        f"Listed Total  : {obs['total']}\n"
        f"Expected Total: {expected_total}\n"
        f"GST Number    : {obs.get('gst_number') or 'NOT PROVIDED'}\n"
        f"Invoice Date  : {obs.get('invoice_date') or 'NOT PROVIDED'}\n"
        f"Duplicate Flag: {obs['duplicate_flag']}\n"
        f"Receipt Att.  : {obs['receipt_attached']}\n\n"
        f"Current Status : {obs['approval_status']}\n"
        f"Issues Found   : {issues_str}\n"
        f"Last Result    : {obs['last_action_result']}\n"
        f"Remaining Steps: {obs['remaining_steps']}\n"
        f"Task           : {obs['task_description']}\n\n"
        "What is your next action? Reply with a single JSON object."
    )


# ---------------------------------------------------------------------------
# LLM call  (OpenAI client — required by hackathon rules)
# ---------------------------------------------------------------------------

def get_model_action(
    messages: List[Dict[str, Any]],
) -> str:
    """Call the LLM; return raw response text. Falls back to 'hello' on error."""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return text if text else "{}"
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "{}"


def parse_action(raw: str) -> Dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = "\n".join(
            line for line in raw.split("\n") if not line.startswith("```")
        )
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON in: {raw!r}")
    return json.loads(raw[start:end])


# ---------------------------------------------------------------------------
# Rule-based fallback agent  (used when no API key is available)
# ---------------------------------------------------------------------------

def rule_based_action(obs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministic agent — always completes the episode without an LLM.

    Decision logic mirrors the expected_action in tasks.py:
      - Only missing-info issues (gst/vendor)      → request_missing_info
      - Calculation error or fraud signals present  → mark_invalid
      - Duplicate flag                              → flag_duplicate
      - No issues found                             → mark_valid
    """
    detected = set(obs.get("detected_issues", []))
    status   = obs.get("approval_status", "pending")

    # --- 1. Detect issues not yet flagged ---
    if not obs.get("vendor_name") and "missing_vendor_name" not in detected:
        return {"action_type": "add_issue", "value": "missing_vendor_name"}
    if not obs.get("gst_number") and "missing_gst_number" not in detected:
        return {"action_type": "add_issue", "value": "missing_gst_number"}

    expected = round(obs["subtotal"] + obs["tax"], 2)
    if abs(obs["total"] - expected) > 0.01 and "wrong_total_calculation" not in detected:
        return {"action_type": "add_issue", "value": "wrong_total_calculation"}
    if obs.get("duplicate_flag") and "duplicate_invoice" not in detected:
        return {"action_type": "add_issue", "value": "duplicate_invoice"}
    if not obs.get("receipt_attached") and "missing_receipt" not in detected:
        return {"action_type": "add_issue", "value": "missing_receipt"}

    # --- 2. Set final status (only if still pending) ---
    if status == "pending":
        # Hard fraud / calculation issues → reject outright
        hard_issues = {"wrong_total_calculation", "duplicate_invoice", "missing_receipt"}
        if detected & hard_issues:
            return {"action_type": "mark_invalid"}
        # Only missing-info issues → request more info from vendor
        missing_info_issues = {"missing_gst_number", "missing_vendor_name"}
        if detected and detected.issubset(missing_info_issues):
            return {"action_type": "request_missing_info"}
        # No issues at all → approve
        if not detected:
            return {"action_type": "mark_valid"}
        return {"action_type": "mark_invalid"}

    # --- 3. Finalise ---
    return {"action_type": "finalize_review"}


# ---------------------------------------------------------------------------
# HTTP environment client
# ---------------------------------------------------------------------------

class EnvClient:
    def __init__(self, env_url: Optional[str] = None) -> None:
        self.env_url = env_url.rstrip("/") if env_url else None
        if self.env_url is None:
            from app.env import InvoiceValidationEnv
            self._env = InvoiceValidationEnv()

    def reset(self, difficulty: str = "easy") -> Dict[str, Any]:
        if self.env_url:
            r = requests.post(
                f"{self.env_url}/openenv/reset",
                json={"difficulty": difficulty},
                timeout=30,
            )
            r.raise_for_status()
            return r.json()
        obs = self._env.reset(difficulty=difficulty)
        return obs.model_dump()

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        if self.env_url:
            r = requests.post(
                f"{self.env_url}/openenv/step",
                json=action,
                timeout=30,
            )
            r.raise_for_status()
            return r.json()
        from app.models import InvoiceAction
        result = self._env.step(InvoiceAction(**action))
        return result.model_dump()


# ---------------------------------------------------------------------------
# Async main loop  (pattern required by OpenEnv sample)
# ---------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an invoice-validation episode."
    )
    parser.add_argument(
        "--task",
        choices=["easy", "medium", "hard"],
        default=TASK_NAME,
        help="Task difficulty (default: easy)",
    )
    parser.add_argument(
        "--env-url",
        default=ENV_URL_DEFAULT,
        help="FastAPI server URL (default: HF Space URL)",
    )
    args = parser.parse_args()

    task    = args.task
    env_url = None if args.env_url in ("", "local", "none") else args.env_url
    env     = EnvClient(env_url=env_url)

    history:  List[Dict[str, Any]] = []
    rewards:  List[float]          = []
    steps_taken = 0
    score       = 0.0
    success     = False

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME if USE_LLM else "rule-based")

    try:
        result = env.reset(difficulty=task)
        obs    = result
        done   = obs.get("done", False)

        messages: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            error: Optional[str] = None
            action_str = "finalize_review"

            try:
                if USE_LLM:
                    user_msg = build_user_prompt(obs, step)
                    messages.append({"role": "user", "content": user_msg})
                    raw   = get_model_action(messages)
                    action_dict = parse_action(raw)
                    messages.append({"role": "assistant", "content": json.dumps(action_dict)})
                else:
                    action_dict = rule_based_action(obs)

                action_str = action_dict.get("action_type", "finalize_review")

                step_result = env.step(action_dict)
                # reward is InvoiceReward dict — extract float total
                reward_raw  = step_result.get("reward", {})
                if isinstance(reward_raw, dict):
                    reward = float(reward_raw.get("total", 0.0))
                else:
                    reward = float(reward_raw or 0.0)

                done = step_result.get("done", False)
                obs  = step_result.get("observation", obs)

            except Exception as exc:
                error  = str(exc)
                reward = 0.0
                # Fall back to rule-based agent so episode continues cleanly
                try:
                    action_dict    = rule_based_action(obs)
                    action_str     = action_dict.get("action_type", "finalize_review")
                    step_result    = env.step(action_dict)
                    reward_raw     = step_result.get("reward", {})
                    reward = float(reward_raw.get("total", 0.0) if isinstance(reward_raw, dict) else reward_raw or 0.0)
                    done           = step_result.get("done", False)
                    obs            = step_result.get("observation", obs)
                except Exception as exc2:
                    error = f"{exc} | fallback: {exc2}"

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            history.append(
                {"step": step, "action": action_str, "reward": reward, "done": done}
            )

            if done:
                break

            # small pause to stay within free-tier rate limits
            await asyncio.sleep(0.3)

        # Normalise score to [0, 1]
        # The terminal reward from finalize_review includes the cumulative step
        # rewards already. Use max(rewards) as a robust fallback for edge cases
        # where only intermediate (non-terminal) rewards are positive.
        if rewards:
            terminal = rewards[-1] if done else 0.0
            score = max(terminal, min(1.0, sum(r for r in rewards if r > 0)))
            score = min(max(score, 0.0), 1.0)

        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[ERROR] Episode failed: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run() -> None:
    """Sync wrapper — required for pyproject.toml [project.scripts] entry."""
    asyncio.run(main())


if __name__ == "__main__":
    run()
