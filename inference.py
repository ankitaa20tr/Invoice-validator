"""
inference.py — run a full episode using an LLM as the agent.

This script:
  1. Connects to the FastAPI environment server (or uses the env directly
  2. Calls reset() with a given difficulty level
  3. At each step, formats the observation into a prompt and calls the LLM
  4. Parses the LLM response into an InvoiceAction
  5. Calls env.step() with that action
  6. Logs every step to stdout in the exact OpenEnv structured format
  7. Prints a final score line

Environment variables
---------------------
  API_BASE_URL   — LLM provider base URL     (default: Groq endpoint)
  MODEL_NAME     — model identifier          (default: llama-3.3-70b-versatile)
  OPENAI_API_KEY — API key for the provider  (no default — must be set)
  HF_TOKEN       — Hugging Face token        (no default — optional)
  LOCAL_IMAGE_NAME — Docker image name       (no default — optional)

Supported free providers
------------------------
  Groq:  API_BASE_URL=https://api.groq.com/openai/v1
  HF:    API_BASE_URL=https://api-inference.huggingface.co/v1

Usage
-----
  python inference.py --task easy
  python inference.py --task hard --env-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# Environment variable config
# Checklist requirements:
#   - API_BASE_URL and MODEL_NAME MUST have defaults
#   - OPENAI_API_KEY, HF_TOKEN, LOCAL_IMAGE_NAME must NOT have defaults
# ---------------------------------------------------------------------------

API_BASE_URL     = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME       = os.getenv("MODEL_NAME",   "llama-3.3-70b-versatile")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
HF_TOKEN         = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# Build the OpenAI-compatible client using the env-provided variables.
# This covers Groq, HF Inference, and any other OpenAI-compatible provider.
client = OpenAI(
    api_key=OPENAI_API_KEY or "not-set",
    base_url=API_BASE_URL,
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
    issues_found = obs.get("detected_issues", [])
    issues_str   = ", ".join(issues_found) if issues_found else "none yet"

    expected_total = obs["subtotal"] + obs["tax"]

    return f"""Step {step} — Invoice Review

Invoice ID    : {obs["invoice_id"]}
Vendor        : {obs.get("vendor_name") or "NOT PROVIDED"}
Subtotal      : {obs["subtotal"]}
Tax           : {obs["tax"]}
Listed Total  : {obs["total"]}
Expected Total: {expected_total}
GST Number    : {obs.get("gst_number") or "NOT PROVIDED"}
Invoice Date  : {obs.get("invoice_date") or "NOT PROVIDED"}
Duplicate Flag: {obs["duplicate_flag"]}
Receipt Att.  : {obs["receipt_attached"]}

Current Status : {obs["approval_status"]}
Issues Found   : {issues_str}
Last Result    : {obs["last_action_result"]}
Remaining Steps: {obs["remaining_steps"]}
Task           : {obs["task_description"]}

What is your next action? Reply with a single JSON object."""


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_llm(messages: list) -> str:
    """Call the LLM and return raw text content."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.0,
        max_tokens=256,
    )
    return response.choices[0].message.content.strip()


def parse_action(raw: str) -> Dict[str, Any]:
    """
    Extract the first JSON object from the LLM response.
    Handles cases where the model wraps JSON in markdown code blocks.
    """
    raw = raw.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(
            line for line in lines
            if not line.startswith("```")
        )
    # Find the first { ... } block
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON found in LLM response: {raw!r}")
    return json.loads(raw[start:end])


# ---------------------------------------------------------------------------
# Environment client
# ---------------------------------------------------------------------------

class EnvClient:
    """
    Thin wrapper that calls the FastAPI endpoints.
    If env_url is None, import and use the env directly (in-process mode).
    """

    def __init__(self, env_url: Optional[str] = None):
        self.env_url = env_url.rstrip("/") if env_url else None

        if self.env_url is None:
            # import directly — avoids needing the server running
            from app.env import InvoiceValidationEnv
            self._env = InvoiceValidationEnv()

    def reset(self, difficulty: str) -> Dict[str, Any]:
        if self.env_url:
            r = requests.post(
                f"{self.env_url}/reset",
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
                f"{self.env_url}/step",
                json=action,
                timeout=30,
            )
            r.raise_for_status()
            return r.json()
        from app.models import InvoiceAction
        result = self._env.step(InvoiceAction(**action))
        return result.model_dump()

    def get_state(self) -> Optional[Dict[str, Any]]:
        if self.env_url:
            try:
                r = requests.get(f"{self.env_url}/state", timeout=10)
                return r.json()
            except Exception:
                return None
        try:
            return self._env.state().model_dump()
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_episode(task: str, env_url: Optional[str]) -> float:
    env = EnvClient(env_url=env_url)

    # Runtime check — warn if key is missing rather than crashing at import
    if not OPENAI_API_KEY:
        print("[ERROR] OPENAI_API_KEY is not set. Set it via environment variable.")
        sys.exit(1)

    print(f"[START] task={task} env={env_url or 'in-process'} model={MODEL_NAME}")

    obs = env.reset(difficulty=task)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    step_num      = 0
    episode_score = 0.0   # final clamped score returned by env on done=True
    done          = obs.get("done", False)


    while not done:
        step_num += 1
        user_msg = build_user_prompt(obs, step_num)
        messages.append({"role": "user", "content": user_msg})

        # Get action from LLM
        try:
            raw_response = call_llm(messages)
            action_dict  = parse_action(raw_response)
        except Exception as exc:
            print(f"[STEP] step={step_num} action=parse_error reward=0.0 done=False error={exc}")
            action_dict = {"action_type": "finalize_review"}

        messages.append({"role": "assistant", "content": json.dumps(action_dict)})

        result = env.step(action_dict)
        reward = result["reward"]["total"]
        done   = result["done"]
        obs    = result["observation"]

        print(
            f"[STEP] step={step_num} "
            f"action={action_dict.get('action_type')} "
            f"value={action_dict.get('value', 'N/A')} "
            f"reward={reward:.4f} "
            f"done={done}"
        )

        # When done=True env returns the combined (step rewards + terminal)
        # score already clamped to [0, 1].  We use that directly.
        if done:
            episode_score = reward
        else:
            # small pause to stay within free-tier rate limits
            time.sleep(0.3)

    success = episode_score >= 0.5
    print(f"[END] success={success} steps={step_num} score={round(episode_score, 4)}")
    return episode_score



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run an invoice-validation episode with an LLM agent."
    )
    parser.add_argument(
        "--task",
        choices=["easy", "medium", "hard"],
        default="easy",
        help="Task difficulty (default: easy)",
    )
    parser.add_argument(
        "--env-url",
        default=None,
        help=(
            "Optional FastAPI server URL (e.g. http://localhost:8000). "
            "If not provided, the env runs in-process."
        ),
    )
    args = parser.parse_args()
    run_episode(task=args.task, env_url=args.env_url)


if __name__ == "__main__":
    main()
