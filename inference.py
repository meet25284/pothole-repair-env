"""
Inference Script — City Pothole Repair Scheduler
=================================================
MANDATORY FORMAT (judges auto-evaluate this stdout):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>

Environment variables required:
    API_BASE_URL       LLM API endpoint
    MODEL_NAME         Model identifier
    HF_TOKEN           HuggingFace / API key
"""

from __future__ import annotations
import asyncio
import os
import textwrap
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from env import PotholeRepairEnv
from models import Action, ActionType, Observation

# ─────────────────────────────────────────────
# Config — read from environment variables
# ─────────────────────────────────────────────

load_dotenv()

_API_KEY_RAW = (
    os.getenv("HF_TOKEN")
    or os.getenv("GROQ_API_KEY")
    or os.getenv("OPENAI_API_KEY")
    or os.getenv("API_KEY")
    or ""
)
API_KEY = _API_KEY_RAW.strip() or None
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
BENCHMARK = "pothole-repair-env"

# OpenAI client requires a non-empty string; real calls still fail gracefully in get_agent_action.
_DUMMY_API_KEY = "local-missing-key"

# Tasks to run in order: easy → medium → hard
TASKS = ["critical_repair", "budget_optimizer", "full_city_manager"]

MAX_STEPS_PER_TASK   = 45
SUCCESS_THRESHOLD    = 0.5
TEMPERATURE          = 0.3   # low = more deterministic decisions
MAX_TOKENS           = 100


# ─────────────────────────────────────────────
# MANDATORY log functions — do NOT change format
# ─────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ─────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
    You are a city road maintenance AI agent for Ahmedabad, India.
    Each step you see a list of potholes with their severity, road type, traffic, and repair cost.
    You also see your remaining budget, crew count, current day, and weather.

    Your job: choose ONE action per step.

    Available actions:
      dispatch <pothole_id>     → send crew to fix this pothole today
      defer <pothole_id>        → postpone repair (saves budget/crew for now)
      mark_low <pothole_id>     → mark as low priority (skip for this episode)

    Rules:
      - Do NOT dispatch on rainy days (weather.is_raining=True) — work quality suffers
      - Prioritize severity 4-5 potholes on highways and arterial roads first
      - Do NOT dispatch if repair_cost > budget_remaining
      - Prefer fixing high-traffic roads for maximum public benefit

    Reply with EXACTLY this format (nothing else):
      dispatch POT_001
    or
      defer POT_005
    or
      mark_low POT_012
""").strip()


def build_user_prompt(obs: Observation, step: int) -> str:
    """Build a clear prompt describing current observation."""
    pending = [p for p in obs.potholes if p.status == "pending"]
    pending_sorted = sorted(pending, key=lambda p: (-p.severity, -p.daily_traffic))

    pothole_lines = []
    for p in pending_sorted[:10]:  # show top 10 to keep prompt short
        pothole_lines.append(
            f"  {p.id} | sev={p.severity} | {p.road_type} | traffic={p.daily_traffic:,} | cost=₹{p.repair_cost:.0f}"
        )

    pothole_block = "\n".join(pothole_lines) if pothole_lines else "  (none pending)"

    return textwrap.dedent(f"""
        Step {step} | Day {obs.day}/{obs.max_days}
        Budget remaining : ₹{obs.budget_remaining:,.0f}
        Crews available  : {obs.crews_available}
        Weather today    : {obs.weather.condition} | raining={obs.weather.is_raining}
        Fixed so far     : {obs.total_fixed} | Still pending: {obs.total_pending}

        Pending potholes (sorted by priority):
        {pothole_block}

        Choose ONE action now:
    """).strip()


# ─────────────────────────────────────────────
# LLM call
# ─────────────────────────────────────────────

def get_agent_action(client: OpenAI, obs: Observation, step: int) -> tuple[str, Action]:
    """
    Call LLM to choose an action.
    Returns (raw_action_str, Action object).
    Falls back to defer on first pending pothole if LLM fails.
    """
    # Find first pending pothole as fallback
    pending = [p for p in obs.potholes if p.status == "pending"]
    fallback_id = pending[0].id if pending else "POT_001"
    fallback_action = Action(action_type=ActionType.DEFER, pothole_id=fallback_id)

    try:
        user_prompt = build_user_prompt(obs, step)
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip().lower()
        action = parse_action(raw, fallback_id)
        return raw, action

    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        return f"defer {fallback_id}", fallback_action


def parse_action(raw: str, fallback_id: str) -> Action:
    """
    Parse LLM output into an Action object.
    Expected format: 'dispatch POT_001' / 'defer POT_005' / 'mark_low POT_012'
    """
    try:
        parts = raw.strip().split()
        if len(parts) < 2:
            raise ValueError("Too few parts")

        verb      = parts[0].lower()
        pothole_id = parts[1].upper()

        if verb == "dispatch":
            return Action(action_type=ActionType.DISPATCH, pothole_id=pothole_id)
        elif verb == "defer":
            return Action(action_type=ActionType.DEFER, pothole_id=pothole_id)
        elif verb in ("mark_low", "mark_low_priority"):
            return Action(action_type=ActionType.MARK_LOW_PRIORITY, pothole_id=pothole_id)
        else:
            raise ValueError(f"Unknown verb: {verb}")

    except Exception:
        return Action(action_type=ActionType.DEFER, pothole_id=fallback_id)


# ─────────────────────────────────────────────
# Single task runner
# ─────────────────────────────────────────────

async def run_task(client: OpenAI, task_name: str) -> float:
    """
    Run one full episode of a task.
    Prints [START], [STEP]×n, [END] logs.
    Returns final score in [0.0, 1.0].
    """
    env = PotholeRepairEnv(task_name=task_name)
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset(task_name=task_name)

        for step in range(1, MAX_STEPS_PER_TASK + 1):
            if not [p for p in obs.potholes if p.status == "pending"]:
                # No more pending potholes — done
                break

            raw_action, action = get_agent_action(client, obs, step)
            result = env.step(action)

            reward = result.reward or 0.0
            done   = result.done
            error  = result.last_action_error

            rewards.append(reward)
            steps_taken = step
            obs = result.observation

            log_step(step=step, action=raw_action, reward=reward, done=done, error=error)

            if done:
                break

        # Compute final score using grader
        score = env.get_final_score()
        score = min(max(score, 0.0), 1.0)   # always clamp to [0, 1]
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Task {task_name} error: {exc}", flush=True)

    finally:
        try:
            env.close()
        except Exception:
            pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ─────────────────────────────────────────────
# Main — runs all 3 tasks
# ─────────────────────────────────────────────

async def main() -> None:
    if not API_KEY:
        print(
            "[WARN] No HF_TOKEN / GROQ_API_KEY / OPENAI_API_KEY in environment; "
            "LLM calls will fail and the rule-based fallback in get_agent_action will be used.",
            flush=True,
        )
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY or _DUMMY_API_KEY,
    )

    print(f"[INFO] Starting inference | model={MODEL_NAME} | tasks={TASKS}", flush=True)
    print(f"[INFO] API base: {API_BASE_URL}", flush=True)
    print("─" * 60, flush=True)

    all_scores = {}
    for task_name in TASKS:
        score = await run_task(client, task_name)
        all_scores[task_name] = score
        print("─" * 60, flush=True)

    # Final summary
    print("\n[SUMMARY]", flush=True)
    for task, s in all_scores.items():
        status = "PASS" if s >= SUCCESS_THRESHOLD else "FAIL"
        print(f"  {task:<25} score={s:.3f}  [{status}]", flush=True)

    avg = sum(all_scores.values()) / len(all_scores)
    print(f"\n  Average score: {avg:.3f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
