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
import json
import os
import textwrap
from urllib import request as urlrequest
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import httpx

from env import PotholeRepairEnv
from models import Action, ActionType, Observation, PotholeStatus


def load_trained_model(model_path: str):
    """
    Load trained CivicMind model from HF Hub or local path.
    Falls back to base model if trained model not found.
    """
    print(f"Loading model from {model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available()
            else torch.float32,
            device_map="auto" if torch.cuda.is_available()
            else "cpu",
            trust_remote_code=True
        )
        model.eval()
        device = next(model.parameters()).device
        print(f"✅ Trained model loaded on {device}")
        return model, tokenizer

    except Exception as e:
        print(f"[WARN] Could not load {model_path}: {e}")
        print("[WARN] Falling back to base model...")
        base = "Qwen/Qwen2.5-0.5B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(base)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            base,
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        model.eval()
        return model, tokenizer


def call_tool(space_url, endpoint, data) -> dict:
    """
    Makes HTTP call to tool endpoint on HF Space.
    Returns response dict or empty dict on error.
    """
    try:
        url = f"{space_url.rstrip('/')}{endpoint}"
        with httpx.Client(timeout=15.0) as client:
            if endpoint in {"/inspect", "/record_score"}:
                response = client.post(url, json=data or {})
            elif endpoint in {"/complaints", "/score"}:
                response = client.get(url)
            else:
                return {}
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else {}
    except Exception as exc:
        print(f"[DEBUG] tool call failed {endpoint}: {exc}", flush=True)
        return {}


def get_trained_model_action(
    model, tokenizer, obs: dict, task_name: str, inspect_result: Optional[dict] = None
) -> tuple[str, str]:
    """
    Run trained model inference to get action.
    Returns (action_type, pothole_id)
    """
    # Build prompt same as training
    potholes = obs.get("potholes", [])
    pending = [p for p in potholes
               if p.get("status") == "pending"]
    pending_sorted = sorted(
        pending,
        key=lambda p: (-p.get("severity", 0),
                       -p.get("daily_traffic", 0))
    )[:5]

    # Fallback
    fallback_id = pending[0].get("id") if pending else "POT_001"

    weather = obs.get("weather", {})
    rain = "RAINING - avoid dispatch!" \
        if weather.get("is_raining") else "Clear"

    lines = []
    for p in pending_sorted:
        sev = p.get("severity", 0)
        sev_str = "UNKNOWN" if sev == 0 else str(sev)
        lines.append(
            f"  {p.get('id')} | sev={sev_str} | "
            f"{p.get('road_type')} | "
            f"traffic={p.get('daily_traffic',0):,} | "
            f"cost=Rs{p.get('repair_cost',0):.0f}"
        )

    pothole_block = "\n".join(lines) if lines else "None pending"

    prompt = f"""You are a city road repair agent.
Task: {task_name}
Day: {obs.get('day',1)} of {obs.get('max_days',30)}
Budget: Rs{obs.get('budget_remaining',0):,.0f}
Crews: {obs.get('crews_available',0)}
Weather: {rain}
Fixed: {obs.get('total_fixed',0)} | Pending: {obs.get('total_pending',0)}

Top pending potholes:
{pothole_block}

Choose ONE action. Reply with EXACTLY:
dispatch POT_001
or defer POT_001
or mark_low POT_001

Your action:"""
    if inspect_result:
        prompt += (
            "\nInspector says: "
            f"severity={inspect_result.get('real_severity', 'unknown')}, "
            f"recommendation={inspect_result.get('recommendation', 'n/a')}"
        )

    try:
        device = next(model.parameters()).device
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(
            new_tokens, skip_special_tokens=True
        ).strip()

        print(f"[DEBUG] Model said: {repr(response)}")

        # Parse response
        text = response.strip().lower().split("\n")[0]
        parts = text.split()
        if len(parts) < 2:
            return "defer", fallback_id

        verb = parts[0]
        pot_id = parts[1].upper()

        valid = {
            "dispatch": "dispatch",
            "defer": "defer",
            "mark_low": "mark_low_priority",
            "mark_low_priority": "mark_low_priority"
        }
        action_type = valid.get(verb, "defer")

        all_ids = [p.get("id") for p in potholes]
        if pot_id not in all_ids:
            pot_id = fallback_id

        return action_type, pot_id

    except Exception as e:
        print(f"[DEBUG] Model inference failed: {e}")
        return "defer", fallback_id

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
BASE_URL = os.getenv(
    "HF_SPACE_URL",
    "https://meet25284-pothole-repair-env.hf.space"
)
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

async def run_task(client, task_name, model, tokenizer) -> float:
    """
    Run one full episode of a task.
    Prints [START], [STEP]×n, [END] logs.
    Returns final score in [0.0, 1.0].
    """
    env = PotholeRepairEnv(task_name=task_name)
    rewards: List[float] = []
    actions_taken: List[dict] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset(task_name=task_name)

        for step in range(1, MAX_STEPS_PER_TASK + 1):
            if not [
                p
                for p in obs.potholes
                if p.status == PotholeStatus.PENDING or str(p.status) == "pending"
            ]:
                # No more pending potholes — done
                break

            # Step 1 — inspect top pending pothole before deciding.
            pending = [p for p in obs.potholes if p.status == "pending"]
            inspect_result: dict = {}
            if pending:
                top_pothole = sorted(pending, key=lambda p: -p.severity)[0]
                inspect_result = call_tool(
                    BASE_URL, "/inspect", {"pothole_id": top_pothole.id}
                )
                print(f"[TOOL] inspect → {inspect_result}", flush=True)

            # Step 2 — pass enriched tool context to the model prompt.
            action_type, pothole_id = get_trained_model_action(
                model, tokenizer, obs.model_dump(), task_name, inspect_result
            )
            action = Action(action_type=action_type, pothole_id=pothole_id)
            result = env.step(action)

            # Step 3 — query current score and attach to info.
            current_score = call_tool(BASE_URL, "/score", {})
            if isinstance(result.info, dict):
                result.info["tool_score"] = current_score

            reward = result.reward or 0.0
            done   = result.done
            error  = result.last_action_error

            rewards.append(reward)
            steps_taken = step
            cost = 0.0
            if isinstance(result.info, dict):
                try:
                    cost = float(result.info.get("cost", 0.0))
                except (TypeError, ValueError):
                    cost = 0.0
            actions_taken.append(
                {
                    "action_type": action.action_type,
                    "pothole_id": action.pothole_id,
                    "day": obs.day,
                    "cost": cost,
                }
            )
            obs = result.observation

            raw_action = f"{action_type} {pothole_id}"
            action_str = raw_action
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

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
        record_result = call_tool(
            BASE_URL,
            "/record_score",
            {"task": task_name, "score": score, "steps": steps_taken},
        )
        if not record_result.get("recorded"):
            print(f"[DEBUG] record_score failed: {record_result}", flush=True)

    return score


# ─────────────────────────────────────────────
# Main — runs all 3 tasks
# ─────────────────────────────────────────────

async def main() -> None:
    TRAINED_MODEL_PATH = os.getenv(
        "TRAINED_MODEL_PATH",
        "meet25284/civicmind-agent"
    )
    model, tokenizer = load_trained_model(TRAINED_MODEL_PATH)

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
        score = await run_task(client, task_name, model, tokenizer)
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
