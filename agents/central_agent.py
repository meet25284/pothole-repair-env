"""
central_agent.py - Orchestrator agent for CivicMind multi-agent workflow.
"""

from __future__ import annotations

import asyncio
import os
import textwrap
from typing import Any

from openai import OpenAI

from agents.budget_agent import BudgetAgent
from agents.inspector_agent import InspectorAgent
from agents.risk_agent import RiskAgent
from agents.scheduler_agent import SchedulerAgent
from models import Action, ActionType, Observation, PotholeStatus


SYSTEM_PROMPT = """You are the Central Command Agent managing city road repairs.
You have received reports from 4 specialist agents.
Your job is to choose ONE final action based on all their reports.
Always trust the Risk Agent on weather decisions.
Always trust the Budget Agent on financial decisions.
Choose the highest impact action available.
Respond with EXACTLY this format:
ACTION: dispatch <pothole_id>
OR
ACTION: defer <pothole_id>
OR
ACTION: mark_low <pothole_id>
REASON: <one sentence why>"""


class CentralAgent:
    def __init__(self, client: OpenAI):
        self.client = client
        self.model = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
        self.inspector = InspectorAgent(client)
        self.budget = BudgetAgent(client)
        self.risk = RiskAgent(client)
        self.scheduler = SchedulerAgent(client)

    async def decide(
        self, obs: Observation, actions_taken: list[dict], env: Any = None
    ) -> tuple[Action, dict]:
        pending = [p for p in obs.potholes if p.status == PotholeStatus.PENDING]
        if not pending:
            fallback_action = Action(action_type=ActionType.DEFER, pothole_id="POT_001")
            return (
                fallback_action,
                {
                    "inspector_report": {},
                    "risk_report": {},
                    "budget_report": {},
                    "scheduler_report": {},
                    "final_action_str": "defer POT_001",
                    "final_reason": "No pending potholes found; using safe default defer action.",
                },
            )

        fallback_id = pending[0].id

        try:
            # Keep async signature explicit while using sync specialists.
            await asyncio.sleep(0)

            # Partial observability: reveal top unknown pending potholes before inspection.
            if env is not None and hasattr(env, "reveal_severity") and hasattr(env, "state"):
                unknown_pending = [
                    p for p in pending if p.severity == 0 and p.status == PotholeStatus.PENDING
                ]
                unknown_pending = sorted(
                    unknown_pending, key=lambda p: p.daily_traffic, reverse=True
                )[:3]
                for pothole in unknown_pending:
                    env.reveal_severity(pothole.id)
                if unknown_pending:
                    obs = env.state()
                    pending = [p for p in obs.potholes if p.status == PotholeStatus.PENDING]
                    if pending:
                        fallback_id = pending[0].id

            inspector_report = self.inspector.inspect(obs)

            urgent_ids = inspector_report.get("urgent_ids", [])
            if not urgent_ids:
                urgent_ids = [p.id for p in pending[:3]]

            risk_report = self.risk.assess(obs, urgent_ids)
            cleared_ids = risk_report.get("cleared_ids", [])
            emergency_ids = risk_report.get("emergency_ids", [])

            if emergency_ids:
                candidates = emergency_ids
            else:
                candidates = cleared_ids

            if not candidates and not obs.weather.is_raining:
                candidates = urgent_ids
            if not candidates and obs.weather.is_raining:
                action = Action(action_type=ActionType.DEFER, pothole_id=fallback_id)
                consultation_log = {
                    "inspector_report": inspector_report,
                    "risk_report": risk_report,
                    "budget_report": {},
                    "scheduler_report": {},
                    "final_action_str": f"defer {fallback_id}",
                    "final_reason": "No safe candidates in rainy conditions; deferring by risk policy.",
                }
                return action, consultation_log

            budget_report = self.budget.evaluate(obs, candidates, actions_taken)
            approved_ids = budget_report.get("approved_ids", [])

            if not approved_ids:
                action = Action(action_type=ActionType.DEFER, pothole_id=fallback_id)
                consultation_log = {
                    "inspector_report": inspector_report,
                    "risk_report": risk_report,
                    "budget_report": budget_report,
                    "scheduler_report": {},
                    "final_action_str": f"defer {fallback_id}",
                    "final_reason": "No budget-approved candidates; deferring safest fallback item.",
                }
                return action, consultation_log

            scheduler_report = self.scheduler.schedule(obs, approved_ids)
            assigned = scheduler_report.get("actual_assignments", {})

            user_prompt = textwrap.dedent(
                f"""
                Day: {obs.day}

                Inspector says:
                - urgent_ids: {inspector_report.get("urgent_ids", [])}
                - summary: {inspector_report.get("summary", "")}

                Risk says:
                - weather: {"SAFE" if risk_report.get("weather_safe") else "RISKY"}
                - cleared_ids: {risk_report.get("cleared_ids", [])}
                - advice: {risk_report.get("advice", "")}

                Budget says:
                - approved_ids: {budget_report.get("approved_ids", [])}
                - reason: {budget_report.get("reason", "")}
                - budget_status: {budget_report.get("budget_status", {})}

                Scheduler says:
                - assignments: {assigned}
                - note: {scheduler_report.get("note", "")}

                Choose ONE final action now.
                """
            ).strip()

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            raw_response = (response.choices[0].message.content or "").strip()
            action = self._parse_action(raw_response, fallback_id=fallback_id)

            final_reason = ""
            for line in raw_response.splitlines():
                if line.strip().upper().startswith("REASON:"):
                    final_reason = line.split(":", 1)[1].strip()
                    break

            consultation_log = {
                "inspector_report": inspector_report,
                "risk_report": risk_report,
                "budget_report": budget_report,
                "scheduler_report": scheduler_report,
                "final_action_str": f"{action.action_type} {action.pothole_id}",
                "final_reason": final_reason,
            }
            return action, consultation_log

        except Exception as exc:
            error_action = Action(action_type=ActionType.DEFER, pothole_id=fallback_id)
            error_log = {
                "inspector_report": {},
                "risk_report": {},
                "budget_report": {},
                "scheduler_report": {},
                "final_action_str": f"defer {fallback_id}",
                "final_reason": "Central decision failed; deferred by safety fallback.",
                "error": str(exc),
            }
            return error_action, error_log

    def _parse_action(self, text: str, fallback_id: str) -> Action:
        """
        Parse ACTION line and convert to Action object.
        Expected line: ACTION: dispatch POT_001 / defer POT_001 / mark_low POT_001
        """
        for line in text.splitlines():
            cleaned = line.strip()
            if not cleaned.upper().startswith("ACTION:"):
                continue

            payload = cleaned.split(":", 1)[1].strip()
            parts = payload.split()
            if len(parts) < 2:
                break

            verb = parts[0].lower()
            pothole_id = parts[1].strip()
            if not pothole_id:
                break

            if verb == "dispatch":
                return Action(action_type=ActionType.DISPATCH, pothole_id=pothole_id)
            if verb == "defer":
                return Action(action_type=ActionType.DEFER, pothole_id=pothole_id)
            if verb in {"mark_low", "mark_low_priority"}:
                return Action(action_type=ActionType.MARK_LOW_PRIORITY, pothole_id=pothole_id)
            break

        return Action(action_type=ActionType.DEFER, pothole_id=fallback_id)
