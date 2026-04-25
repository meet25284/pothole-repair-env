"""
budget_agent.py - Budget Agent for CivicMind multi-agent system.
"""

from __future__ import annotations

import os
import textwrap

from openai import OpenAI

from models import Observation
from tools.budget_tools import (
    approve_spend,
    check_budget,
    estimate_cost,
    get_spending_history,
)


SYSTEM_PROMPT = """You are the Budget Agent for a city road repair team.
Your job is to approve or reject repair requests based on available funds.
You must be strict but fair — approve critical repairs even if expensive,
reject low-priority repairs when budget is below 20%.
Respond in this exact format:
APPROVED: <comma separated pothole_ids>
REJECTED: <comma separated pothole_ids>
REASON: <one sentence explaining your decision>
BUDGET_AFTER: <estimated remaining budget as number>"""


class BudgetAgent:
    def __init__(self, client: OpenAI):
        self.client = client
        self.model = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")

    def evaluate(
        self, obs: Observation, requested_ids: list[str], actions_taken: list[dict]
    ) -> dict:
        budget_status = check_budget(obs.budget_remaining, obs.initial_budget)

        try:
            estimates = [
                estimate_cost(p_id, obs.potholes, budget_remaining=obs.budget_remaining)
                for p_id in requested_ids
            ]
            spending_history = get_spending_history(actions_taken)

            estimate_lines = []
            for requested_id, estimate in zip(requested_ids, estimates):
                if "error" in estimate:
                    estimate_lines.append(f"- {requested_id}: not found")
                    continue

                final_cost = float(estimate.get("final_cost", 0.0))
                affordable_now = final_cost <= obs.budget_remaining
                estimate_lines.append(
                    (
                        f"- {requested_id}: base={estimate.get('base_cost')} | "
                        f"multiplier={estimate.get('urgency_multiplier')} | "
                        f"final={final_cost:.2f} | affordable_now={affordable_now}"
                    )
                )

            user_prompt = textwrap.dedent(
                f"""
                Day: {obs.day}
                Budget status:
                - remaining: {budget_status["remaining"]}
                - initial: {budget_status["initial"]}
                - used_percent: {budget_status["used_percent"]}
                - status: {budget_status["status"]}

                Requested repairs:
                {chr(10).join(estimate_lines) if estimate_lines else "- none"}

                Spending history:
                - total_dispatches: {spending_history["total_dispatches"]}
                - total_spent: {spending_history["total_spent"]}
                - avg_cost_per_repair: {spending_history["avg_cost_per_repair"]}
                - most_expensive_day: {spending_history["most_expensive_day"]}

                Which requests should be approved or rejected?
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
            parsed = self._parse_response(raw_response, requested_ids, obs)

            return {
                "approved_ids": parsed["approved_ids"],
                "rejected_ids": parsed["rejected_ids"],
                "reason": parsed["reason"],
                "estimated_budget_after": parsed["estimated_budget_after"],
                "budget_status": budget_status,
                "raw_response": raw_response,
            }
        except Exception:
            remaining_budget = float(obs.budget_remaining)
            approved_ids: list[str] = []
            rejected_ids: list[str] = []

            for requested_id in requested_ids:
                estimate = estimate_cost(requested_id, obs.potholes, remaining_budget)
                if "error" in estimate:
                    rejected_ids.append(requested_id)
                    continue

                decision = approve_spend(
                    amount=float(estimate["final_cost"]),
                    budget_remaining=remaining_budget,
                    reason="Fallback affordability check",
                )
                if decision["approved"]:
                    approved_ids.append(requested_id)
                    remaining_budget = float(decision["budget_after"])
                else:
                    rejected_ids.append(requested_id)

            return {
                "approved_ids": approved_ids,
                "rejected_ids": rejected_ids,
                "reason": "Budget evaluation fallback used",
                "estimated_budget_after": remaining_budget,
                "budget_status": budget_status,
                "raw_response": "",
            }

    def _parse_response(self, text: str, requested_ids: list[str], obs: Observation) -> dict:
        """
        Parse APPROVED/REJECTED/REASON/BUDGET_AFTER lines with affordability fallback.
        """
        approved_ids: list[str] = []
        rejected_ids: list[str] = []
        reason = ""
        estimated_budget_after: float | None = None

        for line in text.splitlines():
            cleaned = line.strip()
            upper_cleaned = cleaned.upper()

            if upper_cleaned.startswith("APPROVED:"):
                payload = cleaned.split(":", 1)[1].strip()
                approved_ids = [
                    item.strip()
                    for item in payload.split(",")
                    if item.strip() and item.strip().lower() not in {"none", "n/a"}
                ]
            elif upper_cleaned.startswith("REJECTED:"):
                payload = cleaned.split(":", 1)[1].strip()
                rejected_ids = [
                    item.strip()
                    for item in payload.split(",")
                    if item.strip() and item.strip().lower() not in {"none", "n/a"}
                ]
            elif upper_cleaned.startswith("REASON:"):
                reason = cleaned.split(":", 1)[1].strip()
            elif upper_cleaned.startswith("BUDGET_AFTER:"):
                payload = cleaned.split(":", 1)[1].strip()
                numeric_payload = payload.replace(",", "").replace("₹", "").replace("$", "")
                try:
                    estimated_budget_after = float(numeric_payload)
                except ValueError:
                    estimated_budget_after = None

        requested_set = set(requested_ids)
        approved_ids = [p_id for p_id in approved_ids if p_id in requested_set]
        rejected_ids = [p_id for p_id in rejected_ids if p_id in requested_set]

        # If parser yields nothing useful, fallback to affordability checks.
        if not approved_ids and not rejected_ids:
            remaining_budget = float(obs.budget_remaining)
            for requested_id in requested_ids:
                estimate = estimate_cost(requested_id, obs.potholes, remaining_budget)
                if "error" in estimate:
                    rejected_ids.append(requested_id)
                    continue
                if float(estimate["final_cost"]) <= remaining_budget:
                    approved_ids.append(requested_id)
                    remaining_budget -= float(estimate["final_cost"])
                else:
                    rejected_ids.append(requested_id)
            if not reason:
                reason = "Parsed response incomplete; used affordability fallback"
            if estimated_budget_after is None:
                estimated_budget_after = remaining_budget

        # Keep outputs coherent if LLM includes only one side.
        if not rejected_ids:
            rejected_ids = [p_id for p_id in requested_ids if p_id not in set(approved_ids)]
        if not approved_ids:
            approved_ids = [p_id for p_id in requested_ids if p_id not in set(rejected_ids)]

        if estimated_budget_after is None:
            running_budget = float(obs.budget_remaining)
            for approved_id in approved_ids:
                estimate = estimate_cost(approved_id, obs.potholes, running_budget)
                if "error" in estimate:
                    continue
                running_budget -= float(estimate["final_cost"])
            estimated_budget_after = running_budget

        return {
            "approved_ids": approved_ids,
            "rejected_ids": rejected_ids,
            "reason": reason,
            "estimated_budget_after": estimated_budget_after,
        }
