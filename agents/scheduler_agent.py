"""
scheduler_agent.py - Scheduler Agent for CivicMind multi-agent system.
"""

from __future__ import annotations

import os
import textwrap

from openai import OpenAI

from models import Observation
from tools.scheduler_tools import (
    assign_crew,
    get_crew_status,
    get_workload,
    reset_crews,
)


SYSTEM_PROMPT = """You are the Scheduler Agent for a city road repair team.
Your job is to assign available crews to approved repair jobs.
Never assign more jobs than available crews.
Prioritize assigning crews to highest severity potholes first.
Respond in this exact format:
ASSIGNMENTS: <pothole_id>=<crew_id>, <pothole_id>=<crew_id>
SKIPPED: <pothole_ids that could not be assigned due to no crew>
WORKLOAD: <current crew utilization as percentage>
NOTE: <one sentence about crew availability>"""


class SchedulerAgent:
    def __init__(self, client: OpenAI):
        self.client = client
        self.model = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")

    def schedule(self, obs: Observation, approved_ids: list[str]) -> dict:
        # Reset daily crew state before planning assignments.
        reset_crews(obs.crews_available)
        crew_status = get_crew_status(obs.crews_available)
        initial_workload = get_workload()

        pothole_by_id = {p.id: p for p in obs.potholes}
        sorted_ids = sorted(
            approved_ids,
            key=lambda p_id: (
                pothole_by_id.get(p_id).severity if pothole_by_id.get(p_id) else -1,
                pothole_by_id.get(p_id).daily_traffic if pothole_by_id.get(p_id) else -1,
            ),
            reverse=True,
        )

        actual_assignments: dict[str, str] = {}
        skipped_ids: list[str] = []

        for pothole_id in sorted_ids:
            result = assign_crew(pothole_id, obs.crews_available)
            if result.get("assigned"):
                actual_assignments[pothole_id] = result["crew_id"]
            else:
                skipped_ids.append(pothole_id)
                # Stop once crews are exhausted, remaining are skipped.
                remaining = [p_id for p_id in sorted_ids if p_id not in actual_assignments and p_id != pothole_id]
                skipped_ids.extend(remaining)
                break

        final_workload = get_workload()
        post_status = get_crew_status(obs.crews_available)

        jobs_with_severity = []
        for p_id in sorted_ids:
            pothole = pothole_by_id.get(p_id)
            if pothole is None:
                jobs_with_severity.append(f"- {p_id}: not found")
            else:
                jobs_with_severity.append(
                    f"- {p_id}: severity={pothole.severity}, traffic={pothole.daily_traffic}"
                )

        try:
            user_prompt = textwrap.dedent(
                f"""
                Day: {obs.day}
                Available crews: {crew_status["available"]}/{crew_status["total_crews"]}

                Approved jobs (highest severity first):
                {chr(10).join(jobs_with_severity) if jobs_with_severity else "- none"}

                Current workload:
                - before_utilization_percent: {initial_workload["utilization_percent"]}
                - after_utilization_percent: {final_workload["utilization_percent"]}
                - total_jobs_today: {final_workload["total_jobs_today"]}
                - can_dispatch_more: {post_status["can_dispatch"]}

                Proposed actual assignments from scheduler tools:
                - {actual_assignments}
                - skipped: {skipped_ids}

                Confirm assignments.
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
            parsed = self._parse_response(raw_response)

            return {
                "assignments": parsed["assignments"],
                "skipped_ids": parsed["skipped_ids"],
                "utilization": parsed["utilization"],
                "note": parsed["note"],
                "actual_assignments": actual_assignments,
                "raw_response": raw_response,
            }
        except Exception:
            # Greedy fallback already happened above via direct assign_crew calls.
            return {
                "assignments": dict(actual_assignments),
                "skipped_ids": skipped_ids,
                "utilization": f"{final_workload['utilization_percent']:.1f}%",
                "note": "LLM confirmation failed; using direct crew assignment results.",
                "actual_assignments": actual_assignments,
                "raw_response": "",
            }

    def _parse_response(self, text: str) -> dict:
        """
        Parse ASSIGNMENTS/SKIPPED/WORKLOAD/NOTE from model output.
        ASSIGNMENTS format example: POT_001=crew_1, POT_003=crew_2
        """
        assignments: dict[str, str] = {}
        skipped_ids: list[str] = []
        utilization = ""
        note = ""

        for line in text.splitlines():
            cleaned = line.strip()
            upper_cleaned = cleaned.upper()

            if upper_cleaned.startswith("ASSIGNMENTS:"):
                payload = cleaned.split(":", 1)[1].strip()
                if payload and payload.lower() not in {"none", "n/a"}:
                    for pair in payload.split(","):
                        item = pair.strip()
                        if "=" not in item:
                            continue
                        pothole_id, crew_id = item.split("=", 1)
                        pothole_id = pothole_id.strip()
                        crew_id = crew_id.strip()
                        if pothole_id and crew_id:
                            assignments[pothole_id] = crew_id
            elif upper_cleaned.startswith("SKIPPED:"):
                payload = cleaned.split(":", 1)[1].strip()
                skipped_ids = [
                    item.strip()
                    for item in payload.split(",")
                    if item.strip() and item.strip().lower() not in {"none", "n/a"}
                ]
            elif upper_cleaned.startswith("WORKLOAD:"):
                utilization = cleaned.split(":", 1)[1].strip()
            elif upper_cleaned.startswith("NOTE:"):
                note = cleaned.split(":", 1)[1].strip()

        return {
            "assignments": assignments,
            "skipped_ids": skipped_ids,
            "utilization": utilization,
            "note": note,
        }
