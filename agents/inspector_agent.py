"""
inspector_agent.py - Inspector Agent for CivicMind multi-agent system.
"""

from __future__ import annotations

import os
import textwrap

from openai import OpenAI

from models import Observation, PotholeStatus
from tools.inspection_tools import get_severity_report, inspect_pothole, scan_area


SYSTEM_PROMPT = """You are the Inspector Agent for a city road repair team.
Your job is to survey potholes and report what needs urgent attention.
You have 3 tools: inspect_pothole, scan_area, get_severity_report.
Always inspect the top 3 most suspicious potholes each day.
Respond in this exact format:
INSPECTED: <pothole_id>=<severity>, <pothole_id>=<severity>
URGENT: <comma separated pothole_ids that need immediate dispatch>
DEFER: <comma separated pothole_ids that can wait>
SUMMARY: <one sentence about today's findings>"""


class InspectorAgent:
    def __init__(self, client: OpenAI):
        self.client = client
        self.model = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")

    def inspect(self, obs: Observation) -> dict:
        try:
            severity_report = get_severity_report(obs.potholes)
            pending_potholes = [p for p in obs.potholes if p.status == PotholeStatus.PENDING]
            top_three = sorted(
                pending_potholes,
                key=lambda p: (p.severity, p.daily_traffic),
                reverse=True,
            )[:3]

            inspections = [inspect_pothole(p.id, obs.potholes) for p in top_three]
            central_scan = scan_area("central", obs.potholes)

            inspection_lines = []
            for pothole, inspected in zip(top_three, inspections):
                inspection_lines.append(
                    (
                        f"- {pothole.id}: sev={inspected.get('confirmed_severity')} | "
                        f"road={inspected.get('road_type')} | "
                        f"traffic={inspected.get('daily_traffic')} | "
                        f"cost={inspected.get('estimated_repair_cost')} | "
                        f"rec={inspected.get('recommendation')}"
                    )
                )

            user_prompt = textwrap.dedent(
                f"""
                Day: {obs.day}
                Budget remaining: {obs.budget_remaining}
                Pending potholes: {severity_report["total"]}

                Severity report:
                - by_severity: {severity_report["by_severity"]}
                - critical_pending: {severity_report["critical_pending"]}
                - highway_critical: {severity_report["highway_critical"]}
                - most_urgent_id: {severity_report["most_urgent_id"]}

                Central scan:
                - total_found: {central_scan["total_found"]}
                - pending_count: {central_scan["pending_count"]}
                - critical_count: {central_scan["critical_count"]}
                - urgent_ids: {central_scan["urgent_ids"]}

                Top 3 inspections:
                {chr(10).join(inspection_lines) if inspection_lines else "- No pending potholes"}

                Based on this, which potholes are urgent and which to defer?
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
                "urgent_ids": parsed["urgent_ids"],
                "defer_ids": parsed["defer_ids"],
                "summary": parsed["summary"],
                "raw_response": raw_response,
            }
        except Exception:
            pending = [p for p in obs.potholes if p.status == PotholeStatus.PENDING]
            fallback_urgent = [pending[0].id] if pending else []
            return {
                "urgent_ids": fallback_urgent,
                "defer_ids": [],
                "summary": "Inspection failed",
                "raw_response": "",
            }

    def _parse_response(self, text: str) -> dict:
        """
        Parse strict-format LLM response into usable fields.
        """
        urgent_ids: list[str] = []
        defer_ids: list[str] = []
        summary = ""

        for line in text.splitlines():
            cleaned = line.strip()
            upper_cleaned = cleaned.upper()

            if upper_cleaned.startswith("URGENT:"):
                payload = cleaned.split(":", 1)[1].strip()
                urgent_ids = [
                    item.strip()
                    for item in payload.split(",")
                    if item.strip() and item.strip().lower() not in {"none", "n/a"}
                ]
            elif upper_cleaned.startswith("DEFER:"):
                payload = cleaned.split(":", 1)[1].strip()
                defer_ids = [
                    item.strip()
                    for item in payload.split(",")
                    if item.strip() and item.strip().lower() not in {"none", "n/a"}
                ]
            elif upper_cleaned.startswith("SUMMARY:"):
                summary = cleaned.split(":", 1)[1].strip()

        return {"urgent_ids": urgent_ids, "defer_ids": defer_ids, "summary": summary}
