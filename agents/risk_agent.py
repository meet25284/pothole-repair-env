"""
risk_agent.py - Risk Agent for CivicMind multi-agent system.
"""

from __future__ import annotations

import os
import textwrap

from openai import OpenAI

from models import Observation, WeatherWindow
from tools.risk_tools import calc_risk_score, flag_critical, get_weather_forecast


SYSTEM_PROMPT = """You are the Risk Agent for a city road repair team.
Your job is to assess weather and danger before any crew is dispatched.
Never recommend dispatching on rainy days unless it is a life-threatening emergency.
Respond in this exact format:
WEATHER: <SAFE or RISKY>
CLEARED: <comma separated pothole_ids safe to work on today>
BLOCKED: <comma separated pothole_ids blocked due to weather or risk>
EMERGENCY: <pothole_ids that must be fixed regardless of weather, or NONE>
ADVICE: <one sentence of advice for today>"""


class RiskAgent:
    def __init__(self, client: OpenAI):
        self.client = client
        self.model = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")

    def assess(self, obs: Observation, candidate_ids: list[str]) -> dict:
        try:
            # Touch import as requested and keep type intent explicit.
            weather_schedule: list[WeatherWindow] = [obs.weather]

            forecast = get_weather_forecast(
                obs.day,
                weather_schedule=weather_schedule,
                days_ahead=1,
            )
            critical = flag_critical(obs.potholes)
            risk_items = [
                calc_risk_score(p_id, obs.potholes, obs.weather.is_raining)
                for p_id in candidate_ids
            ]

            risk_lines = []
            for p_id, risk in zip(candidate_ids, risk_items):
                if "error" in risk:
                    risk_lines.append(f"- {p_id}: not found")
                    continue
                risk_lines.append(
                    (
                        f"- {p_id}: score={risk['risk_score']} | "
                        f"level={risk['risk_level']} | "
                        f"dispatch_recommended={risk['dispatch_recommended']}"
                    )
                )

            user_prompt = textwrap.dedent(
                f"""
                Day: {obs.day}
                Today's weather:
                - condition: {obs.weather.condition}
                - is_raining: {obs.weather.is_raining}
                - temperature: {obs.weather.temperature}

                Forecast summary:
                - safe_days_ahead: {forecast["safe_days_ahead"]}
                - rain_days_ahead: {forecast["rain_days_ahead"]}
                - next_safe_day: {forecast["next_safe_day"]}
                - recommendation: {forecast["recommendation"]}

                Candidate pothole risk:
                {chr(10).join(risk_lines) if risk_lines else "- none"}

                Emergency flagged potholes:
                - emergency_ids: {critical["emergency_ids"]}
                - message: {critical["message"]}

                What is safe to dispatch today?
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
            parsed = self._parse_response(raw_response, candidate_ids, obs.weather.is_raining)

            return {
                "weather_safe": parsed["weather_safe"],
                "cleared_ids": parsed["cleared_ids"],
                "blocked_ids": parsed["blocked_ids"],
                "emergency_ids": parsed["emergency_ids"],
                "advice": parsed["advice"],
                "raw_response": raw_response,
            }
        except Exception:
            if obs.weather.is_raining:
                return {
                    "weather_safe": False,
                    "cleared_ids": [],
                    "blocked_ids": candidate_ids,
                    "emergency_ids": [],
                    "advice": "Rainy conditions detected; block dispatch unless emergency.",
                    "raw_response": "",
                }
            return {
                "weather_safe": True,
                "cleared_ids": candidate_ids,
                "blocked_ids": [],
                "emergency_ids": [],
                "advice": "Weather is clear; dispatch can proceed carefully.",
                "raw_response": "",
            }

    def _parse_response(
        self, text: str, candidate_ids: list[str], is_raining: bool
    ) -> dict:
        """
        Parse WEATHER/CLEARED/BLOCKED/EMERGENCY/ADVICE lines with safe fallback.
        """
        weather_safe: bool | None = None
        cleared_ids: list[str] = []
        blocked_ids: list[str] = []
        emergency_ids: list[str] = []
        advice = ""

        for line in text.splitlines():
            cleaned = line.strip()
            upper_cleaned = cleaned.upper()

            if upper_cleaned.startswith("WEATHER:"):
                payload = cleaned.split(":", 1)[1].strip().upper()
                weather_safe = payload == "SAFE"
            elif upper_cleaned.startswith("CLEARED:"):
                payload = cleaned.split(":", 1)[1].strip()
                cleared_ids = [
                    item.strip()
                    for item in payload.split(",")
                    if item.strip() and item.strip().lower() not in {"none", "n/a"}
                ]
            elif upper_cleaned.startswith("BLOCKED:"):
                payload = cleaned.split(":", 1)[1].strip()
                blocked_ids = [
                    item.strip()
                    for item in payload.split(",")
                    if item.strip() and item.strip().lower() not in {"none", "n/a"}
                ]
            elif upper_cleaned.startswith("EMERGENCY:"):
                payload = cleaned.split(":", 1)[1].strip()
                if payload.upper() in {"NONE", "N/A", ""}:
                    emergency_ids = []
                else:
                    emergency_ids = [item.strip() for item in payload.split(",") if item.strip()]
            elif upper_cleaned.startswith("ADVICE:"):
                advice = cleaned.split(":", 1)[1].strip()

        candidate_set = set(candidate_ids)
        cleared_ids = [p_id for p_id in cleared_ids if p_id in candidate_set]
        blocked_ids = [p_id for p_id in blocked_ids if p_id in candidate_set]
        emergency_ids = [p_id for p_id in emergency_ids if p_id in candidate_set]

        if weather_safe is None:
            weather_safe = not is_raining

        if not cleared_ids and not blocked_ids:
            if is_raining:
                cleared_ids = []
                blocked_ids = list(candidate_ids)
            else:
                cleared_ids = list(candidate_ids)
                blocked_ids = []

        if not blocked_ids:
            blocked_ids = [p_id for p_id in candidate_ids if p_id not in set(cleared_ids)]
        if not cleared_ids and not is_raining:
            cleared_ids = [p_id for p_id in candidate_ids if p_id not in set(blocked_ids)]

        if not advice:
            advice = (
                "Rain present, avoid non-emergency dispatches."
                if is_raining
                else "Conditions are workable; prioritize high-risk repairs."
            )

        return {
            "weather_safe": weather_safe,
            "cleared_ids": cleared_ids,
            "blocked_ids": blocked_ids,
            "emergency_ids": emergency_ids,
            "advice": advice,
        }
