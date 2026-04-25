"""
risk_tools.py - Simulated risk and weather tools for CivicMind pothole environment.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

try:
    from models import PotholeReport, PotholeStatus, WeatherWindow
except ModuleNotFoundError:
    # Allow running this file directly from `tools/`.
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from models import PotholeReport, PotholeStatus, WeatherWindow


def get_weather_forecast(
    current_day: int, weather_schedule: list[WeatherWindow], days_ahead: int = 7
) -> dict:
    """
    Return weather outlook from current day through current_day + days_ahead.
    """
    end_day = current_day + days_ahead
    window = [w for w in weather_schedule if current_day <= w.day <= end_day]
    window = sorted(window, key=lambda w: w.day)

    forecast = [
        {
            "day": w.day,
            "condition": w.condition,
            "is_raining": w.is_raining,
            "temperature": w.temperature,
        }
        for w in window
    ]

    safe_days_ahead = [w.day for w in window if not w.is_raining]
    rain_days_ahead = [w.day for w in window if w.is_raining]

    next_safe_day = next((w.day for w in window if not w.is_raining), None)
    today_weather = next((w for w in window if w.day == current_day), None)
    tomorrow_weather = next((w for w in window if w.day == current_day + 1), None)

    consecutive_rain_days = 0
    for w in window:
        if w.day < current_day:
            continue
        if w.is_raining:
            consecutive_rain_days += 1
        else:
            break

    if today_weather is not None and not today_weather.is_raining:
        recommendation = "SAFE: Good day to dispatch crews"
    elif (
        today_weather is not None
        and today_weather.is_raining
        and tomorrow_weather is not None
        and not tomorrow_weather.is_raining
    ):
        recommendation = "WAIT: Dispatch tomorrow"
    elif consecutive_rain_days >= 3:
        recommendation = "PLAN: Consider indoor prep work"
    else:
        recommendation = "CAUTION: Monitor forecast"

    return {
        "current_day": current_day,
        "forecast": forecast,
        "safe_days_ahead": safe_days_ahead,
        "rain_days_ahead": rain_days_ahead,
        "next_safe_day": next_safe_day,
        "recommendation": recommendation,
    }


def calc_risk_score(pothole_id: str, potholes: list[PotholeReport], is_raining: bool) -> dict:
    """
    Calculate a pothole risk score using severity, traffic class, and weather.
    """
    pothole = next((p for p in potholes if p.id == pothole_id), None)
    if pothole is None:
        return {"error": "Pothole not found", "pothole_id": pothole_id}

    base_risk = pothole.severity * 10
    road_type = str(pothole.road_type)

    if road_type == "highway":
        risk_score = base_risk * 2.0
    elif road_type == "arterial":
        risk_score = base_risk * 1.5
    else:
        risk_score = base_risk * 1.0

    if is_raining:
        risk_score += 20

    if risk_score >= 80:
        risk_level = "CRITICAL"
    elif risk_score >= 50:
        risk_level = "HIGH"
    elif risk_score >= 30:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    dispatch_recommended = risk_level in {"CRITICAL", "HIGH"} and not is_raining
    reason = (
        f"Severity {pothole.severity} on {road_type} road gives score {risk_score:.1f}; "
        f"weather={'rain' if is_raining else 'clear'} so dispatch="
        f"{'recommended' if dispatch_recommended else 'not recommended'}."
    )

    return {
        "pothole_id": pothole_id,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "is_raining": is_raining,
        "dispatch_recommended": dispatch_recommended,
        "reason": reason,
    }


def flag_critical(potholes: list[PotholeReport]) -> dict:
    """
    Flag pending critical potholes and identify emergency-level cases.
    """
    critical_potholes = [
        p for p in potholes if p.severity >= 4 and p.status == PotholeStatus.PENDING
    ]

    flagged = []
    emergency_ids: list[str] = []
    for pothole in critical_potholes:
        risk = calc_risk_score(pothole.id, potholes, is_raining=False)
        flagged.append(
            {
                "pothole_id": pothole.id,
                "severity": pothole.severity,
                "road_type": str(pothole.road_type),
                "daily_traffic": pothole.daily_traffic,
                "risk_level": risk.get("risk_level", "UNKNOWN"),
            }
        )
        if pothole.severity == 5 and str(pothole.road_type) == "highway":
            emergency_ids.append(pothole.id)

    if emergency_ids:
        message = "EMERGENCY: Immediate dispatch required"
    elif len(critical_potholes) > 0:
        message = "WARNING: Critical potholes need attention"
    else:
        message = "OK: No critical potholes pending"

    return {
        "critical_count": len(critical_potholes),
        "flagged": flagged,
        "emergency_ids": emergency_ids,
        "message": message,
    }


def get_accident_history(zone_id: str) -> dict:
    """
    Simulate accident trends by zone with deterministic randomness per zone id.
    """
    seed = sum((idx + 1) * ord(ch) for idx, ch in enumerate(zone_id.lower()))
    rng = random.Random(seed)

    accidents_last_30_days = rng.randint(0, 15)
    most_common_road_type = rng.choice(["highway", "arterial", "residential"])

    if accidents_last_30_days >= 11:
        severity_of_accidents = "HIGH"
        recommendation = "Immediate safety patrol and accelerated repairs recommended"
    elif accidents_last_30_days >= 6:
        severity_of_accidents = "MEDIUM"
        recommendation = "Increase monitoring and prioritize high-traffic potholes"
    else:
        severity_of_accidents = "LOW"
        recommendation = "Maintain routine inspections and preventive scheduling"

    return {
        "zone": zone_id,
        "accidents_last_30_days": accidents_last_30_days,
        "severity_of_accidents": severity_of_accidents,
        "most_common_road_type": most_common_road_type,
        "recommendation": recommendation,
    }


if __name__ == "__main__":
    from data_gen import generate_potholes, generate_weather

    sample_potholes = generate_potholes(n=10, seed=42)
    sample_weather = generate_weather(days=10, seed=42)

    print("=== get_weather_forecast ===")
    print(get_weather_forecast(current_day=1, weather_schedule=sample_weather, days_ahead=7))

    print("\n=== calc_risk_score ===")
    first_id = sample_potholes[0].id
    print(calc_risk_score(first_id, sample_potholes, is_raining=False))
    print(calc_risk_score(first_id, sample_potholes, is_raining=True))
    print(calc_risk_score("POT_999", sample_potholes, is_raining=False))

    print("\n=== flag_critical ===")
    print(flag_critical(sample_potholes))

    print("\n=== get_accident_history ===")
    for zone in ["north", "south", "east", "west", "central"]:
        print(get_accident_history(zone))
