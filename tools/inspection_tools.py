"""
inspection_tools.py - Simulated inspection tool functions for Inspector Agent.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from models import PotholeReport, PotholeStatus
    from data_gen import BASE_LAT, BASE_LNG, generate_potholes
except ModuleNotFoundError:
    # Allow running this file directly from `tools/`.
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from models import PotholeReport, PotholeStatus
    from data_gen import BASE_LAT, BASE_LNG, generate_potholes


def inspect_pothole(pothole_id: str, potholes: list[PotholeReport]) -> dict:
    """
    Inspect a single pothole and return inspection details and recommendation.
    """
    pothole = next((p for p in potholes if p.id == pothole_id), None)
    if pothole is None:
        return {"error": "Pothole not found", "pothole_id": pothole_id}

    severity = pothole.severity
    road_type = str(pothole.road_type)

    if severity >= 4 and road_type == "highway":
        recommendation = "URGENT: dispatch immediately"
    elif severity >= 4 and road_type == "arterial":
        recommendation = "HIGH: dispatch within 2 days"
    elif severity <= 2:
        recommendation = "LOW: can defer safely"
    else:
        recommendation = "MEDIUM: schedule within week"

    return {
        "confirmed_severity": severity,
        "road_type": road_type,
        "daily_traffic": pothole.daily_traffic,
        "estimated_repair_cost": pothole.repair_cost,
        "recommendation": recommendation,
        "inspection_time_days": 1,
    }


def scan_area(zone_id: str, all_potholes: list[PotholeReport]) -> dict:
    """
    Scan a city zone and return aggregate pothole information.
    """
    valid_zones = {"north", "south", "east", "west", "central"}
    if zone_id not in valid_zones:
        raise ValueError(f"Invalid zone_id '{zone_id}'. Must be one of {sorted(valid_zones)}.")

    if zone_id == "north":
        potholes_in_zone = [p for p in all_potholes if p.lat > BASE_LAT]
    elif zone_id == "south":
        potholes_in_zone = [p for p in all_potholes if p.lat < BASE_LAT]
    elif zone_id == "east":
        potholes_in_zone = [p for p in all_potholes if p.lng > BASE_LNG]
    elif zone_id == "west":
        potholes_in_zone = [p for p in all_potholes if p.lng < BASE_LNG]
    else:
        potholes_in_zone = [
            p
            for p in all_potholes
            if abs(p.lat - BASE_LAT) <= 0.02 and abs(p.lng - BASE_LNG) <= 0.02
        ]

    pending_count = sum(1 for p in potholes_in_zone if p.status == PotholeStatus.PENDING)
    critical_count = sum(1 for p in potholes_in_zone if p.severity >= 4)

    pothole_ids = [p.id for p in potholes_in_zone]
    urgent_ids = [
        p.id for p in potholes_in_zone if p.severity >= 4 and p.status == PotholeStatus.PENDING
    ]

    return {
        "zone": zone_id,
        "total_found": len(potholes_in_zone),
        "pending_count": pending_count,
        "critical_count": critical_count,
        "pothole_ids": pothole_ids,
        "urgent_ids": urgent_ids,
    }


def get_severity_report(potholes: list[PotholeReport]) -> dict:
    """
    Build severity-level and urgency summary across all potholes.
    """
    by_severity = {level: 0 for level in range(1, 6)}
    for pothole in potholes:
        by_severity[pothole.severity] += 1

    critical_pending = sum(
        1 for p in potholes if p.severity >= 4 and p.status == PotholeStatus.PENDING
    )
    highway_critical = sum(
        1 for p in potholes if p.severity >= 4 and str(p.road_type) == "highway"
    )

    if potholes:
        most_urgent = max(potholes, key=lambda p: (p.severity, p.daily_traffic))
        most_urgent_id = most_urgent.id
    else:
        most_urgent_id = ""

    return {
        "total": len(potholes),
        "by_severity": by_severity,
        "critical_pending": critical_pending,
        "highway_critical": highway_critical,
        "most_urgent_id": most_urgent_id,
    }


if __name__ == "__main__":
    sample_potholes = generate_potholes(n=10, seed=42)

    print("=== inspect_pothole ===")
    print(inspect_pothole(sample_potholes[0].id, sample_potholes))
    print(inspect_pothole("POT_999", sample_potholes))

    print("\n=== scan_area ===")
    for zone in ["north", "south", "east", "west", "central"]:
        print(scan_area(zone, sample_potholes))

    print("\n=== get_severity_report ===")
    print(get_severity_report(sample_potholes))
