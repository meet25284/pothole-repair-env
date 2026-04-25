"""
scheduler_tools.py - Simulated scheduling tool functions for CivicMind pothole environment.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from models import PotholeReport, PotholeStatus
except ModuleNotFoundError:
    # Allow running this file directly from `tools/`.
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from models import PotholeReport, PotholeStatus


CREW_STATE = {
    "crew_1": {"status": "available", "assigned_to": None, "jobs_today": 0},
    "crew_2": {"status": "available", "assigned_to": None, "jobs_today": 0},
    "crew_3": {"status": "available", "assigned_to": None, "jobs_today": 0},
}


def _active_crew_ids(total_crews: int) -> list[str]:
    max_crews = max(0, min(total_crews, len(CREW_STATE)))
    return [f"crew_{i}" for i in range(1, max_crews + 1)]


def get_crew_status(total_crews: int) -> dict:
    """
    Return snapshot of crew availability for currently active crews.
    """
    crew_ids = _active_crew_ids(total_crews)
    crews = []

    for crew_id in crew_ids:
        state = CREW_STATE[crew_id]
        crews.append(
            {
                "crew_id": crew_id,
                "status": state["status"],
                "assigned_to": state["assigned_to"],
                "jobs_today": state["jobs_today"],
            }
        )

    available = sum(1 for c in crews if c["status"] == "available")
    busy = sum(1 for c in crews if c["status"] == "busy")

    return {
        "total_crews": len(crew_ids),
        "available": available,
        "busy": busy,
        "crews": crews,
        "can_dispatch": available > 0,
    }


def assign_crew(pothole_id: str, total_crews: int) -> dict:
    """
    Assign first available active crew to a pothole.
    """
    for crew_id in _active_crew_ids(total_crews):
        state = CREW_STATE[crew_id]
        if state["status"] == "available":
            state["status"] = "busy"
            state["assigned_to"] = pothole_id
            state["jobs_today"] += 1
            return {
                "assigned": True,
                "crew_id": crew_id,
                "pothole_id": pothole_id,
                "message": f"Crew {crew_id} dispatched to {pothole_id}",
            }

    return {"assigned": False, "reason": "No crews available"}


def release_crew(crew_id: str) -> dict:
    """
    Release a crew back to available state.
    """
    if crew_id not in CREW_STATE:
        return {
            "released": False,
            "crew_id": crew_id,
            "message": f"{crew_id} not found",
        }

    CREW_STATE[crew_id]["status"] = "available"
    CREW_STATE[crew_id]["assigned_to"] = None

    return {
        "released": True,
        "crew_id": crew_id,
        "message": f"{crew_id} is now available",
    }


def reset_crews(total_crews: int) -> dict:
    """
    Reset active crews for a new day.
    """
    crew_ids = _active_crew_ids(total_crews)
    for crew_id in crew_ids:
        CREW_STATE[crew_id]["status"] = "available"
        CREW_STATE[crew_id]["assigned_to"] = None
        CREW_STATE[crew_id]["jobs_today"] = 0

    return {"message": "All crews reset for new day", "crews_ready": len(crew_ids)}


def get_workload() -> dict:
    """
    Return workload summary across all crews in CREW_STATE.
    """
    crew_items = list(CREW_STATE.items())
    total_jobs_today = sum(state["jobs_today"] for _, state in crew_items)
    total_crews = len(crew_items)
    busy = sum(1 for _, state in crew_items if state["status"] == "busy")

    if crew_items:
        busiest_crew_id, busiest_state = max(crew_items, key=lambda item: item[1]["jobs_today"])
        busiest_crew = busiest_crew_id if busiest_state["jobs_today"] > 0 else None
    else:
        busiest_crew = None

    underutilized_crews = [
        crew_id for crew_id, state in crew_items if state["jobs_today"] == 0
    ]

    utilization_percent = (busy / total_crews * 100.0) if total_crews > 0 else 0.0

    return {
        "total_jobs_today": total_jobs_today,
        "busiest_crew": busiest_crew,
        "underutilized_crews": underutilized_crews,
        "utilization_percent": utilization_percent,
    }


if __name__ == "__main__":
    # Touch imports so they're part of module API usage expectations.
    _ = (PotholeReport, PotholeStatus)

    print("=== Initial Crew Status ===")
    print(get_crew_status(total_crews=3))

    print("\n=== Assign Crews ===")
    print(assign_crew("POT_001", total_crews=3))
    print(assign_crew("POT_002", total_crews=3))
    print(assign_crew("POT_003", total_crews=3))
    print(assign_crew("POT_004", total_crews=3))  # no crew available

    print("\n=== Crew Status After Assignments ===")
    print(get_crew_status(total_crews=3))

    print("\n=== Release Crew ===")
    print(release_crew("crew_2"))
    print(get_crew_status(total_crews=3))

    print("\n=== Workload ===")
    print(get_workload())

    print("\n=== Reset Crews ===")
    print(reset_crews(total_crews=3))
    print(get_crew_status(total_crews=3))
    print(get_workload())
