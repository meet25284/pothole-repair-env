"""
graders.py - Scoring functions for City Pothole Repair Scheduler
Each grader returns a float in [0.0, 1.0] — required by OpenEnv spec.
"""

from __future__ import annotations
from typing import List, Dict, Any
from models import PotholeReport, PotholeStatus


def _clamp(value: float) -> float:
    """Always clamp score to valid [0.0, 1.0] range."""
    return min(max(float(value), 0.0), 1.0)


# ─────────────────────────────────────────────
# TASK 1 — EASY
# ─────────────────────────────────────────────

def grader_easy(
    initial_potholes: List[PotholeReport],
    final_potholes: List[PotholeReport],
) -> float:
    """
    Easy task grader: Fix all severity 4 and 5 potholes.

    Score = number of sev-4/5 potholes fixed / total sev-4/5 potholes.
    Returns 1.0 if there are no critical potholes (trivially solved).
    """
    critical_ids = {
        p.id for p in initial_potholes
        if p.severity >= 4
    }

    if not critical_ids:
        return 1.0

    fixed_critical = sum(
        1 for p in final_potholes
        if p.id in critical_ids and p.status == PotholeStatus.FIXED
    )

    score = fixed_critical / len(critical_ids)
    return _clamp(score)


# ─────────────────────────────────────────────
# TASK 2 — MEDIUM
# ─────────────────────────────────────────────

def grader_medium(
    initial_potholes: List[PotholeReport],
    final_potholes: List[PotholeReport],
    initial_budget: float,
    final_budget: float,
) -> float:
    """
    Medium task grader: Maximize repairs within tight budget.

    Score = 0.6 * fix_rate + 0.4 * budget_efficiency
    fix_rate         = potholes fixed / total potholes
    budget_efficiency = budget remaining / initial_budget
                        (rewards spending wisely, not wastefully)
    """
    total = len(initial_potholes)
    if total == 0:
        return 1.0

    fixed = sum(
        1 for p in final_potholes
        if p.status == PotholeStatus.FIXED
    )

    fix_rate = fixed / total

    # Budget efficiency: rewarded for not overspending
    budget_used = initial_budget - max(final_budget, 0.0)
    if initial_budget > 0:
        budget_efficiency = 1.0 - (budget_used / initial_budget)
    else:
        budget_efficiency = 0.0

    score = 0.6 * fix_rate + 0.4 * budget_efficiency
    return _clamp(score)


# ─────────────────────────────────────────────
# TASK 3 — HARD
# ─────────────────────────────────────────────

def grader_hard(
    initial_potholes: List[PotholeReport],
    final_potholes: List[PotholeReport],
    initial_budget: float,
    final_budget: float,
    weather_days: List[Dict[str, Any]],
    actions_taken: List[Dict[str, Any]],
) -> float:
    """
    Hard task grader: Balance severity, budget, weather, traffic.

    Weighted score:
      40% → critical potholes (sev 4-5) fixed
      30% → budget efficiency
      20% → avoided dispatching on rainy days
      10% → high-traffic roads prioritized first
    """
    total = len(initial_potholes)
    if total == 0:
        return 1.0

    # ── 40%: critical potholes fixed ────────────────────
    critical_ids = {p.id for p in initial_potholes if p.severity >= 4}
    fixed_critical = sum(
        1 for p in final_potholes
        if p.id in critical_ids and p.status == PotholeStatus.FIXED
    )
    critical_score = (fixed_critical / len(critical_ids)) if critical_ids else 1.0

    # ── 30%: budget efficiency ───────────────────────────
    budget_used = initial_budget - max(final_budget, 0.0)
    budget_efficiency = 1.0 - (budget_used / initial_budget) if initial_budget > 0 else 0.0
    budget_efficiency = max(budget_efficiency, 0.0)

    # ── 20%: avoided rainy day dispatches ────────────────
    rainy_days = {w["day"] for w in weather_days if w.get("is_raining", False)}
    dispatch_actions = [a for a in actions_taken if a.get("action_type") == "dispatch"]

    if dispatch_actions:
        rainy_dispatches = sum(
            1 for a in dispatch_actions
            if a.get("day") in rainy_days
        )
        weather_score = 1.0 - (rainy_dispatches / len(dispatch_actions))
    else:
        weather_score = 1.0

    # ── 10%: high-traffic roads fixed early ──────────────
    # Build a lookup of pothole traffic by id
    traffic_map = {p.id: p.daily_traffic for p in initial_potholes}
    high_traffic_threshold = 20000

    high_traffic_ids = {
        p.id for p in initial_potholes
        if p.daily_traffic >= high_traffic_threshold
    }

    if high_traffic_ids:
        fixed_high_traffic = sum(
            1 for p in final_potholes
            if p.id in high_traffic_ids and p.status == PotholeStatus.FIXED
        )
        traffic_score = fixed_high_traffic / len(high_traffic_ids)
    else:
        traffic_score = 1.0

    # ── Final weighted score ─────────────────────────────
    score = (
        0.40 * critical_score +
        0.30 * budget_efficiency +
        0.20 * weather_score +
        0.10 * traffic_score
    )

    return _clamp(score)


if __name__ == "__main__":
    from data_gen import generate_potholes, generate_weather

    potholes = generate_potholes(n=15, seed=42)
    weather   = generate_weather(days=30, seed=42)
    weather_dicts = [w.model_dump() for w in weather]

    # Simulate some fixed potholes
    import copy
    final = copy.deepcopy(potholes)
    for p in final[:5]:
        p.status = PotholeStatus.FIXED

    print("Easy grader score  :", grader_easy(potholes, final))
    print("Medium grader score:", grader_medium(potholes, final, 50000, 30000))
    print("Hard grader score  :", grader_hard(
        potholes, final, 50000, 30000,
        weather_dicts,
        [{"action_type": "dispatch", "day": 2}]
    ))
