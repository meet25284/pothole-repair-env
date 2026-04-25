"""
budget_tools.py - Simulated budgeting tool functions for CivicMind pothole environment.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from models import PotholeReport
except ModuleNotFoundError:
    # Allow running this file directly from `tools/`.
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from models import PotholeReport


def check_budget(budget_remaining: float, initial_budget: float) -> dict:
    """
    Return current budget health metrics and status.
    """
    remaining = round(budget_remaining, 2)
    initial = float(initial_budget)
    used = initial - remaining

    if initial > 0:
        used_percent = round((used / initial) * 100, 1)
    else:
        used_percent = 0.0

    if remaining > 0.50 * initial:
        status = "HEALTHY"
    elif remaining > 0.25 * initial:
        status = "CAUTION"
    elif remaining > 0.10 * initial:
        status = "LOW"
    else:
        status = "CRITICAL"

    return {
        "remaining": remaining,
        "initial": initial,
        "used": used,
        "used_percent": used_percent,
        "status": status,
        "can_still_spend": remaining > 0,
    }


def estimate_cost(
    pothole_id: str, potholes: list[PotholeReport], budget_remaining: float = 0.0
) -> dict:
    """
    Estimate final repair cost with urgency-based multiplier.
    """
    pothole = next((p for p in potholes if p.id == pothole_id), None)
    if pothole is None:
        return {"error": "not found"}

    severity = pothole.severity
    base_cost = float(pothole.repair_cost)

    if severity >= 4:
        urgency_multiplier = 1.0
    elif severity == 3:
        urgency_multiplier = 0.9
    else:
        urgency_multiplier = 0.8

    final_cost = round(base_cost * urgency_multiplier, 2)
    budget_after_repair = budget_remaining - final_cost

    return {
        "pothole_id": pothole_id,
        "base_cost": base_cost,
        "urgency_multiplier": urgency_multiplier,
        "final_cost": final_cost,
        "affordable": final_cost <= budget_remaining,
        "budget_remaining": budget_remaining,
        "budget_after_repair": budget_after_repair,
    }


def approve_spend(amount: float, budget_remaining: float, reason: str = "") -> dict:
    """
    Approve or reject a spend request based on remaining budget and amount validity.
    """
    approved = amount <= budget_remaining and amount > 0

    if amount > budget_remaining:
        rejection_reason = "Insufficient funds"
    elif amount <= 0:
        rejection_reason = "Invalid amount"
    else:
        rejection_reason = None

    budget_after = budget_remaining - amount if approved else budget_remaining

    return {
        "approved": approved,
        "amount_requested": amount,
        "budget_before": budget_remaining,
        "budget_after": budget_after,
        "reason": reason,
        "rejection_reason": rejection_reason,
    }


def get_spending_history(actions_taken: list[dict]) -> dict:
    """
    Summarize spending and action patterns from action history.
    """
    total_dispatches = sum(1 for a in actions_taken if a.get("action_type") == "dispatch")
    total_deferred = sum(1 for a in actions_taken if a.get("action_type") == "defer")
    total_low_priority = sum(
        1 for a in actions_taken if a.get("action_type") == "mark_low_priority"
    )

    dispatch_costs = [
        float(a.get("cost", 0.0)) for a in actions_taken if a.get("action_type") == "dispatch"
    ]
    total_spent = sum(dispatch_costs)
    avg_cost_per_repair = total_spent / total_dispatches if total_dispatches > 0 else 0.0

    spending_by_day: dict[int, float] = {}
    for action in actions_taken:
        if action.get("action_type") != "dispatch":
            continue
        day = action.get("day")
        if day is None:
            continue
        spending_by_day[day] = spending_by_day.get(day, 0.0) + float(action.get("cost", 0.0))

    most_expensive_day = (
        max(spending_by_day, key=spending_by_day.get) if spending_by_day else None
    )

    return {
        "total_dispatches": total_dispatches,
        "total_deferred": total_deferred,
        "total_low_priority": total_low_priority,
        "total_spent": total_spent,
        "avg_cost_per_repair": avg_cost_per_repair,
        "most_expensive_day": most_expensive_day,
    }


if __name__ == "__main__":
    from data_gen import generate_potholes

    sample_potholes = generate_potholes(n=10, seed=42)
    sample_budget = 50000.0
    initial_budget = 100000.0

    print("=== check_budget ===")
    print(check_budget(sample_budget, initial_budget))

    print("\n=== estimate_cost ===")
    print(estimate_cost(sample_potholes[0].id, sample_potholes, budget_remaining=sample_budget))
    print(estimate_cost("POT_999", sample_potholes, budget_remaining=sample_budget))

    print("\n=== approve_spend ===")
    print(approve_spend(12000.0, sample_budget, reason="Repair critical arterial pothole"))
    print(approve_spend(0.0, sample_budget, reason="Invalid test"))
    print(approve_spend(999999.0, sample_budget, reason="Over-budget test"))

    print("\n=== get_spending_history ===")
    sample_actions = [
        {"action_type": "dispatch", "pothole_id": "POT_001", "day": 1, "cost": 8000.0},
        {"action_type": "defer", "pothole_id": "POT_002", "day": 1},
        {"action_type": "dispatch", "pothole_id": "POT_003", "day": 2, "cost": 12000.0},
        {"action_type": "mark_low_priority", "pothole_id": "POT_004", "day": 2},
        {"action_type": "dispatch", "pothole_id": "POT_005", "day": 2, "cost": 15000.0},
    ]
    print(get_spending_history(sample_actions))
