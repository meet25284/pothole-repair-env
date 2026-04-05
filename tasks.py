"""
tasks.py - Task definitions for City Pothole Repair Scheduler
Defines easy / medium / hard tasks with grader references.
"""

from __future__ import annotations
from typing import Dict
from models import TaskConfig
from graders import grader_easy, grader_medium, grader_hard


# ─────────────────────────────────────────────
# Task Registry
# ─────────────────────────────────────────────

TASK_CONFIGS: Dict[str, TaskConfig] = {

    "critical_repair": TaskConfig(
        name="critical_repair",
        description=(
            "Fix all severity 4 and 5 potholes in Ahmedabad within 30 days. "
            "You have 3 crews and a budget of ₹50,000. "
            "Prioritize highways and arterial roads with heavy traffic first."
        ),
        difficulty="easy",
        max_days=30,
        initial_budget=50000.0,
        crew_count=3,
        num_potholes=15,
        success_threshold=0.8,
        seed=42,
    ),

    "budget_optimizer": TaskConfig(
        name="budget_optimizer",
        description=(
            "Maximize the number of potholes fixed within a tight budget of ₹20,000 "
            "and only 2 crews over 30 days. You cannot fix everything — "
            "choose wisely based on severity and traffic impact."
        ),
        difficulty="medium",
        max_days=30,
        initial_budget=20000.0,
        crew_count=2,
        num_potholes=25,
        success_threshold=0.6,
        seed=99,
    ),

    "full_city_manager": TaskConfig(
        name="full_city_manager",
        description=(
            "Manage city-wide pothole repairs over 45 days balancing: "
            "severity urgency, tight budget of ₹30,000, only 2 crews, "
            "and weather windows (no dispatching on rainy days). "
            "This task requires multi-factor optimization."
        ),
        difficulty="hard",
        max_days=45,
        initial_budget=30000.0,
        crew_count=2,
        num_potholes=30,
        success_threshold=0.5,
        seed=77,
    ),
}

# Grader function mapping — used by inference.py to compute final score
GRADER_MAP = {
    "critical_repair":   grader_easy,
    "budget_optimizer":  grader_medium,
    "full_city_manager": grader_hard,
}


def get_task(task_name: str) -> TaskConfig:
    """Return task config by name. Raises ValueError if not found."""
    if task_name not in TASK_CONFIGS:
        valid = list(TASK_CONFIGS.keys())
        raise ValueError(f"Unknown task '{task_name}'. Valid tasks: {valid}")
    return TASK_CONFIGS[task_name]


def list_tasks() -> list:
    """Return all task configs as a list (for /tasks endpoint)."""
    return [
        {
            "name": cfg.name,
            "description": cfg.description,
            "difficulty": cfg.difficulty,
            "max_days": cfg.max_days,
            "initial_budget": cfg.initial_budget,
            "crew_count": cfg.crew_count,
            "num_potholes": cfg.num_potholes,
            "success_threshold": cfg.success_threshold,
        }
        for cfg in TASK_CONFIGS.values()
    ]


if __name__ == "__main__":
    for name, cfg in TASK_CONFIGS.items():
        print(f"\n{cfg.difficulty.upper()} — {cfg.name}")
        print(f"  {cfg.description}")
        print(f"  Days: {cfg.max_days} | Budget: ₹{cfg.initial_budget:,.0f} | Crews: {cfg.crew_count} | Potholes: {cfg.num_potholes}")
