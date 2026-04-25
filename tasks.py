"""
tasks.py - Task definitions for City Pothole Repair Scheduler
Defines easy / medium / hard tasks with grader references.
"""

from __future__ import annotations
import random
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

    "generated_city": TaskConfig(
        name="generated_city",
        description="Procedurally generated city with extreme constraints",
        difficulty="extreme",
        max_days=60,
        initial_budget=15000.0,
        crew_count=1,
        num_potholes=35,
        success_threshold=0.4,
        seed=random.randint(1, 9999),  # different every time!
    ),
}

# Grader function mapping — used by inference.py to compute final score
GRADER_MAP = {
    "critical_repair":   grader_easy,
    "budget_optimizer":  grader_medium,
    "full_city_manager": grader_hard,
    "generated_city":    grader_hard,
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


def get_next_task(current_task_name: str, score: float) -> str:
    """
    Auto-escalate task difficulty based on score.
    Returns next task name.

    Rules:
    critical_repair (easy):   score >= 0.8 -> budget_optimizer
    budget_optimizer (medium): score >= 0.7 -> full_city_manager
    full_city_manager (hard):  score >= 0.6 -> generated_city (new)
    Any task: score < threshold -> stay on same task
    """
    if current_task_name == "critical_repair" and score >= 0.8:
        return "budget_optimizer"
    if current_task_name == "budget_optimizer" and score >= 0.7:
        return "full_city_manager"
    if current_task_name == "full_city_manager" and score >= 0.6:
        return "generated_city"
    return current_task_name


def generate_dynamic_task(base_difficulty_multiplier: float = 1.5) -> TaskConfig:
    """
    Generate a completely new random city scenario.
    Called when agent beats full_city_manager.
    multiplier increases budget pressure and pothole count.
    """
    import random

    seed = random.randint(1000, 9999)
    return TaskConfig(
        name=f"dynamic_city_{seed}",
        description=f"Auto-generated extreme scenario {seed}",
        difficulty="extreme",
        max_days=int(45 * base_difficulty_multiplier),
        initial_budget=max(10000, 30000 / base_difficulty_multiplier),
        crew_count=1,
        num_potholes=int(25 * base_difficulty_multiplier),
        success_threshold=0.35,
        seed=seed,
    )


if __name__ == "__main__":
    for name, cfg in TASK_CONFIGS.items():
        print(f"\n{cfg.difficulty.upper()} — {cfg.name}")
        print(f"  {cfg.description}")
        print(f"  Days: {cfg.max_days} | Budget: ₹{cfg.initial_budget:,.0f} | Crews: {cfg.crew_count} | Potholes: {cfg.num_potholes}")
