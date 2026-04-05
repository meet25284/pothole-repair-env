"""
models.py - Pydantic typed models for City Pothole Repair Scheduler
All OpenEnv observation, action, and reward types are defined here.
"""

from __future__ import annotations
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, ConfigDict, Field
from enum import Enum


# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────

class RoadType(str, Enum):
    HIGHWAY = "highway"
    ARTERIAL = "arterial"
    RESIDENTIAL = "residential"


class PotholeStatus(str, Enum):
    PENDING = "pending"
    FIXED = "fixed"
    DEFERRED = "deferred"
    LOW_PRIORITY = "low_priority"


class ActionType(str, Enum):
    DISPATCH = "dispatch"
    DEFER = "defer"
    MARK_LOW_PRIORITY = "mark_low_priority"


# ─────────────────────────────────────────────
# Core Data Models
# ─────────────────────────────────────────────

class PotholeReport(BaseModel):
    """Represents a single pothole report in the city."""
    id: str
    lat: float
    lng: float
    severity: int = Field(..., ge=1, le=5, description="Severity from 1 (minor) to 5 (critical)")
    road_type: RoadType
    daily_traffic: int = Field(..., ge=0, description="Estimated daily vehicle count")
    repair_cost: float = Field(..., ge=0.0, description="Estimated cost in INR")
    status: PotholeStatus = PotholeStatus.PENDING
    days_pending: int = Field(default=0, description="How many days this has been pending")

    model_config = ConfigDict(use_enum_values=True)


class WeatherWindow(BaseModel):
    """Weather info for a single day."""
    day: int
    is_raining: bool = False
    temperature: float = Field(default=28.0, description="Temperature in Celsius")
    condition: str = Field(default="clear", description="clear / rainy / cloudy")


class Observation(BaseModel):
    """Full environment observation returned by reset() and step()."""
    potholes: List[PotholeReport]
    budget_remaining: float
    initial_budget: float
    crews_available: int
    total_crews: int
    day: int
    max_days: int
    weather: WeatherWindow
    total_fixed: int = 0
    total_pending: int = 0
    episode_reward_so_far: float = 0.0


class Action(BaseModel):
    """Action the agent takes each step."""
    action_type: ActionType
    pothole_id: str
    defer_days: int = Field(default=1, ge=1, le=7, description="Days to defer (only for defer action)")

    model_config = ConfigDict(use_enum_values=True)


class StepResult(BaseModel):
    """Result returned by env.step()."""
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)
    last_action_error: Optional[str] = None


class TaskConfig(BaseModel):
    """Configuration for each task."""
    name: str
    description: str
    difficulty: str
    max_days: int
    initial_budget: float
    crew_count: int
    num_potholes: int
    success_threshold: float
    seed: int = 42
