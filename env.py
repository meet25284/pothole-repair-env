"""
env.py - Main OpenEnv environment: City Pothole Repair Scheduler
Implements reset() / step() / state() as per OpenEnv spec.
"""

from __future__ import annotations
import copy
from typing import List, Optional, Dict, Any

try:
    from openenv.env import Env
except ModuleNotFoundError:
    # Fallback for runtime environments where openenv package is unavailable.
    class Env:
        def __init__(
            self,
            name: str = "Env",
            state_space: Optional[dict] = None,
            action_space: Optional[dict] = None,
            episode_max_length: int = 300,
        ):
            self.name = name
            self.state_space = state_space
            self.action_space = action_space
            self.episode_max_length = episode_max_length

from models import (
    PotholeReport, PotholeStatus, WeatherWindow,
    Observation, Action, ActionType, StepResult, TaskConfig,
)
from data_gen import generate_potholes, generate_weather, get_traffic_factor
from tasks import get_task, GRADER_MAP


class PotholeRepairEnv(Env):
    """
    City Pothole Repair Scheduler — OpenEnv compliant environment.

    The agent must schedule road repair crews to fix potholes across
    a city, balancing severity, budget, crew limits, and weather windows.

    Methods:
        reset(task_name)  → Observation
        step(action)      → StepResult
        state()           → Observation
        close()           → None
    """

    def __init__(self, task_name: str = "critical_repair"):
        self.task_name = task_name
        self.task: TaskConfig = get_task(task_name)

        state_space = {
            "potholes": "list of PotholeReport objects",
            "budget_remaining": "float",
            "initial_budget": "float",
            "crews_available": "int",
            "day": "int",
            "max_days": "int",
            "weather": "WeatherWindow object",
            "total_fixed": "int",
            "total_pending": "int",
        }
        action_space = {
            "action_type": "dispatch | defer | mark_low_priority",
            "pothole_id": "str — e.g. POT_001",
            "defer_days": "int 1-7 (only for defer action)",
        }
        super().__init__(
            name="CivicMind-PotholeRepairEnv",
            state_space=state_space,
            action_space=action_space,
            episode_max_length=self.task.max_days,
        )

        # State — initialized properly on reset()
        self._potholes: List[PotholeReport] = []
        self._initial_potholes: List[PotholeReport] = []
        self._weather_schedule: List[WeatherWindow] = []
        self._actions_taken: List[Dict[str, Any]] = []
        self._inspected_ids: set[str] = set()

        self._budget: float = 0.0
        self._day: int = 1
        self._done: bool = False
        self._episode_reward: float = 0.0
        self._initialized: bool = False

    def reward(self, state, action) -> float:
        """
        OpenEnv reward wrapper.
        state = current pothole (PotholeReport), action = Action object.
        """
        return self._compute_reward(state)

    def print_info(self):
        print(f"Environment : {self.name}")
        print(f"Task        : {self.task_name}")
        print(f"State space : {list(self.state_space.keys())}")
        print(f"Action space: {list(self.action_space.keys())}")
        print(f"Max days    : {self.task.max_days}")
        print(f"Budget      : ₹{self.task.initial_budget:,.0f}")
        print(f"Crews       : {self.task.crew_count}")

    # ─────────────────────────────────────────
    # reset()
    # ─────────────────────────────────────────

    def reset(self, task_name: Optional[str] = None) -> Observation:
        """
        Reset environment to a fresh episode.
        Returns initial Observation.
        """
        if task_name and task_name != self.task_name:
            self.task_name = task_name
            self.task = get_task(task_name)

        self._potholes = generate_potholes(
            n=self.task.num_potholes,
            seed=self.task.seed,
        )
        self._initial_potholes = copy.deepcopy(self._potholes)
        self._weather_schedule = generate_weather(
            days=self.task.max_days,
            seed=self.task.seed,
        )
        self._actions_taken = []
        self._inspected_ids = set()
        self._budget = self.task.initial_budget
        self._day = 1
        self._done = False
        self._episode_reward = 0.0
        self._initialized = True

        return self._build_observation()

    # ─────────────────────────────────────────
    # step()
    # ─────────────────────────────────────────

    def step(self, action: Action) -> StepResult:
        """
        Execute one action and return StepResult.
        Advances the simulation by 1 day.
        """
        if not self._initialized:
            raise RuntimeError("Call reset() before step()")

        if self._done:
            obs = self._build_observation()
            return StepResult(
                observation=obs,
                reward=0.0,
                done=True,
                info={"message": "Episode already done. Call reset()."},
                last_action_error="Episode already done",
            )

        # Find pothole
        pothole = self._get_pothole(action.pothole_id)
        error_msg: Optional[str] = None

        if pothole is None:
            error_msg = f"Pothole {action.pothole_id} not found"
            reward = -0.1
        elif pothole.status == PotholeStatus.FIXED:
            error_msg = f"Pothole {action.pothole_id} already fixed"
            reward = -0.05
        else:
            reward, error_msg = self._execute_action(action, pothole)

        # Record action for grader
        current_weather = self._get_today_weather()
        self._actions_taken.append({
            "action_type": action.action_type,
            "pothole_id": action.pothole_id,
            "day": self._day,
            "is_rainy_day": current_weather.is_raining if current_weather else False,
        })

        self._episode_reward += reward
        self._day += 1

        # Check episode done
        done = self._check_done()
        self._done = done

        obs = self._build_observation()
        return StepResult(
            observation=obs,
            reward=round(reward, 4),
            done=done,
            info={
                "day": self._day - 1,
                "budget_remaining": self._budget,
                "total_fixed": sum(1 for p in self._potholes if p.status == PotholeStatus.FIXED),
                "episode_reward": round(self._episode_reward, 4),
            },
            last_action_error=error_msg,
        )

    # ─────────────────────────────────────────
    # state()
    # ─────────────────────────────────────────

    def state(self) -> Observation:
        """Return current environment state without advancing."""
        if not self._initialized:
            raise RuntimeError("Call reset() before state()")
        return self._build_observation()

    # ─────────────────────────────────────────
    # close()
    # ─────────────────────────────────────────

    def close(self) -> None:
        """Clean up environment resources."""
        self._initialized = False

    # ─────────────────────────────────────────
    # get_final_score()
    # ─────────────────────────────────────────

    def get_final_score(self) -> float:
        """
        Compute the final task score using the appropriate grader.
        Returns float in [0.0, 1.0].
        """
        grader = GRADER_MAP.get(self.task_name)
        if grader is None:
            return 0.0

        weather_dicts = [w.model_dump() for w in self._weather_schedule]

        if self.task_name == "critical_repair":
            return grader(self._initial_potholes, self._potholes)

        elif self.task_name == "budget_optimizer":
            return grader(
                self._initial_potholes, self._potholes,
                self.task.initial_budget, self._budget,
            )

        elif self.task_name == "full_city_manager":
            return grader(
                self._initial_potholes, self._potholes,
                self.task.initial_budget, self._budget,
                weather_dicts, self._actions_taken,
            )

        return 0.0

    # ─────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────

    def _execute_action(self, action: Action, pothole: PotholeReport):
        """Execute a valid action on a valid pothole. Returns (reward, error)."""
        error = None
        current_weather = self._get_today_weather()
        is_raining = current_weather.is_raining if current_weather else False

        if action.action_type == ActionType.DISPATCH:
            # Check budget
            if self._budget < pothole.repair_cost:
                error = "Insufficient budget"
                return -0.3, error

            # Penalty for dispatching on rainy day
            rain_penalty = -0.15 if is_raining else 0.0

            # Fix the pothole
            pothole.status = PotholeStatus.FIXED
            self._budget -= pothole.repair_cost
            reward = self._compute_reward(pothole) + rain_penalty

        elif action.action_type == ActionType.DEFER:
            pothole.status = PotholeStatus.DEFERRED
            pothole.days_pending += action.defer_days

            # Penalty for deferring critical potholes on busy roads
            if pothole.severity >= 4 and pothole.road_type in ("highway", "arterial"):
                reward = -0.2
                error = "Warning: deferring critical pothole on busy road"
            else:
                reward = -0.02  # tiny penalty for deferring

        elif action.action_type == ActionType.MARK_LOW_PRIORITY:
            pothole.status = PotholeStatus.LOW_PRIORITY
            # Penalty if marking a high severity pothole as low priority
            if pothole.severity >= 4:
                reward = -0.25
                error = "Warning: marking critical pothole as low priority"
            else:
                reward = 0.01  # small reward for correctly triaging

        else:
            reward = -0.1
            error = f"Unknown action type: {action.action_type}"

        return round(reward, 4), error

    def _compute_reward(self, pothole: PotholeReport) -> float:
        """
        Compute reward for fixing a pothole.
        Higher severity + higher traffic = higher reward.
        Deduct fraction of repair cost from reward to encourage efficiency.
        """
        severity_factor = pothole.severity * 0.15       # 0.15 to 0.75
        traffic_factor  = get_traffic_factor(pothole.road_type)  # 1.0–2.0
        cost_penalty    = (pothole.repair_cost / self.task.initial_budget) * 0.3

        reward = (severity_factor * traffic_factor) - cost_penalty
        return max(min(reward, 1.0), -1.0)

    def _get_pothole(self, pothole_id: str) -> Optional[PotholeReport]:
        for p in self._potholes:
            if p.id == pothole_id:
                return p
        return None

    def _get_today_weather(self) -> Optional[WeatherWindow]:
        idx = self._day - 1
        if 0 <= idx < len(self._weather_schedule):
            return self._weather_schedule[idx]
        return None

    def _check_done(self) -> bool:
        """Episode ends if: day limit reached, budget exhausted, or all resolved."""
        if self._day > self.task.max_days:
            return True
        if self._budget <= 0:
            return True
        all_resolved = all(
            p.status != PotholeStatus.PENDING
            for p in self._potholes
        )
        if all_resolved:
            return True
        return False

    def reveal_severity(self, pothole_id: str) -> dict:
        """
        Reveal real severity for a pothole and mark it as inspected.
        """
        pothole = self._get_pothole(pothole_id)
        if pothole is None:
            return {"error": "not found", "revealed": False}

        self._inspected_ids.add(pothole_id)
        return {
            "pothole_id": pothole.id,
            "real_severity": pothole.severity,
            "revealed": True,
        }

    def _build_observation(self) -> Observation:
        weather = self._get_today_weather() or WeatherWindow(day=self._day)
        pending = sum(1 for p in self._potholes if p.status == PotholeStatus.PENDING)
        fixed   = sum(1 for p in self._potholes if p.status == PotholeStatus.FIXED)
        visible_potholes = copy.deepcopy(self._potholes)

        for pothole in visible_potholes:
            if pothole.status == "pending" and pothole.id not in self._inspected_ids:
                pothole.severity = 0

        return Observation(
            potholes=visible_potholes,
            budget_remaining=round(self._budget, 2),
            initial_budget=self.task.initial_budget,
            crews_available=self.task.crew_count,
            total_crews=self.task.crew_count,
            day=self._day,
            max_days=self.task.max_days,
            weather=weather,
            total_fixed=fixed,
            total_pending=pending,
            episode_reward_so_far=round(self._episode_reward, 4),
        )


# ─────────────────────────────────────────────
# Quick smoke test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    env = PotholeRepairEnv(task_name="critical_repair")
    env.print_info()
    obs = env.reset()

    print(f"State space keys: {list(env.state_space.keys())}")
    print(f"Action space    : {env.action_space}")
    print(f"Episode max len : {env.episode_max_length}")
    print()

    print(f"Task    : {env.task_name}")
    print(f"Day     : {obs.day}/{obs.max_days}")
    print(f"Budget  : ₹{obs.budget_remaining:,.0f}")
    print(f"Potholes: {len(obs.potholes)}")
    print(f"Weather : {obs.weather.condition} | rain={obs.weather.is_raining}")
    print()

    # Take a sample action
    first_pothole = obs.potholes[0]
    print(f"Dispatching crew to {first_pothole.id} (sev={first_pothole.severity})")
    from models import Action, ActionType
    result = env.step(Action(action_type=ActionType.DISPATCH, pothole_id=first_pothole.id))
    print(f"Reward  : {result.reward}")
    print(f"Done    : {result.done}")
    print(f"Budget  : ₹{result.observation.budget_remaining:,.0f}")
    print(f"Score   : {env.get_final_score():.3f}")
