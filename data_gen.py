"""
data_gen.py - Synthetic data generator for City Pothole Repair Scheduler
Generates realistic pothole and weather data for Ahmedabad city.
"""

from __future__ import annotations
import random
from typing import List, Tuple
from models import PotholeReport, WeatherWindow, RoadType, PotholeStatus


# Ahmedabad city center coordinates
BASE_LAT = 23.0225
BASE_LNG = 72.5714

# Cost factors per road type (INR)
REPAIR_COST_BASE = {
    "highway": 8000,
    "arterial": 4000,
    "residential": 1500,
}

# Traffic ranges per road type (vehicles/day)
TRAFFIC_RANGES = {
    "highway":     (40000, 120000),
    "arterial":    (10000, 40000),
    "residential": (500,   8000),
}


def generate_potholes(n: int = 20, seed: int = 42) -> List[PotholeReport]:
    """
    Generate n synthetic pothole reports for Ahmedabad city.
    Uses seed for full reproducibility.
    """
    rng = random.Random(seed)
    potholes: List[PotholeReport] = []

    road_types = list(RoadType)
    # Weight: more residential/arterial potholes than highway
    road_weights = [0.25, 0.40, 0.35]

    # Severity weights: more low severity than critical
    severity_weights = [0.10, 0.20, 0.30, 0.25, 0.15]  # sev 1-5

    for i in range(n):
        road_type = rng.choices(road_types, weights=road_weights, k=1)[0]
        severity  = rng.choices([1, 2, 3, 4, 5], weights=severity_weights, k=1)[0]

        # Scatter lat/lng around Ahmedabad center (~15km radius)
        lat = BASE_LAT + rng.uniform(-0.12, 0.12)
        lng = BASE_LNG + rng.uniform(-0.12, 0.12)

        traffic_min, traffic_max = TRAFFIC_RANGES[road_type.value]
        daily_traffic = rng.randint(traffic_min, traffic_max)

        base_cost = REPAIR_COST_BASE[road_type.value]
        repair_cost = base_cost * severity * rng.uniform(0.8, 1.4)

        potholes.append(PotholeReport(
            id=f"POT_{i+1:03d}",
            lat=round(lat, 6),
            lng=round(lng, 6),
            severity=severity,
            road_type=road_type,
            daily_traffic=daily_traffic,
            repair_cost=round(repair_cost, 2),
            status=PotholeStatus.PENDING,
            days_pending=0,
        ))

    return potholes


def generate_weather(days: int = 45, seed: int = 42) -> List[WeatherWindow]:
    """
    Generate weather forecast for each day of the episode.
    ~20% chance of rain per day. Realistic Ahmedabad temperatures.
    """
    rng = random.Random(seed + 100)
    weather: List[WeatherWindow] = []

    for day in range(1, days + 1):
        is_raining = rng.random() < 0.20  # 20% rain chance
        temperature = rng.uniform(22.0, 42.0)

        if is_raining:
            condition = "rainy"
            temperature = rng.uniform(18.0, 28.0)  # cooler on rain days
        elif temperature > 36:
            condition = "hot"
        else:
            condition = "clear"

        weather.append(WeatherWindow(
            day=day,
            is_raining=is_raining,
            temperature=round(temperature, 1),
            condition=condition,
        ))

    return weather


def get_traffic_factor(road_type: str) -> float:
    """Return reward multiplier based on road type (higher traffic = higher impact)."""
    factors = {
        "highway":     2.0,
        "arterial":    1.5,
        "residential": 1.0,
    }
    return factors.get(road_type, 1.0)


if __name__ == "__main__":
    potholes = generate_potholes(n=10, seed=42)
    weather   = generate_weather(days=5, seed=42)

    print("=== Sample Potholes ===")
    for p in potholes:
        print(f"  {p.id} | sev={p.severity} | {p.road_type} | traffic={p.daily_traffic} | cost=₹{p.repair_cost:.0f}")

    print("\n=== Sample Weather ===")
    for w in weather:
        print(f"  Day {w.day:2d} | {w.condition:6s} | rain={w.is_raining} | temp={w.temperature}°C")
