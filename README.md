---
title: City Pothole Repair Scheduler
emoji: 🚧
colorFrom: gray
colorTo: blue
sdk: docker
app_port: 7860
short_description: OpenEnv pothole repair — budget, crews, weather & traffic.
tags:
  - openenv
  - scheduling
  - urban-planning
  - resource-optimization
  - fastapi
  - pydantic
fullWidth: true
---

# 🚧 City Pothole Repair Scheduler — OpenEnv

An OpenEnv-compliant environment where an AI agent schedules road repair crews
across a city, balancing pothole severity, traffic impact, budget, crew limits,
and weather windows.

---

## Environment Description

The agent manages pothole repairs in Ahmedabad city. Each day it observes
all known potholes and must decide: fix it now (dispatch), postpone (defer),
or skip it (mark low priority). The goal is to maximize road safety within
real-world constraints of budget, crew count, and weather.

**Why this is a real-world task:** Municipal corporations worldwide face exactly
this problem daily — limited budgets, limited workers, variable weather,
and a backlog of complaints with varying urgency.

---

## Action Space

| Action | Description |
|--------|-------------|
| `dispatch <pothole_id>` | Send crew to fix pothole today |
| `defer <pothole_id>` | Postpone repair (saves budget) |
| `mark_low_priority <pothole_id>` | Skip this pothole for the episode |

---

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `potholes` | list | All potholes with severity, road type, traffic, cost, status |
| `budget_remaining` | float | Remaining repair budget (INR) |
| `crews_available` | int | Repair crews per day |
| `day` | int | Current simulation day |
| `max_days` | int | Episode length |
| `weather` | object | Today's weather (rain flag, temperature) |
| `total_fixed` | int | Potholes fixed so far |
| `total_pending` | int | Potholes still waiting |

---

## Tasks

### 🟢 Easy — `critical_repair`
Fix all severity 4–5 potholes within 30 days.
- Budget: ₹50,000 | Crews: 3 | Potholes: 15
- Grader: % of critical potholes fixed
- Success threshold: 0.8

### 🟡 Medium — `budget_optimizer`
Maximize repairs within a tight budget.
- Budget: ₹20,000 | Crews: 2 | Potholes: 25
- Grader: fix rate × budget efficiency
- Success threshold: 0.6

### 🔴 Hard — `full_city_manager`
Balance severity, budget, weather windows, and traffic over 45 days.
- Budget: ₹30,000 | Crews: 2 | Potholes: 30
- Grader: weighted (40% critical fixed, 30% budget, 20% weather, 10% traffic)
- Success threshold: 0.5

---

## Reward Function

The reward function provides **partial progress signals throughout the episode**,
not just at the end.

| Event | Reward |
|-------|--------|
| Fix high-severity highway pothole | up to +1.0 |
| Fix low-severity residential pothole | ~+0.15 |
| Dispatch on rainy day | -0.15 penalty |
| Defer critical pothole on highway | -0.20 penalty |
| Mark critical pothole as low priority | -0.25 penalty |
| Budget goes negative | -0.30 penalty |
| Already-fixed pothole action | -0.05 penalty |

---

## Setup Instructions

### 1. Clone and install

```bash
git clone https://github.com/meet25284/pothole-repair-env.git
cd pothole-repair-env
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install --only-binary=:all: -r requirements.txt
```

### 2. Set environment variables

```bash
cp .env.example .env
# Edit .env with your HF_TOKEN and MODEL_NAME
```

### 3. Run the server locally

```bash
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
```

### 4. Test endpoints

open http://localhost:7860/docs

### 5. Run baseline inference

```bash
python inference.py
```

### 6. Docker

```bash
docker build -t pothole-repair-env .
docker run -p 7860:7860 --env-file .env pothole-repair-env
```

---

## Baseline Scores

| Task | Difficulty | Baseline Score |
|------|-----------|----------------|
| critical_repair | Easy | ~0.72 |
| budget_optimizer | Medium | ~0.55 |
| full_city_manager | Hard | ~0.41 |

*Scores from Qwen2.5-72B-Instruct via HuggingFace router.*

---

## Project Structure

```
pothole-repair-env/
├── env.py           Main OpenEnv class (reset/step/state)
├── models.py        Pydantic typed models
├── tasks.py         3 task definitions
├── graders.py       Scoring functions (0.0–1.0)
├── data_gen.py      Synthetic data generator
├── app.py           FastAPI HTTP server
├── inference.py     Baseline inference script
├── openenv.yaml     Environment metadata
├── Dockerfile       Container config
├── requirements.txt Python dependencies
└── README.md        This file (includes Hugging Face Spaces YAML config)
```
