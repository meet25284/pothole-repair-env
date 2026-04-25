# CivicMind — Multi-Agent Urban Infrastructure Environment

## Problem
Cities lose time, money, and public trust when potholes stay unresolved. Untreated potholes damage vehicles, slow ambulances, and increase crash risk, especially on busy roads.

Municipal teams receive many complaints but have limited crews, budget, and safe weather windows. CivicMind uses AI coordination to prioritize critical repairs while keeping spending and safety under control.

## What Makes CivicMind Different
- Multi-agent (4 specialists)
- Real tool use (inspect before spending)
- Self-improving difficulty
- Covers Theme 1, 3.1, 4

## Multi-Agent Architecture
```text
Central Command
├── Inspector Agent → [inspect_pothole, scan_area]
├── Budget Agent    → [check_budget, approve_spend]
├── Risk Agent      → [get_weather, calc_risk_score]
└── Scheduler Agent → [assign_crew, get_workload]
```

## Environment Details
| Observation Field | Meaning |
|---|---|
| `potholes` | Reports (severity hidden until inspected) |
| `budget_remaining` | Budget left |
| `initial_budget` | Start budget |
| `crews_available` | Daily crews |
| `day`, `max_days` | Day and episode limit |
| `weather` | Rain/condition/temperature |
| `total_fixed`, `total_pending` | Progress counters |

| Action | Effect |
|---|---|
| `dispatch <id>` | Fix pothole now (cost applied) |
| `defer <id>` | Postpone to later |
| `mark_low_priority <id>` | Mark as low-priority backlog |

| Reward Signal | Value |
|---|---|
| Fix critical/high-traffic pothole | Positive (up to `+1.0`) |
| Dispatch in rain | `-0.15` penalty |
| Defer critical busy-road pothole | `-0.20` penalty |
| Mark critical as low priority | `-0.25` penalty |
| Poor actions | Small penalties |

## Tasks
| Task | Difficulty | Goal | Auto-Escalation |
|---|---|---|---|
| `critical_repair` | Easy | Fix severe potholes first | `score >= 0.8` → `budget_optimizer` |
| `budget_optimizer` | Medium | Max repairs with low budget | `score >= 0.7` → `full_city_manager` |
| `full_city_manager` | Hard | Balance weather, budget, urgency | `score >= 0.6` → `generated_city` |

## Results
![Reward Curve](plots/reward_curve.png)  
*Rewards improve and cross threshold.*

![Difficulty Progression](plots/difficulty.png)  
*Difficulty increases as performance improves.*

| Behavior | Before Multi-Agent | After Multi-Agent |
|---|---|---|
| Prioritization | Mostly static rules | Dynamic, inspection-based |
| Rain handling | Occasional unsafe dispatches | Risk-gated dispatch decisions |
| Budget control | Reactive | Planned approvals with fallback |
| Crew usage | First-come assignments | Severity-first scheduling |

## Themes Covered
**Theme 1:** CivicMind models a municipal workflow: detect, inspect, approve, and dispatch. It targets road safety and citizen service quality.  
**Theme 3.1:** Agents use tools (inspection, budget, risk, scheduling) before deciding. Decisions are grounded in environment data.  
**Theme 4:** Automatic task escalation increases difficulty when performance improves. This creates a self-improving loop.

## Links
- HF Space: https://huggingface.co/spaces/meet25284/pothole-repair-env
- Training Colab: https://colab.research.google.com/drive/1l9ESInFrEGa3CRCmxsOZz7aCHWsUeYCQ?usp=sharing
- Blog Post: [URL]
- Demo Video: [URL]

## Quick Start
```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
python inference.py
```

## API Endpoints
| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Health check |
| `GET` | `/reset` | Reset default task |
| `POST` | `/reset` | Reset selected task |
| `POST` | `/step` | Execute one action | 
| `GET` | `/state` | Current observation |
| `POST` | `/inspect` | Reveal pothole severity |
| `GET` | `/complaints` | Simulated complaint queue |
| `GET` | `/leaderboard` | Recorded scores |
| `POST` | `/record_score` | Add score entry |
| `GET` | `/escalate` | Get next task by score |
| `GET` | `/score` | Current grader score |
| `GET` | `/tasks` | List available tasks |
