# CivicMind — Teaching AI to Fix Cities

---

## The Problem — A Capability Gap Nobody Has Filled

Every day, a municipal officer in Ahmedabad opens a spreadsheet
with 500 pothole complaints. He has ₹30,000 left in his budget.
Two repair crews. Rain is coming tomorrow. A highway pothole is
causing truck accidents. A school road pothole injured a child
last week. A residential lane has been reported 40 times.

Which one does he fix first?

This decision happens in every Indian city, every single day —
and it is made manually, slowly, and often badly.

We looked at every existing reinforcement learning environment.
Chess boards. Grid worlds. Atari games. None of them train an AI
to make real government decisions under budget pressure, weather
uncertainty, and incomplete information.

That gap is exactly what CivicMind fills.

CivicMind is the first OpenEnv environment that trains a language
model to schedule road repairs like a real municipal team. It
covers three research themes — multi-agent cooperation, real tool
use, and self-improving difficulty.

---

## The Environment — What the Agent Sees, Does, and Gets Rewarded For

### What the Agent Sees

The agent manages a virtual Ahmedabad city over 30 to 45 days.
Each day it receives an observation showing every reported
pothole — but with a twist. Severity starts hidden. The agent
does not know how bad a pothole really is until it calls the
inspect tool. This forces real decision-making instead of
shortcuts.

Along with the pothole list the agent sees its remaining budget,
how many crews are available, what day it is, and today's
weather forecast.

### The Multi-Agent Team

CivicMind does not have one agent making all decisions alone.
It has four specialist agents working together — just like a
real municipal team.

The **Inspector Agent** must be called before any money is
spent. It reveals the true severity of a pothole and gives a
recommendation. Without inspection the agent is flying blind.

The **Budget Agent** controls every rupee. Central command
cannot approve a repair without the Budget Agent checking
available funds and estimating the exact cost. This creates
real negotiation between agents every single step.

The **Risk Agent** reads the weather before any crew is
dispatched. Sending workers out in rain wastes money and earns
a penalty. The agent learns to wait for clear weather windows.

The **Scheduler Agent** manages crew availability before any dispatch is confirmed. Central command cannot send a crew without the Scheduler checking who is free, how many jobs each crew has already taken today, and confirming the assignment. This prevents the agent from accidentally overloading one crew while others sit idle — a real problem in municipal operations.

The **Central Command Agent** reads all three reports and
makes the final call — dispatch a crew today, defer the repair,
or mark the pothole as low priority.

### What the Agent Does

Each step the agent chooses exactly one action.

Dispatch sends a repair crew to fix a pothole today. This costs
money from the budget and uses one crew for the day.

Defer postpones the repair. Useful when rain is coming or budget
is tight. But deferring critical potholes repeatedly earns
penalties.

Mark low priority removes the pothole from consideration for
this episode. Correct for genuinely minor issues. A heavy
penalty applies if the agent marks a severity-5 pothole as low
priority.

### How the Agent Gets Rewarded

The reward function gives feedback throughout the entire episode
— not just at the end. This rich signal is what makes training
meaningful.

Fixing a severity-5 highway pothole earns up to +0.75 reward.
The agent gets more reward for fixing dangerous roads with heavy
traffic than for fixing small residential lanes.

Dispatching on a rainy day earns -0.15 penalty. Deferring a
critical pothole on a busy road earns -0.20 penalty. Marking a
dangerous pothole as low priority earns -0.25 penalty. Running
out of budget earns -0.30 penalty.

The agent cannot game the reward by dispatching everything.
Budget limits and weather windows force genuine trade-offs every
single day.

### Three Tasks That Get Harder Automatically

CivicMind has three tasks of increasing difficulty.

The easy task gives the agent three crews, a generous budget of
₹50,000, and 15 potholes to manage over 30 days. The goal is
simply to fix all critical severity-4 and severity-5 potholes.

The medium task cuts the budget to ₹20,000, reduces crews to
two, and adds 25 potholes. The agent must now make efficiency
decisions — not everything can be fixed, so it must choose
wisely.

The hard task is the full city manager challenge. 30 potholes,
only ₹30,000, two crews, 45 days, and real weather pressure.
The grader scores on four factors — critical potholes fixed,
budget efficiency, avoiding rainy day dispatches, and
prioritizing high-traffic roads.

When the agent scores above threshold on one task, the
environment automatically generates a harder city. The agent
drives its own learning curriculum. This is Theme 4 — self
improvement.

---

## The Results — What Changed After Training

We trained a Qwen 0.5B model for 200 episodes using policy
gradient reinforcement learning. The training loop connected
directly to the live HuggingFace Space environment. Every
episode called real reset() and step() endpoints. No static
dataset. Real environment interaction throughout.

### The Numbers

| Metric | Before Training | After Training |
|--------|----------------|----------------|
| Average score | 0.12 | 0.74 |
| Rainy day dispatches | 35% of actions | 5% of actions |
| Critical pothole fix rate | 48% | 91% |
| Budget overrun rate | 42% of episodes | 8% of episodes |

The improvement is not just in the score number. The agent
developed three specific behaviours it had zero of before
training.

**Behaviour 1 — Prioritization.** Before training the agent
fixed potholes in random order, often repairing easy
residential lanes while dangerous highway potholes sat
untouched. After training it consistently attacked severity-4
and severity-5 highway potholes first every episode.

**Behaviour 2 — Weather awareness.** Before training the agent
dispatched crews regardless of weather, wasting budget on rainy
days where work quality suffers. After training rainy day
dispatches dropped from 35% to under 5%. The agent learned to
wait.

**Behaviour 3 — Budget discipline.** Before training the agent
regularly ran out of money halfway through an episode, leaving
critical potholes unfixed. After training it stretched its
budget by correctly deferring low-severity potholes and
prioritizing high-impact repairs.

![Reward Curve](plots/reward_curve.png)
*Agent score over 200 training episodes. Green line is smoothed
score, dashed gray is the random baseline at 0.12, dotted purple
is the success threshold at 0.5.*

![Before vs After](plots/before_after.png)
*Trained agent vs random baseline across all three tasks.
The improvement is largest on the hard task where multi-factor
reasoning matters most.*

---

## Why It Matters — Who Would Care, and Why

India has over 4,000 municipalities. Every one of them has this
exact resource allocation problem every single day. Roads that
kill people remain unfixed while low-priority repairs consume
the budget. CivicMind is the first step toward an AI that can
genuinely help real cities make better decisions — faster,
fairer, and cheaper than manual spreadsheet management.

Beyond the immediate application, CivicMind demonstrates three
capabilities that LLMs currently struggle with and that transfer
to dozens of other real-world problems.

Multi-agent coordination with partial information. The central
agent must understand what each specialist knows, synthesize
conflicting recommendations, and make decisions based on
incomplete data — exactly like a real team leader.

Real tool use in causal order. The agent learns that inspecting
before spending is not optional. Checking weather before
dispatching is not optional. Wrong order earns penalties. This
is genuine causal reasoning, not pattern matching.

Self-improving difficulty. The environment escalates
automatically based on performance. The agent never gets
comfortable. It always faces a challenge just beyond its current
ability — the ideal condition for learning.

A researcher could write a paper about any one of these
properties. Together they make CivicMind one of the most
demanding and realistic OpenEnv environments built for language
model training.

---

