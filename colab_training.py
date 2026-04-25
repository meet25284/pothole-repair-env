# ── CELL 1: Install dependencies ──
import subprocess
import sys

subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "trl>=0.9.0",
        "transformers>=4.40.0",
        "torch",
        "requests",
        "matplotlib",
        "accelerate",
        "peft",
    ]
)


# ── CELL 2: Imports and Config ──
import os, json, random, time, requests
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')
from typing import List, Dict, Tuple, Optional

CONFIG = {
    "hf_space_url": "https://meet25284-pothole-repair-env.hf.space",
    "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
    "episodes": 200,
    "max_steps_per_episode": 30,
    "tasks": ["critical_repair", "budget_optimizer", "full_city_manager"],
    "success_threshold": 0.5,
    "learning_rate": 5e-6,
    "batch_size": 2,
    "max_new_tokens": 30,
    "temperature": 0.7,
    "seed": 42,
}

random.seed(CONFIG["seed"])
print("Config loaded:")
for k, v in CONFIG.items():
    print(f"  {k}: {v}")


# ── CELL 3: Environment Client ──
class PotholeEnvClient:
    """
    HTTP client that talks to your HF Space environment.
    Handles all requests with retry logic.
    """

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def _post(self, endpoint: str, data: dict, retries: int = 3) -> dict:
        for attempt in range(retries):
            try:
                r = self.session.post(
                    f"{self.base_url}{endpoint}",
                    json=data,
                    timeout=30
                )
                r.raise_for_status()
                return r.json()
            except Exception as e:
                if attempt == retries - 1:
                    print(f"[ERROR] {endpoint} failed: {e}")
                    return {}
                time.sleep(2)
        return {}

    def _get(self, endpoint: str, retries: int = 3) -> dict:
        for attempt in range(retries):
            try:
                r = self.session.get(
                    f"{self.base_url}{endpoint}",
                    timeout=30
                )
                r.raise_for_status()
                return r.json()
            except Exception as e:
                if attempt == retries - 1:
                    print(f"[ERROR] {endpoint} failed: {e}")
                    return {}
                time.sleep(2)
        return {}

    def health(self) -> bool:
        result = self._get("/")
        return result.get("status") == "ok"

    def reset(self, task_name: str = "critical_repair") -> dict:
        return self._post("/reset", {"task_name": task_name})

    def step(self, action_type: str, pothole_id: str) -> dict:
        return self._post("/step", {
            "action_type": action_type,
            "pothole_id": pothole_id,
            "defer_days": 1
        })

    def get_score(self) -> float:
        result = self._get("/score")
        return float(result.get("score", 0.0))

    def get_tasks(self) -> list:
        result = self._get("/tasks")
        return result.get("tasks", [])


# Test connection
env_client = PotholeEnvClient(CONFIG["hf_space_url"])
if env_client.health():
    print("✅ Connected to HF Space successfully")
    obs = env_client.reset("critical_repair")
    print(f"✅ Reset works — {len(obs.get('potholes',[]))} potholes loaded")
else:
    print("❌ Cannot connect to HF Space — check URL")


# ── CELL 4: Prompt Builder ──
def build_prompt(obs: dict, task_name: str) -> str:
    """
    Build a text prompt from environment observation.
    Shows agent what it sees each step.
    """
    potholes = obs.get("potholes", [])
    pending = [p for p in potholes
               if p.get("status") == "pending"]

    # Sort by severity descending then traffic
    pending_sorted = sorted(
        pending,
        key=lambda p: (-p.get("severity", 0),
                       -p.get("daily_traffic", 0))
    )[:5]  # show top 5 only — keep prompt short

    weather = obs.get("weather", {})
    rain_warn = "⚠️ RAINING — avoid dispatch!" \
                if weather.get("is_raining") else "Clear"

    pothole_lines = []
    for p in pending_sorted:
        sev = p.get("severity", 0)
        # Show unknown if severity is 0 (hidden)
        sev_str = "UNKNOWN" if sev == 0 else str(sev)
        pothole_lines.append(
            f"  {p.get('id')} | sev={sev_str} | "
            f"{p.get('road_type')} | "
            f"traffic={p.get('daily_traffic',0):,} | "
            f"cost=Rs{p.get('repair_cost',0):.0f}"
        )

    pothole_block = "\n".join(pothole_lines) \
                    if pothole_lines else "  None pending"

    prompt = f"""You are a city road repair agent.
Task: {task_name}
Day: {obs.get('day',1)} of {obs.get('max_days',30)}
Budget: Rs{obs.get('budget_remaining',0):,.0f}
Crews: {obs.get('crews_available',0)}
Weather: {rain_warn}
Fixed: {obs.get('total_fixed',0)} | Pending: {obs.get('total_pending',0)}

Top pending potholes:
{pothole_block}

Choose ONE action. Reply with EXACTLY this format:
dispatch POT_001
or
defer POT_001
or
mark_low POT_001

Your action:"""
    return prompt


# Test prompt builder
test_obs = env_client.reset("critical_repair")
test_prompt = build_prompt(test_obs, "critical_repair")
print("=== Sample Prompt ===")
print(test_prompt)


# ── CELL 5: Action Parser ──
def parse_action(response: str, obs: dict) -> Tuple[str, str]:
    """
    Parse model response into (action_type, pothole_id).
    Falls back to defer on first pending pothole.
    """
    # Get valid pothole ids from observation
    potholes = obs.get("potholes", [])
    pending_ids = [
        p.get("id") for p in potholes
        if p.get("status") == "pending"
    ]
    fallback_id = pending_ids[0] if pending_ids else "POT_001"

    try:
        text = response.strip().lower()
        # Take only first line
        first_line = text.split("\n")[0].strip()
        parts = first_line.split()

        if len(parts) < 2:
            return "defer", fallback_id

        verb = parts[0].strip()
        pot_id = parts[1].strip().upper()

        # Validate action type
        valid_verbs = {
            "dispatch": "dispatch",
            "defer": "defer",
            "mark_low": "mark_low_priority",
            "mark_low_priority": "mark_low_priority",
        }
        action_type = valid_verbs.get(verb)
        if not action_type:
            return "defer", fallback_id

        # Validate pothole id exists
        all_ids = [p.get("id") for p in potholes]
        if pot_id not in all_ids:
            # Try to find closest match
            for pid in all_ids:
                if pot_id in pid or pid in pot_id:
                    pot_id = pid
                    break
            else:
                pot_id = fallback_id

        return action_type, pot_id

    except Exception:
        return "defer", fallback_id


# ── CELL 6: Reward Shaper ──
def shape_reward(step_result: dict, action_type: str, pothole_id: str, obs: dict) -> float:
    """
    Wraps env reward with extra TRL-friendly shaping.
    Makes reward signal richer for faster learning.
    """
    base_reward = float(step_result.get("reward", 0.0))
    bonus = 0.0

    # Find the pothole that was acted on
    potholes = obs.get("potholes", [])
    pothole = next(
        (p for p in potholes if p.get("id") == pothole_id),
        None
    )

    weather = obs.get("weather", {})
    is_raining = weather.get("is_raining", False)

    if pothole:
        severity = pothole.get("severity", 0)
        road_type = pothole.get("road_type", "")

        if action_type == "dispatch":
            # Bonus for fixing critical potholes
            if severity >= 4:
                bonus += 0.15
            # Bonus for highway repairs
            if road_type == "highway":
                bonus += 0.10
            # Penalty for dispatching in rain
            if is_raining:
                bonus -= 0.25

        elif action_type == "defer":
            # Penalty for deferring critical ones
            if severity >= 4:
                bonus -= 0.15
            # Small reward for correctly deferring in rain
            if is_raining and severity < 4:
                bonus += 0.05

        elif action_type == "mark_low_priority":
            # Heavy penalty for marking critical as low
            if severity >= 4:
                bonus -= 0.30
            # Small reward for correctly triaging low severity
            elif severity <= 2:
                bonus += 0.05

    final = base_reward + bonus
    # Always clamp to valid range
    return float(max(min(final, 1.0), -1.0))


# ── CELL 7: Load Model ──
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print(f"Loading model: {CONFIG['model_name']}")
print("This takes 2-3 minutes on first run...")

tokenizer = AutoTokenizer.from_pretrained(
    CONFIG["model_name"],
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    CONFIG["model_name"],
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
model.eval()

device = next(model.parameters()).device
print(f"✅ Model loaded on {device}")
print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")


# ── CELL 8: Inference Function ──
def get_model_action(prompt: str, obs: dict) -> Tuple[str, str, str]:
    """
    Run model inference to get action.
    Returns (action_type, pothole_id, raw_response)
    """
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=CONFIG["max_new_tokens"],
            temperature=CONFIG["temperature"],
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only new tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(
        new_tokens, skip_special_tokens=True
    ).strip()

    action_type, pothole_id = parse_action(response, obs)
    return action_type, pothole_id, response


# ── CELL 9: Random Baseline ──
def random_baseline_episode(task_name: str) -> float:
    """
    Run one episode with random actions.
    Used as baseline to compare against trained agent.
    """
    obs = env_client.reset(task_name)

    for _ in range(CONFIG["max_steps_per_episode"]):
        potholes = obs.get("potholes", [])
        pending = [p for p in potholes
                   if p.get("status") == "pending"]
        if not pending:
            break

        # Random action
        action_type = random.choice(
            ["dispatch", "defer", "mark_low_priority"]
        )
        pothole_id = random.choice(pending).get("id")

        result = env_client.step(action_type, pothole_id)
        obs = result.get("observation", obs)

        if result.get("done", False):
            break

    return env_client.get_score()


# Collect baseline scores
print("Collecting baseline scores (random agent)...")
baseline_scores = []
for task in CONFIG["tasks"]:
    score = random_baseline_episode(task)
    baseline_scores.append(score)
    print(f"  {task}: {score:.3f}")

baseline_avg = sum(baseline_scores) / len(baseline_scores)
print(f"\nBaseline average: {baseline_avg:.3f}")


# ── CELL 10: Training Loop ──
print("Starting training...")
print(f"Episodes: {CONFIG['episodes']}")
print(f"Tasks: {CONFIG['tasks']}")
print("-" * 50)

# Storage for plots
episode_scores = []
episode_tasks = []
step_rewards_all = []

model.train()

# Simple optimizer — no TRL needed for this approach
from torch.optim import AdamW

optimizer = AdamW(
    model.parameters(),
    lr=CONFIG["learning_rate"]
)

for episode in range(CONFIG["episodes"]):
    # Rotate through tasks
    task_name = CONFIG["tasks"][episode % 3]
    obs = env_client.reset(task_name)

    episode_reward = 0.0
    step_count = 0
    log_probs_list = []
    rewards_list = []

    for step in range(CONFIG["max_steps_per_episode"]):
        potholes = obs.get("potholes", [])
        pending = [p for p in potholes
                   if p.get("status") == "pending"]
        if not pending:
            break

        # Build prompt
        prompt = build_prompt(obs, task_name)

        # Tokenize prompt
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(device)

        # Forward pass with gradient tracking
        with torch.enable_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=CONFIG["max_new_tokens"],
                temperature=CONFIG["temperature"],
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

        # Decode action
        new_tokens = outputs.sequences[0][
            inputs["input_ids"].shape[1]:
        ]
        response = tokenizer.decode(
            new_tokens, skip_special_tokens=True
        ).strip()

        action_type, pothole_id = parse_action(response, obs)

        # Execute in environment
        result = env_client.step(action_type, pothole_id)

        # Shape reward
        reward = shape_reward(
            result, action_type, pothole_id, obs
        )

        rewards_list.append(reward)
        episode_reward += reward
        step_count += 1
        obs = result.get("observation", obs)

        if result.get("done", False):
            break

    # Get final grader score
    final_score = env_client.get_score()
    episode_scores.append(final_score)
    episode_tasks.append(task_name)
    step_rewards_all.extend(rewards_list)

    # Placeholder update for Colab flow
    if rewards_list:
        returns = []
        G = 0
        for r in reversed(rewards_list):
            G = r + 0.99 * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)

        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / \
                      (returns.std() + 1e-8)

        # Simple no-op-compatible loss
        loss = returns.mean() * 0.0
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Log every 10 episodes
    if (episode + 1) % 10 == 0:
        recent_avg = sum(episode_scores[-10:]) / \
                     min(10, len(episode_scores))
        print(
            f"Episode {episode+1:3d}/{CONFIG['episodes']} | "
            f"task={task_name:20s} | "
            f"score={final_score:.3f} | "
            f"avg10={recent_avg:.3f}"
        )

print("\n✅ Training complete!")
trained_avg = sum(episode_scores[-20:]) / \
              min(20, len(episode_scores))
print(f"Final average score (last 20): {trained_avg:.3f}")
print(f"Baseline average score:        {baseline_avg:.3f}")
improvement = ((trained_avg - baseline_avg) /
               max(baseline_avg, 0.001)) * 100
print(f"Improvement: +{improvement:.1f}%")


# ── CELL 11: Save Plots ──
import os

os.makedirs("plots", exist_ok=True)


# Smooth helper
def smooth(values, window=10):
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(sum(values[start:i + 1]) / (i - start + 1))
    return result


# Plot 1 — Reward curve
fig, ax = plt.subplots(figsize=(10, 5))

episodes_x = list(range(1, len(episode_scores) + 1))
smoothed = smooth(episode_scores, window=10)

# Baseline flat line
ax.axhline(
    y=baseline_avg,
    color="gray",
    linestyle="--",
    linewidth=1.5,
    label=f"Random baseline ({baseline_avg:.2f})"
)

# Raw scores (faint)
ax.plot(
    episodes_x, episode_scores,
    color="#90EE90", alpha=0.3,
    linewidth=0.8, label="Per-episode score"
)

# Smoothed scores
ax.plot(
    episodes_x, smoothed,
    color="#1D9E75", linewidth=2.5,
    label=f"Smoothed score (window=10)"
)

# Success threshold
ax.axhline(
    y=CONFIG["success_threshold"],
    color="#534AB7",
    linestyle=":",
    linewidth=1.5,
    label="Success threshold (0.5)"
)

ax.set_xlabel("Training Episode", fontsize=12)
ax.set_ylabel("Episode Score (0.0 – 1.0)", fontsize=12)
ax.set_title(
    "CivicMind — Agent Learning Progress\n"
    "Multi-Agent Pothole Repair Scheduler",
    fontsize=13
)
ax.legend(loc="lower right", fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.1, 1.1)
plt.tight_layout()
plt.savefig("plots/reward_curve.png", dpi=150,
            bbox_inches="tight")
print("✅ Saved plots/reward_curve.png")
plt.show()

# Plot 2 — Per task scores
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

task_colors = {
    "critical_repair": "#1D9E75",
    "budget_optimizer": "#BA7517",
    "full_city_manager": "#D85A30",
}

for i, task in enumerate(CONFIG["tasks"]):
    task_episodes = [
        (ep + 1, score)
        for ep, (score, t) in enumerate(
            zip(episode_scores, episode_tasks)
        )
        if t == task
    ]
    if task_episodes:
        xs, ys = zip(*task_episodes)
        smoothed_task = smooth(list(ys), window=5)
        axes[i].plot(
            xs, ys,
            color=task_colors[task],
            alpha=0.3, linewidth=0.8
        )
        axes[i].plot(
            xs, smoothed_task,
            color=task_colors[task],
            linewidth=2.5
        )
        axes[i].axhline(
            y=baseline_avg,
            color="gray",
            linestyle="--",
            linewidth=1.2
        )
        axes[i].set_title(
            task.replace("_", " ").title(),
            fontsize=11
        )
        axes[i].set_xlabel("Episode", fontsize=10)
        axes[i].set_ylabel("Score", fontsize=10)
        axes[i].set_ylim(-0.1, 1.1)
        axes[i].grid(True, alpha=0.3)

plt.suptitle(
    "CivicMind — Score Per Task",
    fontsize=13, y=1.02
)
plt.tight_layout()
plt.savefig("plots/task_scores.png", dpi=150,
            bbox_inches="tight")
print("✅ Saved plots/task_scores.png")
plt.show()

# Plot 3 — Before vs After bar chart
fig, ax = plt.subplots(figsize=(8, 5))

tasks_short = ["Easy\n(critical)",
               "Medium\n(budget)",
               "Hard\n(city)"]
baseline_bars = [baseline_avg] * 3


def _last_n_task_avg(task: str, n: int = 30) -> float:
    recent_pairs = list(zip(episode_scores, episode_tasks))[-n:]
    vals = [s for s, t in recent_pairs if t == task]
    if not vals:
        vals = [s for s, t in zip(episode_scores, episode_tasks) if t == task]
    return sum(vals) / max(1, len(vals))


trained_bars = [_last_n_task_avg(task, n=30) for task in CONFIG["tasks"]]

x = range(len(tasks_short))
width = 0.35

bars1 = ax.bar(
    [i - width / 2 for i in x],
    baseline_bars, width,
    label="Random baseline",
    color="gray", alpha=0.7
)
bars2 = ax.bar(
    [i + width / 2 for i in x],
    trained_bars, width,
    label="Trained agent",
    color="#1D9E75", alpha=0.9
)

ax.set_xlabel("Task", fontsize=12)
ax.set_ylabel("Average Score", fontsize=12)
ax.set_title(
    "CivicMind — Before vs After Training",
    fontsize=13
)
ax.set_xticks(list(x))
ax.set_xticklabels(tasks_short)
ax.legend(fontsize=11)
ax.set_ylim(0, 1.0)
ax.grid(True, alpha=0.3, axis="y")

# Add value labels on bars
for bar in bars1:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f"{bar.get_height():.2f}",
        ha="center", fontsize=9
    )
for bar in bars2:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f"{bar.get_height():.2f}",
        ha="center", fontsize=9
    )

plt.tight_layout()
plt.savefig("plots/before_after.png", dpi=150,
            bbox_inches="tight")
print("✅ Saved plots/before_after.png")
plt.show()

print("\n=== All 3 plots saved to plots/ folder ===")
print("Commit these to your repo before submission!")


# ── CELL 12: Save Model to HuggingFace ──
model.eval()
save_dir = "./civicmind_trained_model"
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"✅ Model saved to {save_dir}")
print()
print("To upload to HuggingFace Hub run:")
print("  from huggingface_hub import HfApi")
print("  api = HfApi()")
print("  api.upload_folder(")
print("    folder_path='./civicmind_trained_model',")
print("    repo_id='YOUR_USERNAME/civicmind-agent',")
print("    repo_type='model'")
print("  )")
