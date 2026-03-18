import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

DATA_DIR = Path("./data/libero_spatial")

print("=" * 60)
print("LIBERO DATASET ANALYSIS")
print("=" * 60)

# ── Info ───────────────────────────────────────────────────────
with open(DATA_DIR / "meta/info.json") as f:
    info = json.load(f)

print(f"\n📦 Robot type:      {info['robot_type']}")
print(f"📊 Total episodes:  {info['total_episodes']}")
print(f"🎞  Total frames:    {info['total_frames']}")
print(f"⏱  FPS:             {info['fps']}")
print(f"📋 Total tasks:     {info['total_tasks']}")

# ── Observation & Action Space ─────────────────────────────────
print("\n── Observation Space ──")
for key, val in info["features"].items():
    if "observation" in key:
        print(f"  {key}: shape={val['shape']}, dtype={val['dtype']}")

print("\n── Action Space ──")
action = info["features"]["action"]
print(f"  action: shape={action['shape']}, dtype={action['dtype']}")
print(f"  (6 DOF end-effector delta + 1 gripper)")

# ── Tasks ──────────────────────────────────────────────────────
tasks_df = pd.read_parquet(DATA_DIR / "meta/tasks.parquet")
print(f"\n── Task List ({len(tasks_df)} tasks) ──")
for _, row in tasks_df.iterrows():
    print(f"  [{row.get('task_index', _):2d}] {row.get('task', row.iloc[0])}")

# ── Episode lengths ────────────────────────────────────────────
print("\n── Loading episode data ──")
df = pd.read_parquet(DATA_DIR / "data/chunk-000/file-000.parquet")
print(f"  Columns: {list(df.columns)}")

all_files = sorted((DATA_DIR / "data/chunk-000").glob("*.parquet"))
print(f"  Parquet files in chunk-000: {len(all_files)}")

# Sample first file for action stats
print("\n── Action Statistics (sample) ──")
print(f"  Action min:  {df['action'].apply(lambda x: np.array(x)).apply(np.min).min():.4f}")
print(f"  Action max:  {df['action'].apply(lambda x: np.array(x)).apply(np.max).max():.4f}")
print(f"  Action mean: {df['action'].apply(lambda x: np.array(x)).apply(np.mean).mean():.4f}")

# ── Episode lengths from stats ─────────────────────────────────
with open(DATA_DIR / "meta/stats.json") as f:
    stats = json.load(f)
print("\n── Dataset Stats (from stats.json) ──")
for key in ["observation.state", "action"]:
    if key in stats:
        s = stats[key]
        print(f"  {key}:")
        print(f"    mean: {np.array(s.get('mean', 'N/A')).round(3)}")
        print(f"    std:  {np.array(s.get('std',  'N/A')).round(3)}")

# ── Train / Val Split ──────────────────────────────────────────
print("\n── Train / Val Split ──")
total_eps = info["total_episodes"]
all_episodes = list(range(total_eps))
np.random.seed(42)
np.random.shuffle(all_episodes)

split = int(0.9 * total_eps)
train_eps = all_episodes[:split]
val_eps   = all_episodes[split:]

print(f"  Total:  {total_eps} episodes")
print(f"  Train:  {len(train_eps)} ({len(train_eps)/total_eps*100:.0f}%)")
print(f"  Val:    {len(val_eps)}  ({len(val_eps)/total_eps*100:.0f}%)")

split_info = {"train_episodes": train_eps, "val_episodes": val_eps}
with open("./data/episode_split.json", "w") as f:
    json.dump(split_info, f)
print("  ✅ Split saved to ./data/episode_split.json")

# ── Plot ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("LIBERO Dataset Analysis", fontsize=14, fontweight="bold")

# Episodes per task (from tasks_df)
task_names = [str(row.iloc[0])[:45] for _, row in tasks_df.iterrows()]
# Approximate equal distribution
eps_per_task = [total_eps // len(tasks_df)] * len(tasks_df)

axes[0].barh(range(len(task_names)), eps_per_task, color="#4f8ef7")
axes[0].set_yticks(range(len(task_names)))
axes[0].set_yticklabels(task_names, fontsize=6)
axes[0].set_title("Tasks")
axes[0].set_xlabel("Est. Episodes")

# Train/val split pie
axes[1].pie(
    [len(train_eps), len(val_eps)],
    labels=[f"Train\n{len(train_eps)}", f"Val\n{len(val_eps)}"],
    colors=["#4f8ef7", "#f7a24f"],
    autopct="%1.1f%%",
    startangle=90
)
axes[1].set_title("Train / Val Split")

plt.tight_layout()
Path("./results/plots").mkdir(parents=True, exist_ok=True)
plt.savefig("./results/plots/dataset_analysis.png", dpi=150, bbox_inches="tight")
print("\n📈 Plot saved to ./results/plots/dataset_analysis.png")
print("\n✅ Analysis complete!")
