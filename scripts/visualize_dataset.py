import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path("./data/libero_spatial")
FILE = DATA_DIR / "data/chunk-000/file-000.parquet"

print("=" * 60)
print("LIBERO DATA VISUALIZATION")
print("=" * 60)

# ──────────────────────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────────────────────
print("\n📦 Loading data...")
df = pd.read_parquet(FILE)
print(f"Loaded {len(df)} rows")

# Convert columns to numpy (robust decoder)
def to_np(x):
    """Convert dataset field to numpy array (handles dict-encoded images)."""
    if isinstance(x, dict):
        # HuggingFace datasets often store images as {'bytes': ..., 'path': ...}
        if "bytes" in x and x["bytes"] is not None:
            import io
            from PIL import Image
            return np.array(Image.open(io.BytesIO(x["bytes"])).convert("RGB"))
        elif "array" in x:
            return np.array(x["array"])
    return np.array(x)

# ──────────────────────────────────────────────────────────────
# 1. IMAGE VISUALIZATION
# ──────────────────────────────────────────────────────────────
print("\n📸 Visualizing sample images...")

samples = df.sample(3, random_state=0)

fig, axes = plt.subplots(3, 2, figsize=(8, 10))

valid_i = 0
for _, row in samples.iterrows():
    try:
        img1 = to_np(row["observation.images.image"])
        img2 = to_np(row["observation.images.image2"])

        axes[valid_i, 0].imshow(img1.astype(np.uint8))
        axes[valid_i, 0].set_title(f"Sample {valid_i} - Front Cam")
        axes[valid_i, 0].axis("off")

        axes[valid_i, 1].imshow(img2.astype(np.uint8))
        axes[valid_i, 1].set_title(f"Sample {valid_i} - Wrist Cam")
        axes[valid_i, 1].axis("off")

        valid_i += 1
        if valid_i >= 3:
            break
    except Exception as e:
        print(f"Skipping bad sample: {e}")

plt.tight_layout()
Path("./results/plots").mkdir(parents=True, exist_ok=True)
plt.savefig("./results/plots/sample_images.png", dpi=150)
print("✅ Saved: sample_images.png")

# ──────────────────────────────────────────────────────────────
# 2. ACTION DISTRIBUTION
# ──────────────────────────────────────────────────────────────
print("\n📊 Plotting action distribution...")

actions = np.stack(df["action"].apply(to_np).values)

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
axes = axes.flatten()

for i in range(actions.shape[1]):
    axes[i].hist(actions[:, i], bins=50)
    axes[i].set_title(f"Action dim {i}")

plt.tight_layout()
plt.savefig("./results/plots/action_distribution.png", dpi=150)
print("✅ Saved: action_distribution.png")

# ──────────────────────────────────────────────────────────────
# 3. TRAJECTORY VISUALIZATION (🔥 VERY IMPORTANT)
# ──────────────────────────────────────────────────────────────
print("\n🎥 Visualizing a trajectory...")

# pick one episode
episode_id = df["episode_index"].iloc[0]
traj = df[df["episode_index"] == episode_id]

print(f"Episode ID: {episode_id}, length: {len(traj)}")

# plot action over time
actions_traj = np.stack(traj["action"].apply(to_np).values)

plt.figure(figsize=(10, 5))
for i in range(actions_traj.shape[1]):
    plt.plot(actions_traj[:, i], label=f"dim {i}")

plt.title("Action trajectory over time")
plt.xlabel("Timestep")
plt.ylabel("Action value")
plt.legend(ncol=4, fontsize=8)
plt.tight_layout()

plt.savefig("./results/plots/action_trajectory.png", dpi=150)
print("✅ Saved: action_trajectory.png")

# ──────────────────────────────────────────────────────────────
# 4. IMAGE SEQUENCE (mini video strip)
# ──────────────────────────────────────────────────────────────
print("\n🎞 Creating trajectory image strip...")

fig, axes = plt.subplots(1, 6, figsize=(12, 3))

indices = np.linspace(0, len(traj)-1, 6).astype(int)

for i, idx in enumerate(indices):
    img = to_np(traj.iloc[idx]["observation.images.image"])
    axes[i].imshow(img.astype(np.uint8))
    axes[i].set_title(f"t={idx}")
    axes[i].axis("off")

plt.tight_layout()
plt.savefig("./results/plots/trajectory_strip.png", dpi=150)
print("✅ Saved: trajectory_strip.png")

print("\n✅ Visualization complete!")