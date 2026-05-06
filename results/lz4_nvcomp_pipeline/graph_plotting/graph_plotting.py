import pandas as pd
import matplotlib.pyplot as plt

# -------- LOAD --------
df = pd.read_csv("results/simple_pipeline/csv_file/simple_results.csv")

# Clean column names (IMPORTANT)
df.columns = df.columns.str.strip()

# Sort
df = df.sort_values(by=["RunLen", "MB"])

plt.figure(figsize=(10,6))

# -------- BASELINE --------
baseline = df[df["RunLen"] == 2].sort_values(by="MB")

plt.plot(
    baseline["MB"],
    baseline["BaselineGBs"],
    linestyle="--",
    color="black",
    marker="o",
    linewidth=2,
    label="Baseline (No Compression)"
)

# -------- PIPELINES --------
selected = [2, 32, 128]

colors = {
    2: "red",
    32: "orange",
    128: "green"
}

for rl in selected:
    subset = df[df["RunLen"] == rl].sort_values(by="MB")

    plt.plot(
        subset["MB"],
        subset["CompressedGBs"],
        marker="o",
        linewidth=2,
        color=colors[rl],
        label=f"RunLen = {rl}"
    )

# -------- SCALE --------
plt.xscale("log", base=2)

# -------- LABELS --------
plt.xlabel("Data Size (MB)", fontsize=12)
plt.ylabel("Throughput (GB/s)", fontsize=12)
plt.title("Impact of Compressibility on Pipeline Performance", fontsize=14)

# FIX TICKS
plt.xticks([1, 4, 16, 64, 128], labels=[1,4,16,64,128])

# GRID
plt.grid(True, linestyle="--", alpha=0.5)

# LEGEND
plt.legend()

# -------- SAVE --------
import os
os.makedirs("results/graphs", exist_ok=True)

plt.savefig("results/graphs/simple_pipeline_lines.png", dpi=300)

plt.tight_layout()
plt.show()
