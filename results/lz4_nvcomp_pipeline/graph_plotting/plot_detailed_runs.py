import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# INPUT / OUTPUT
# -----------------------------
input_csv = "results/lz4_nvcomp_pipeline/csv_file/parallel_cpu_lz4_nvcomp_full_pipeline_detailed_runs.csv"
output_dir = "results/lz4_nvcomp_pipeline/graphs"
output_path = os.path.join(output_dir, "detailed_runs.png")

os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# LOAD
# -----------------------------
df = pd.read_csv(input_csv)
df.columns = df.columns.str.strip()

# Keep order clean
df = df.sort_values(["Mode", "MB", "Run"])

# -----------------------------
# PREPARE DATA
# -----------------------------
sizes = sorted(df["MB"].unique())

# Mean baseline per size
base_mean = (
    df.groupby("MB")["Base_GBps"]
    .mean()
    .reindex(sizes)
)

# Mean pipeline effective throughput per size and mode
pipe_high_mean = (
    df[df["Mode"] == "HIGH"]
    .groupby("MB")["Pipe_Eff_GBps"]
    .mean()
    .reindex(sizes)
)

pipe_medium_mean = (
    df[df["Mode"] == "MEDIUM"]
    .groupby("MB")["Pipe_Eff_GBps"]
    .mean()
    .reindex(sizes)
)

pipe_random_mean = (
    df[df["Mode"] == "RANDOM"]
    .groupby("MB")["Pipe_Eff_GBps"]
    .mean()
    .reindex(sizes)
)

# -----------------------------
# PLOT
# -----------------------------
plt.figure(figsize=(12, 7))

# X positions
x = np.arange(len(sizes))

# Plot all detailed run points with small horizontal offsets
# BASE detailed points
for i, mb in enumerate(sizes):
    df_mb = df[df["MB"] == mb]
    y_vals = df_mb["Base_GBps"].values
    x_vals = np.full(len(y_vals), i - 0.12)
    plt.scatter(x_vals, y_vals, alpha=0.35, marker="o")

# HIGH detailed points
for i, mb in enumerate(sizes):
    df_case = df[(df["Mode"] == "HIGH") & (df["MB"] == mb)]
    y_vals = df_case["Pipe_Eff_GBps"].values
    x_vals = np.full(len(y_vals), i - 0.04)
    plt.scatter(x_vals, y_vals, alpha=0.35, marker="s")

# MEDIUM detailed points
for i, mb in enumerate(sizes):
    df_case = df[(df["Mode"] == "MEDIUM") & (df["MB"] == mb)]
    y_vals = df_case["Pipe_Eff_GBps"].values
    x_vals = np.full(len(y_vals), i + 0.04)
    plt.scatter(x_vals, y_vals, alpha=0.35, marker="^")

# RANDOM detailed points
for i, mb in enumerate(sizes):
    df_case = df[(df["Mode"] == "RANDOM") & (df["MB"] == mb)]
    y_vals = df_case["Pipe_Eff_GBps"].values
    x_vals = np.full(len(y_vals), i + 0.12)
    plt.scatter(x_vals, y_vals, alpha=0.35, marker="x")

# Mean lines on top
plt.plot(x, base_mean.values, marker="o", linewidth=2, label="BASE mean")
plt.plot(x, pipe_high_mean.values, marker="s", linewidth=2, label="PIPE HIGH mean")
plt.plot(x, pipe_medium_mean.values, marker="^", linewidth=2, label="PIPE MEDIUM mean")
plt.plot(x, pipe_random_mean.values, marker="x", linewidth=2, label="PIPE RANDOM mean")

# -----------------------------
# LABELS
# -----------------------------
plt.xticks(x, sizes)
plt.xlabel("Data Size (MB)")
plt.ylabel("Throughput (GB/s)")
plt.title("Detailed Runs + Mean Trends in One Figure")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()

# -----------------------------
# SAVE
# -----------------------------
plt.savefig(output_path, dpi=300)
print("Saved:", output_path)

plt.show()
