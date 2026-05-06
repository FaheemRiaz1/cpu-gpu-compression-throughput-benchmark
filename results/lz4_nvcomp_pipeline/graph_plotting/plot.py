import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------- LOAD ----------------
df_pipe = pd.read_csv("results/lz4_nvcomp_pipeline/csv_file/parallel_cpu_lz4_nvcomp_full_pipeline_results.csv")
df_simple = pd.read_csv("results/simple_pipeline/csv_file/simple_results.csv")

df_pipe.columns = df_pipe.columns.str.strip()
df_simple.columns = df_simple.columns.str.strip()

# ---------------- MODE ----------------
mode = "HIGH"
df_pipe = df_pipe[df_pipe["Mode"] == mode].copy()

runlen_map = {
    "HIGH": 128,
    "MEDIUM": 32,
    "LOW": 2
}

df_simple = df_simple[df_simple["RunLen"] == runlen_map[mode]].copy()

# ---------------- FULL MB LIST (FROM PIPE) ----------------
all_mb = sorted(df_pipe["MB"].unique())

# Reindex both → missing becomes NaN
df_pipe = df_pipe.set_index("MB").reindex(all_mb)
df_simple = df_simple.set_index("MB").reindex(all_mb)

# Fill missing with 0 (for plotting)
df_simple_filled = df_simple.fillna(0)

# ---------------- X ----------------
x_labels = all_mb
x = np.arange(len(x_labels))
width = 0.25

# ---------------- PLOT ----------------
plt.figure(figsize=(12,6))

# BASE
plt.bar(
    x - width,
    df_pipe["Base_GBps_Avg"],
    width,
    yerr=df_pipe["Base_GBps_StdDev"],
    capsize=6,
    error_kw={"ecolor": "red", "elinewidth": 2},
    label="BASE"
)

# SIMPLE
bars_simple = plt.bar(
    x,
    df_simple_filled["CompressedGBs"],
    width,
    label="SIMPLE (RLE)"
)

# PIPE
plt.bar(
    x + width,
    df_pipe["Pipe_Eff_GBps_Avg"],
    width,
    yerr=df_pipe["Pipe_Eff_GBps_StdDev"],
    capsize=6,
    error_kw={"ecolor": "red", "elinewidth": 2},
    label="PIPE (LZ4 + nvCOMP)"
)

# ---------------- MARK "NO DATA" ----------------
for i, val in enumerate(df_simple["CompressedGBs"]):
    if pd.isna(val):
        plt.text(x[i], 0.1, "N/A", ha='center', color='black', fontsize=9, rotation=90)

# ---------------- LABELS ----------------
plt.xlabel("Data Size (MB)")
plt.ylabel("Throughput (GB/s)")
plt.title(f"BASE vs SIMPLE vs PIPE ({mode})")

plt.xticks(x, x_labels)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.legend()

# ---------------- SAVE ----------------
os.makedirs("results/graphs", exist_ok=True)
path = f"results/lz4_nvcomp_pipeline/graphs/{mode.lower()}_comparison.png"
plt.savefig(path, dpi=300)

print("Saved:", path)

plt.tight_layout()
plt.show()
