import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

csv_path = "results/spja_workload/csv/spja_lz4_nvcomp_hybrid_split_overlap_results.csv"
out_dir = "results/spja_workload/graphs"
os.makedirs(out_dir, exist_ok=True)

df = pd.read_csv(csv_path)
df = df[df["Mode"] == "TPCH_FAIR_ASSIGN"].copy()

if df.empty:
    raise ValueError("No TPCH_FAIR_ASSIGN rows found in CSV.")

df = df.sort_values("GPU_Percent").reset_index(drop=True)

gpu_x = df["GPU_Percent"].astype(float).values
cpu_percent = df["CPU_Percent"].astype(float).values

cpu_times = df["CPU_Total_ms_Avg"].astype(float).values
gpu_times = df["GPU_Total_ms_Avg"].astype(float).values

# ------------------------------------------------------------
# Standard deviation values for error bars.
# ------------------------------------------------------------
if "CPU_Total_ms_StdDev" in df.columns:
    cpu_times_std = df["CPU_Total_ms_StdDev"].astype(float).values
else:
    cpu_times_std = np.zeros(len(df))

if "GPU_Total_ms_StdDev" in df.columns:
    gpu_times_std = df["GPU_Total_ms_StdDev"].astype(float).values
else:
    gpu_times_std = np.zeros(len(df))

avg_times = (cpu_times + gpu_times) / 2.0
diff = cpu_times - gpu_times

# ------------------------------------------------------------
# Estimate crossing point where CPU time == GPU time
# ------------------------------------------------------------
sweet_gpu = gpu_x[np.argmin(np.abs(diff))]
sweet_cpu_time = cpu_times[np.argmin(np.abs(diff))]
sweet_gpu_time = gpu_times[np.argmin(np.abs(diff))]
sweet_diff = abs(sweet_cpu_time - sweet_gpu_time)
interpolated = False

for i in range(len(diff) - 1):
    if diff[i] == 0:
        sweet_gpu = gpu_x[i]
        sweet_cpu_time = cpu_times[i]
        sweet_gpu_time = gpu_times[i]
        sweet_diff = 0.0
        interpolated = False
        break

    if diff[i] * diff[i + 1] < 0:
        t = abs(diff[i]) / (abs(diff[i]) + abs(diff[i + 1]))

        sweet_gpu = gpu_x[i] + t * (gpu_x[i + 1] - gpu_x[i])
        sweet_cpu_time = cpu_times[i] + t * (cpu_times[i + 1] - cpu_times[i])
        sweet_gpu_time = gpu_times[i] + t * (gpu_times[i + 1] - gpu_times[i])
        sweet_diff = abs(sweet_cpu_time - sweet_gpu_time)
        interpolated = True
        break

sweet_cpu = 100.0 - sweet_gpu
sweet_avg = (sweet_cpu_time + sweet_gpu_time) / 2.0

# ------------------------------------------------------------
# Use categorical x positions so style matches throughput graph
# ------------------------------------------------------------
x = np.arange(len(df))
bar_width = 0.28

# Convert estimated sweet GPU percentage into categorical x-position
sweet_pos = np.interp(sweet_gpu, gpu_x, x)

labels = [
    f"{int(row.CPU_Percent)}% CPU\n{int(row.GPU_Percent)}% GPU"
    for row in df.itertuples()
]

input_mib = df["Input_MiB"].iloc[0]

assignment_trials = int(df["Assignment_Trials"].iloc[0]) if "Assignment_Trials" in df.columns else 1
timed_runs = int(df["Timed_Runs_Per_Assignment"].iloc[0]) if "Timed_Runs_Per_Assignment" in df.columns else 1

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
plt.figure(figsize=(16, 7))

plt.bar(
    x - bar_width / 2,
    cpu_times,
    width=bar_width,
    yerr=cpu_times_std,
    capsize=4,
    error_kw=dict(elinewidth=1.2, capthick=1.2),
    color="orange",
    label="CPU Execution Time"
)

plt.bar(
    x + bar_width / 2,
    gpu_times,
    width=bar_width,
    yerr=gpu_times_std,
    capsize=4,
    error_kw=dict(elinewidth=1.2, capthick=1.2),
    color="tab:blue",
    label="GPU Execution Time"
)

plt.axvline(
    x=sweet_pos,
    color="red",
    linestyle="--",
    linewidth=2,
    label="Estimated Sweet Spot" if interpolated else "Sweet Spot"
)
plt.annotate(
    f"{'Estimated ' if interpolated else ''}Sweet Spot\n"
    f"{sweet_cpu:.1f}% CPU / {sweet_gpu:.1f}% GPU\n"
    f"CPU: {sweet_cpu_time:.2f} ms\n"
    f"GPU: {sweet_gpu_time:.2f} ms\n"
    f"Diff: {sweet_diff:.2f} ms",
    xy=(sweet_pos, sweet_avg * 1.08),
    xytext=(sweet_pos + 0.35, sweet_avg + 18),
    arrowprops=dict(arrowstyle="->", lw=1.8, color="black"),
    fontsize=9,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.95)
)

plt.xticks(x, labels, rotation=25, ha="right")

plt.title(
    f"CPU/GPU Runtime Balance\n"
    f"TPC-H SPJA Workload, Input: {input_mib:.1f} MiB\n"
    f"{assignment_trials} fair assignments × {timed_runs} timed runs per split"
)

plt.xlabel("CPU/GPU Workload Split")
plt.ylabel("Execution Time (ms)")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.legend(loc="upper right")

ymax = max(
    np.max(cpu_times + cpu_times_std),
    np.max(gpu_times + gpu_times_std),
    sweet_avg
)

plt.ylim(0, ymax * 1.18)
plt.xlim(-0.5, len(x) - 0.5)

plt.subplots_adjust(top=0.82, bottom=0.20, left=0.08, right=0.98)

save_path = os.path.join(out_dir, "cpu_vs_gpu.png")
plt.savefig(save_path, dpi=300)
plt.close()

print("Graph saved:", save_path)