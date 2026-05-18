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
cpu_x = df["CPU_Percent"].astype(float).values

# Device-level throughputs
cpu_thr = df["CPU_Effective_GiBps_Avg"].astype(float).values
gpu_thr = df["GPU_Effective_GiBps_Avg"].astype(float).values

# End-to-end hybrid throughput
hybrid_thr = df["Effective_GiBps_Avg"].astype(float).values

if "Effective_GiBps_StdDev" in df.columns:
    hybrid_thr_std = df["Effective_GiBps_StdDev"].astype(float).values
else:
    hybrid_thr_std = np.zeros(len(df))

cpu_time = df["CPU_Total_ms_Avg"].astype(float).values
gpu_time = df["GPU_Total_ms_Avg"].astype(float).values
total_ms = df["Total_ms_Avg"].astype(float).values

if "Total_ms_StdDev" in df.columns:
    total_ms_std = df["Total_ms_StdDev"].astype(float).values
else:
    total_ms_std = np.zeros(len(df))

labels = [
    f"{int(row.CPU_Percent)}% CPU\n{int(row.GPU_Percent)}% GPU"
    for row in df.itertuples()
]

input_mib = df["Input_MiB"].iloc[0]
assignment_trials = int(df["Assignment_Trials"].iloc[0]) if "Assignment_Trials" in df.columns else 1
timed_runs = int(df["Timed_Runs_Per_Assignment"].iloc[0]) if "Timed_Runs_Per_Assignment" in df.columns else 1

valid_mask = (
    (gpu_x > 0) &
    (gpu_x < 100) &
    (cpu_time > 0) &
    (gpu_time > 0)
)

valid_indices = np.where(valid_mask)[0]

if len(valid_indices) < 2:
    raise ValueError("Need at least two hybrid split points to estimate runtime balance.")

time_diff_signed = cpu_time - gpu_time

balance_est_found = False

for left_idx, right_idx in zip(valid_indices[:-1], valid_indices[1:]):
    d_left = time_diff_signed[left_idx]
    d_right = time_diff_signed[right_idx]

    # Exact measured equality
    if d_left == 0:
        balance_x_pos = float(left_idx)
        balance_gpu = gpu_x[left_idx]
        balance_cpu = cpu_x[left_idx]
        balance_cpu_ms = cpu_time[left_idx]
        balance_gpu_ms = gpu_time[left_idx]
        balance_total_ms = total_ms[left_idx]
        balance_total_ms_std = total_ms_std[left_idx]
        balance_hybrid_thr = hybrid_thr[left_idx]
        balance_hybrid_thr_std = hybrid_thr_std[left_idx]
        balance_cpu_thr = cpu_thr[left_idx]
        balance_gpu_thr = gpu_thr[left_idx]
        balance_diff_ms = 0.0
        balance_est_found = True
        break

    # Crossing between two measured points
    if d_left * d_right < 0:
        alpha = -d_left / (d_right - d_left)

        # x-position in categorical plot
        balance_x_pos = left_idx + alpha * (right_idx - left_idx)

        # Workload split
        balance_gpu = gpu_x[left_idx] + alpha * (gpu_x[right_idx] - gpu_x[left_idx])
        balance_cpu = cpu_x[left_idx] + alpha * (cpu_x[right_idx] - cpu_x[left_idx])

        # Interpolated times
        balance_cpu_ms = cpu_time[left_idx] + alpha * (cpu_time[right_idx] - cpu_time[left_idx])
        balance_gpu_ms = gpu_time[left_idx] + alpha * (gpu_time[right_idx] - gpu_time[left_idx])
        balance_total_ms = total_ms[left_idx] + alpha * (total_ms[right_idx] - total_ms[left_idx])
        balance_total_ms_std = total_ms_std[left_idx] + alpha * (total_ms_std[right_idx] - total_ms_std[left_idx])

        # Interpolated throughput values
        balance_hybrid_thr = hybrid_thr[left_idx] + alpha * (hybrid_thr[right_idx] - hybrid_thr[left_idx])
        balance_hybrid_thr_std = hybrid_thr_std[left_idx] + alpha * (hybrid_thr_std[right_idx] - hybrid_thr_std[left_idx])
        balance_cpu_thr = cpu_thr[left_idx] + alpha * (cpu_thr[right_idx] - cpu_thr[left_idx])
        balance_gpu_thr = gpu_thr[left_idx] + alpha * (gpu_thr[right_idx] - gpu_thr[left_idx])

        balance_diff_ms = abs(balance_cpu_ms - balance_gpu_ms)
        balance_est_found = True
        break

# Fallback: if there is no crossing, use closest measured hybrid split
if not balance_est_found:
    balance_diff = np.abs(cpu_time - gpu_time)
    closest_idx = valid_indices[np.argmin(balance_diff[valid_indices])]

    balance_x_pos = float(closest_idx)
    balance_gpu = gpu_x[closest_idx]
    balance_cpu = cpu_x[closest_idx]
    balance_cpu_ms = cpu_time[closest_idx]
    balance_gpu_ms = gpu_time[closest_idx]
    balance_total_ms = total_ms[closest_idx]
    balance_total_ms_std = total_ms_std[closest_idx]
    balance_hybrid_thr = hybrid_thr[closest_idx]
    balance_hybrid_thr_std = hybrid_thr_std[closest_idx]
    balance_cpu_thr = cpu_thr[closest_idx]
    balance_gpu_thr = gpu_thr[closest_idx]
    balance_diff_ms = balance_diff[closest_idx]

# ------------------------------------------------------------
# Find maximum measured hybrid throughput point
# ------------------------------------------------------------

max_thr_idx = int(np.argmax(hybrid_thr))

max_gpu = gpu_x[max_thr_idx]
max_cpu = cpu_x[max_thr_idx]
max_hybrid_thr = hybrid_thr[max_thr_idx]
max_hybrid_thr_std = hybrid_thr_std[max_thr_idx]
max_total_ms = total_ms[max_thr_idx]
max_total_ms_std = total_ms_std[max_thr_idx]

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

plt.figure(figsize=(16, 7))

x = np.arange(len(df))
bar_width = 0.28

plt.bar(
    x - bar_width / 2,
    cpu_thr,
    width=bar_width,
    color="orange",
    label="CPU Throughput"
)

plt.bar(
    x + bar_width / 2,
    gpu_thr,
    width=bar_width,
    color="tab:blue",
    label="GPU Throughput"
)

plt.plot(
    x,
    hybrid_thr,
    marker="o",
    linewidth=2.2,
    color="green",
    label="Hybrid End-to-End Throughput"
)

if np.any(hybrid_thr_std > 0):
    plt.errorbar(
        x,
        hybrid_thr,
        yerr=hybrid_thr_std,
        fmt="none",
        ecolor="black",
        elinewidth=1.2,
        capsize=4
    )

# ------------------------------------------------------------
# Estimated runtime-balance line
# ------------------------------------------------------------

plt.axvline(
    x=balance_x_pos,
    color="red",
    linestyle="--",
    linewidth=2,
    label="Estimated Runtime Balance"
)

plt.annotate(
    f"Estimated Runtime Balance\n"
    f"{balance_cpu:.1f}% CPU / {balance_gpu:.1f}% GPU\n"
    f"CPU throughput: {balance_cpu_thr:.2f} GiB/s\n"
    f"GPU throughput: {balance_gpu_thr:.2f} GiB/s\n"
    f"Hybrid throughput: {balance_hybrid_thr:.2f}"
    + (f" ± {balance_hybrid_thr_std:.2f}" if balance_hybrid_thr_std > 0 else "")
    + " GiB/s\n"
    f"CPU time: {balance_cpu_ms:.2f} ms\n"
    f"GPU time: {balance_gpu_ms:.2f} ms\n"
    f"Diff: {balance_diff_ms:.2f} ms\n"
    f"Total: {balance_total_ms:.2f}"
    + (f" ± {balance_total_ms_std:.2f}" if balance_total_ms_std > 0 else "")
    + " ms",
    xy=(balance_x_pos, balance_hybrid_thr),
    xytext=(balance_x_pos + 0.55, balance_hybrid_thr + 0.35),
    arrowprops=dict(arrowstyle="->", lw=1.8, color="black"),
    fontsize=9,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.95)
)

# ------------------------------------------------------------
# Maximum throughput annotation
# ------------------------------------------------------------

if abs(max_thr_idx - balance_x_pos) > 0.15:
    plt.annotate(
        f"Maximum Throughput\n"
        f"{max_cpu:.0f}% CPU / {max_gpu:.0f}% GPU\n"
        f"{max_hybrid_thr:.2f}"
        + (f" ± {max_hybrid_thr_std:.2f}" if max_hybrid_thr_std > 0 else "")
        + " GiB/s\n"
        f"Total: {max_total_ms:.2f}"
        + (f" ± {max_total_ms_std:.2f}" if max_total_ms_std > 0 else "")
        + " ms",
        xy=(max_thr_idx, max_hybrid_thr),
        xytext=(max_thr_idx + 0.35, max_hybrid_thr - 0.50),
        arrowprops=dict(arrowstyle="->", lw=1.5, color="black"),
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", alpha=0.90)
    )

plt.xticks(x, labels, rotation=25, ha="right")

plt.title(
    f"CPU vs GPU Throughput\n"
    f"TPC-H SPJA Workload, Input: {input_mib:.1f} MiB\n"
    f"{assignment_trials} fair assignments × {timed_runs} timed runs per split"
)

plt.xlabel("CPU/GPU Workload Split")
plt.ylabel("Throughput (GiB/s)")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.legend()

ymax = max(np.max(cpu_thr), np.max(gpu_thr), np.max(hybrid_thr))
if np.any(hybrid_thr_std > 0):
    ymax = max(ymax, np.max(hybrid_thr + hybrid_thr_std))

plt.ylim(0, ymax * 1.32)
plt.subplots_adjust(top=0.82, bottom=0.20, left=0.08, right=0.98)

save_path = os.path.join(out_dir, "cpu_gpu_throughput.png")
plt.savefig(save_path, dpi=300)
plt.close()

print("Final graph saved:", save_path)
