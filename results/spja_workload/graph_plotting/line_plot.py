import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator


# csv_path = "results/spja_workload/csv/spja_lz4_nvcomp_hybrid_split_results.csv"
csv_path = "results/spja_workload/csv/spja_lz4_nvcomp_hybrid_split_overlap_results.csv"


out_dir = "results/spja_workload/graphs"
os.makedirs(out_dir, exist_ok=True)

runtime_out_path = os.path.join(out_dir, "cpu_gpu_line.png")
throughput_out_path = os.path.join(out_dir, "cpu_gpu_throughput_line.png")

df = pd.read_csv(csv_path)

mode_name = df["Mode"].iloc[0]
df = df[df["Mode"] == mode_name].copy()
df = df.sort_values("GPU_Percent").reset_index(drop=True)

x = df["GPU_Percent"].to_numpy(dtype=float)

cpu_time = df["CPU_Total_ms_Avg"].to_numpy(dtype=float)
gpu_time = df["GPU_Total_ms_Avg"].to_numpy(dtype=float)

cpu_thr = df["CPU_Effective_GiBps_Avg"].to_numpy(dtype=float)
gpu_thr = df["GPU_Effective_GiBps_Avg"].to_numpy(dtype=float)
hybrid_thr = df["Effective_GiBps_Avg"].to_numpy(dtype=float)

input_mib = df["Input_MiB"].iloc[0]
comp_percent = df["Compression_Reduction_Percent"].iloc[0]

x_smooth = np.linspace(x.min(), x.max(), 300)

cpu_time_curve = PchipInterpolator(x, cpu_time)(x_smooth)
gpu_time_curve = PchipInterpolator(x, gpu_time)(x_smooth)

cpu_thr_curve = PchipInterpolator(x, cpu_thr)(x_smooth)
gpu_thr_curve = PchipInterpolator(x, gpu_thr)(x_smooth)
hybrid_thr_curve = PchipInterpolator(x, hybrid_thr)(x_smooth)

# ------------------------------------------------------------
# Find runtime-balance sweet spot.
# ------------------------------------------------------------

diff = cpu_time - gpu_time
sweet_x = None
sweet_y_time = None

for i in range(len(x) - 1):
    if diff[i] * diff[i + 1] < 0:
        x1, x2 = x[i], x[i + 1]
        d1, d2 = diff[i], diff[i + 1]

        sweet_x = x1 - d1 * (x2 - x1) / (d2 - d1)

        cpu_y = np.interp(sweet_x, [x1, x2], [cpu_time[i], cpu_time[i + 1]])
        gpu_y = np.interp(sweet_x, [x1, x2], [gpu_time[i], gpu_time[i + 1]])

        sweet_y_time = (cpu_y + gpu_y) / 2.0
        break

if sweet_x is None:
    idx = np.argmin(np.abs(diff))
    sweet_x = x[idx]
    sweet_y_time = (cpu_time[idx] + gpu_time[idx]) / 2.0

sweet_cpu_percent = 100.0 - sweet_x
sweet_gpu_percent = sweet_x

# Throughput values at same runtime-balance sweet spot
sweet_cpu_thr = np.interp(sweet_x, x, cpu_thr)
sweet_gpu_thr = np.interp(sweet_x, x, gpu_thr)
sweet_hybrid_thr = np.interp(sweet_x, x, hybrid_thr)
sweet_y_thr = max(sweet_cpu_thr, sweet_gpu_thr)

# Maximum hybrid throughput point
max_thr_idx = int(np.argmax(hybrid_thr))
max_gpu_percent = x[max_thr_idx]
max_cpu_percent = 100.0 - max_gpu_percent
max_hybrid_thr = hybrid_thr[max_thr_idx]
max_y_thr = max(cpu_thr[max_thr_idx], gpu_thr[max_thr_idx])

# ------------------------------------------------------------
# Graph 1: Runtime balance graph
# ------------------------------------------------------------

plt.figure(figsize=(15, 7))

plt.fill_between(
    x_smooth,
    cpu_time_curve,
    alpha=0.15,
    label="CPU Execution Area"
)

plt.fill_between(
    x_smooth,
    gpu_time_curve,
    alpha=0.15,
    label="GPU Execution Area"
)

plt.plot(
    x_smooth,
    cpu_time_curve,
    linewidth=2.7,
    label="CPU Execution Time"
)

plt.plot(
    x_smooth,
    gpu_time_curve,
    linewidth=2.7,
    label="GPU Execution Time"
)

plt.scatter(
    x,
    cpu_time,
    s=65,
    zorder=5,
    label="Measured CPU Points"
)

plt.scatter(
    x,
    gpu_time,
    s=65,
    zorder=5,
    label="Measured GPU Points"
)

plt.axvline(
    sweet_x,
    linestyle="--",
    linewidth=2,
    color="red",
    label="Estimated Sweet Spot"
)

plt.scatter(
    [sweet_x],
    [sweet_y_time],
    s=140,
    zorder=6
)

plt.annotate(
    f"Estimated Sweet Spot\n"
    f"{sweet_cpu_percent:.1f}% CPU / {sweet_gpu_percent:.1f}% GPU\n"
    f"CPU: {sweet_y_time:.2f} ms\n"
    f"GPU: {sweet_y_time:.2f} ms\n"
    f"Diff: 0.00 ms",
    xy=(sweet_x, sweet_y_time),
    xytext=(sweet_x + 8, sweet_y_time + 18),
    arrowprops=dict(arrowstyle="->", linewidth=1.5),
    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black", alpha=0.9),
    fontsize=9
)
plt.title(
    f"CPU/GPU Runtime Balance for TPC-H SPJA Workload\n"
    f"Input: {input_mib:.1f} MiB"
    #  Compression Reduction: {comp_percent:.2f}%
)

plt.xlabel("GPU Workload Percentage (%)")
plt.ylabel("Execution Time (ms)")

plt.xticks(
    x,
    [f"{100-int(g)}% CPU / {int(g)}% GPU" for g in x],
    rotation=25
)

plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()

plt.savefig(runtime_out_path, dpi=300)
plt.close()

# ------------------------------------------------------------
# Graph 2: Throughput graph
# ------------------------------------------------------------

plt.figure(figsize=(15, 7))

plt.fill_between(
    x_smooth,
    cpu_thr_curve,
    alpha=0.15,
    label="CPU Throughput Area"
)

plt.fill_between(
    x_smooth,
    gpu_thr_curve,
    alpha=0.15,
    label="GPU Throughput Area"
)

plt.plot(
    x_smooth,
    cpu_thr_curve,
    linewidth=2.7,
    label="CPU Throughput"
)

plt.plot(
    x_smooth,
    gpu_thr_curve,
    linewidth=2.7,
    label="GPU Throughput"
)

plt.plot(
    x_smooth,
    hybrid_thr_curve,
    linewidth=2.2,
    linestyle="--",
    label="Overall End-to-End Throughput"
)

plt.scatter(
    x,
    cpu_thr,
    s=65,
    zorder=5,
    label="Measured CPU Throughput Points"
)

plt.scatter(
    x,
    gpu_thr,
    s=65,
    zorder=5,
    label="Measured GPU Throughput Points"
)

plt.scatter(
    x,
    hybrid_thr,
    s=55,
    zorder=5,
    marker="x",
    label="Measured Throughput Points"
)

plt.axvline(
    sweet_x,
    linestyle="--",
    linewidth=2,
    color="red",
    label="Runtime-Balance Sweet Spot"
)

plt.scatter(
    [sweet_x],
    [sweet_y_thr],
    s=140,
    zorder=6
)

plt.annotate(
    f"Runtime-Balance Sweet Spot\n"
    f"{sweet_cpu_percent:.1f}% CPU / {sweet_gpu_percent:.1f}% GPU\n"
    f"CPU: {sweet_cpu_thr:.2f} GiB/s\n"
    f"GPU: {sweet_gpu_thr:.2f} GiB/s\n"
    f"Hybrid: {sweet_hybrid_thr:.2f} GiB/s",
    xy=(sweet_x, sweet_y_thr),
    xytext=(sweet_x + 10, sweet_y_thr + 0.75),
    arrowprops=dict(arrowstyle="->", linewidth=1.5),
    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black", alpha=0.9),
    fontsize=10
)

plt.subplots_adjust(
    top=0.82,
    bottom=0.22,
    left=0.08,
    right=0.95
)
plt.title(
    f"CPU/GPU Throughput for TPC-H SPJA Workload\n"
    f"Input: {input_mib:.1f} MiB"
    #  Compression Reduction: {comp_percent:.2f}%
)

plt.xlabel("GPU Workload Percentage (%)")
plt.ylabel("Throughput (GiB/s)")

plt.xticks(
    x,
    [f"{100-int(g)}% CPU / {int(g)}% GPU" for g in x],
    rotation=25
)

ymax = max(np.max(cpu_thr), np.max(gpu_thr), np.max(hybrid_thr), sweet_y_thr, max_y_thr)
plt.ylim(0, ymax * 1.25)

plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()

plt.savefig(throughput_out_path, dpi=300)
plt.close()

print(f"Graph saved: {runtime_out_path}")