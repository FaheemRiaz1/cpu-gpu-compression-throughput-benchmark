import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Load the overlap-enabled SPJA CPU/GPU split benchmark.
csv_path = (
    "results/spja_workload/csv/"
    "spja_lz4_nvcomp_hybrid_split_overlap_results.csv"
)

# Store the generated throughput comparison with the other SPJA workload graphs.
out_dir = "results/spja_workload/graphs"

os.makedirs(out_dir, exist_ok=True)


# Load only the fair-assignment measurements used for the final split comparison.
df = pd.read_csv(csv_path)

df = df[df["Mode"] == "TPCH_FAIR_ASSIGN"].copy()

if df.empty:
    raise ValueError("No TPCH_FAIR_ASSIGN rows found in CSV.")

# Keep measured splits ordered by increasing GPU workload share.
df = df.sort_values("GPU_Percent").reset_index(drop=True)


# Extract the CPU/GPU workload percentages used by each measured split.
gpu_x = df["GPU_Percent"].astype(float).values
cpu_x = df["CPU_Percent"].astype(float).values


# Device-level throughput reports the effective throughput of the
# workload portion owned by each processor.
cpu_thr = df["CPU_Effective_GiBps_Avg"].astype(float).values
gpu_thr = df["GPU_Effective_GiBps_Avg"].astype(float).values


# End-to-end hybrid throughput is based on the complete logical input
# and the measured overlapped wall-clock execution time.
hybrid_thr = df["Effective_GiBps_Avg"].astype(float).values

# Use reported throughput variability when available; otherwise keep
# zero-length error bars without changing the mean measurements.
if "Effective_GiBps_StdDev" in df.columns:
    hybrid_thr_std = df["Effective_GiBps_StdDev"].astype(float).values
else:
    hybrid_thr_std = np.zeros(len(df))


# CPU and GPU path times are used to estimate where both processors
# finish their assigned portions at approximately the same time.
cpu_time = df["CPU_Total_ms_Avg"].astype(float).values
gpu_time = df["GPU_Total_ms_Avg"].astype(float).values

# Total_ms represents the directly measured end-to-end overlapped runtime.
total_ms = df["Total_ms_Avg"].astype(float).values

if "Total_ms_StdDev" in df.columns:
    total_ms_std = df["Total_ms_StdDev"].astype(float).values
else:
    total_ms_std = np.zeros(len(df))


# Format each measured split as a two-line CPU/GPU x-axis label.
labels = [
    f"{int(row.CPU_Percent)}% CPU\n{int(row.GPU_Percent)}% GPU"
    for row in df.itertuples()
]


# Dataset size and repetition counts are shown in the figure title.
input_mib = df["Input_MiB"].iloc[0]

assignment_trials = (
    int(df["Assignment_Trials"].iloc[0])
    if "Assignment_Trials" in df.columns
    else 1
)

timed_runs = (
    int(df["Timed_Runs_Per_Assignment"].iloc[0])
    if "Timed_Runs_Per_Assignment" in df.columns
    else 1
)


# Runtime balance is meaningful only for true hybrid splits where
# both CPU and GPU receive work and report non-zero execution times.
valid_mask = (
    (gpu_x > 0) &
    (gpu_x < 100) &
    (cpu_time > 0) &
    (gpu_time > 0)
)

valid_indices = np.where(valid_mask)[0]

if len(valid_indices) < 2:
    raise ValueError(
        "Need at least two hybrid split points to estimate runtime balance."
    )


# A sign change in CPU_time - GPU_time indicates that the CPU/GPU
# runtime curves cross between two adjacent measured hybrid splits.
time_diff_signed = cpu_time - gpu_time

balance_est_found = False


for left_idx, right_idx in zip(
    valid_indices[:-1],
    valid_indices[1:]
):

    d_left = time_diff_signed[left_idx]
    d_right = time_diff_signed[right_idx]


    # If a measured split already has identical CPU/GPU path times,
    # use that point directly rather than interpolating.
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


    # When the runtime difference changes sign, estimate the balance
    # point using linear interpolation between the two measurements.
    if d_left * d_right < 0:

        alpha = -d_left / (d_right - d_left)


        # Interpolate the categorical plot position so the vertical
        # balance line appears between the corresponding measured bars.
        balance_x_pos = (
            left_idx +
            alpha * (right_idx - left_idx)
        )


        # Interpolate the CPU/GPU workload percentages at the crossing.
        balance_gpu = (
            gpu_x[left_idx] +
            alpha * (gpu_x[right_idx] - gpu_x[left_idx])
        )

        balance_cpu = (
            cpu_x[left_idx] +
            alpha * (cpu_x[right_idx] - cpu_x[left_idx])
        )


        # Interpolate CPU, GPU, and system-level timing values at
        # the same estimated workload split.
        balance_cpu_ms = (
            cpu_time[left_idx] +
            alpha * (cpu_time[right_idx] - cpu_time[left_idx])
        )

        balance_gpu_ms = (
            gpu_time[left_idx] +
            alpha * (gpu_time[right_idx] - gpu_time[left_idx])
        )

        balance_total_ms = (
            total_ms[left_idx] +
            alpha * (total_ms[right_idx] - total_ms[left_idx])
        )

        balance_total_ms_std = (
            total_ms_std[left_idx] +
            alpha * (total_ms_std[right_idx] - total_ms_std[left_idx])
        )


        # Interpolate throughput values at the runtime-balanced split.
        balance_hybrid_thr = (
            hybrid_thr[left_idx] +
            alpha * (hybrid_thr[right_idx] - hybrid_thr[left_idx])
        )

        balance_hybrid_thr_std = (
            hybrid_thr_std[left_idx] +
            alpha * (
                hybrid_thr_std[right_idx] -
                hybrid_thr_std[left_idx]
            )
        )

        balance_cpu_thr = (
            cpu_thr[left_idx] +
            alpha * (cpu_thr[right_idx] - cpu_thr[left_idx])
        )

        balance_gpu_thr = (
            gpu_thr[left_idx] +
            alpha * (gpu_thr[right_idx] - gpu_thr[left_idx])
        )

        balance_diff_ms = abs(
            balance_cpu_ms -
            balance_gpu_ms
        )

        balance_est_found = True

        break


# If the measured CPU/GPU runtime curves do not cross, use the
# hybrid split with the smallest observed runtime difference.
if not balance_est_found:

    balance_diff = np.abs(
        cpu_time -
        gpu_time
    )

    closest_idx = valid_indices[
        np.argmin(
            balance_diff[valid_indices]
        )
    ]

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


# Identify the actually measured split that achieved the highest
# end-to-end hybrid throughput.
max_thr_idx = int(
    np.argmax(hybrid_thr)
)

max_gpu = gpu_x[max_thr_idx]
max_cpu = cpu_x[max_thr_idx]

max_hybrid_thr = hybrid_thr[max_thr_idx]
max_hybrid_thr_std = hybrid_thr_std[max_thr_idx]

max_total_ms = total_ms[max_thr_idx]
max_total_ms_std = total_ms_std[max_thr_idx]


# Create a combined device-level and end-to-end throughput figure.
plt.figure(figsize=(16, 7))

x = np.arange(len(df))

bar_width = 0.28


# CPU and GPU bars show the effective throughput of their assigned
# workload portions at each measured split.
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


# Overlay measured end-to-end system throughput as a line so it is
# clearly distinguishable from the processor-level component bars.
plt.plot(
    x,
    hybrid_thr,
    marker="o",
    linewidth=2.2,
    color="green",
    label="Hybrid End-to-End Throughput"
)


# Draw variability for hybrid throughput when the benchmark CSV
# contains non-zero standard deviations.
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


# Mark the estimated point where CPU and GPU path runtimes are balanced.
plt.axvline(
    x=balance_x_pos,
    color="red",
    linestyle="--",
    linewidth=2,
    label="Estimated Runtime Balance"
)


# Report workload split, processor throughputs, hybrid throughput,
# and timing information at the runtime-balance estimate.
plt.annotate(
    f"Estimated Runtime Balance\n"
    f"{balance_cpu:.1f}% CPU / {balance_gpu:.1f}% GPU\n"
    f"CPU throughput: {balance_cpu_thr:.2f} GiB/s\n"
    f"GPU throughput: {balance_gpu_thr:.2f} GiB/s\n"
    f"Hybrid throughput: {balance_hybrid_thr:.2f}"
    + (
        f" ± {balance_hybrid_thr_std:.2f}"
        if balance_hybrid_thr_std > 0
        else ""
    )
    + " GiB/s\n"
    f"CPU time: {balance_cpu_ms:.2f} ms\n"
    f"GPU time: {balance_gpu_ms:.2f} ms\n"
    f"Diff: {balance_diff_ms:.2f} ms\n"
    f"Total: {balance_total_ms:.2f}"
    + (
        f" ± {balance_total_ms_std:.2f}"
        if balance_total_ms_std > 0
        else ""
    )
    + " ms",

    xy=(
        balance_x_pos,
        balance_hybrid_thr
    ),

    xytext=(
        balance_x_pos + 0.55,
        balance_hybrid_thr + 0.35
    ),

    arrowprops=dict(
        arrowstyle="->",
        lw=1.8,
        color="black"
    ),

    fontsize=9,

    bbox=dict(
        boxstyle="round,pad=0.3",
        fc="white",
        ec="black",
        alpha=0.95
    )
)


# Highlight the best measured hybrid-throughput split separately
# when it is not effectively the same position as runtime balance.
if abs(max_thr_idx - balance_x_pos) > 0.15:

    plt.annotate(
        f"Maximum Throughput\n"
        f"{max_cpu:.0f}% CPU / {max_gpu:.0f}% GPU\n"
        f"{max_hybrid_thr:.2f}"
        + (
            f" ± {max_hybrid_thr_std:.2f}"
            if max_hybrid_thr_std > 0
            else ""
        )
        + " GiB/s\n"
        f"Total: {max_total_ms:.2f}"
        + (
            f" ± {max_total_ms_std:.2f}"
            if max_total_ms_std > 0
            else ""
        )
        + " ms",

        xy=(
            max_thr_idx,
            max_hybrid_thr
        ),

        xytext=(
            max_thr_idx + 0.35,
            max_hybrid_thr - 0.50
        ),

        arrowprops=dict(
            arrowstyle="->",
            lw=1.5,
            color="black"
        ),

        fontsize=9,

        bbox=dict(
            boxstyle="round,pad=0.25",
            fc="white",
            ec="black",
            alpha=0.90
        )
    )


# Use the actual measured CPU/GPU split percentages as x-axis labels.
plt.xticks(
    x,
    labels,
    rotation=25,
    ha="right"
)


plt.title(
    f"CPU vs GPU Throughput\n"
    f"TPC-H SPJA Workload, Input: {input_mib:.1f} MiB\n"
    f"{assignment_trials} fair assignments × "
    f"{timed_runs} timed runs per split"
)

plt.xlabel("CPU/GPU Workload Split")
plt.ylabel("Throughput (GiB/s)")

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.6
)

plt.legend()


# Size the y-axis using all plotted throughput values and include
# hybrid standard-deviation error bars when they are available.
ymax = max(
    np.max(cpu_thr),
    np.max(gpu_thr),
    np.max(hybrid_thr)
)

if np.any(hybrid_thr_std > 0):
    ymax = max(
        ymax,
        np.max(
            hybrid_thr +
            hybrid_thr_std
        )
    )

plt.ylim(
    0,
    ymax * 1.32
)


# Reserve enough space for the multi-line title, rotated split labels,
# legend, and annotation boxes.
plt.subplots_adjust(
    top=0.82,
    bottom=0.20,
    left=0.08,
    right=0.98
)


# Save the final high-resolution throughput comparison.
save_path = os.path.join(
    out_dir,
    "cpu_gpu_throughput.png"
)

plt.savefig(
    save_path,
    dpi=300
)

plt.close()

print(
    "Final graph saved:",
    save_path
)


# Run from the repository root:
#
# python3 <path-to-this-script>.py
#
# Input:
# results/spja_workload/csv/
# spja_lz4_nvcomp_hybrid_split_overlap_results.csv
#
# Output:
# results/spja_workload/graphs/cpu_gpu_throughput.png