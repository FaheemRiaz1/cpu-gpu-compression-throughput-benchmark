import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Detailed per-run measurements produced by the full LZ4 + nvCOMP
# pipeline benchmark.
input_csv = (
    "results/lz4_nvcomp_pipeline/csv_file/"
    "parallel_cpu_lz4_nvcomp_full_pipeline_detailed_runs.csv"
)

# Store the detailed-run visualization with the other pipeline graphs.
output_dir = "results/lz4_nvcomp_pipeline/graphs"
output_path = os.path.join(output_dir, "detailed_runs.png")

# Ensure the output directory exists before saving the figure.
os.makedirs(output_dir, exist_ok=True)


# Load the individual benchmark runs and normalize CSV headers.
df = pd.read_csv(input_csv)

df.columns = df.columns.str.strip()

# Preserve a deterministic ordering by data mode, input size, and run number.
df = df.sort_values(["Mode", "MB", "Run"])


# Collect all tested input sizes in ascending order.
sizes = sorted(df["MB"].unique())


# Compute the mean uncompressed baseline throughput for each input size.
# The baseline is grouped only by size because it does not depend on
# the compressibility mode.
base_mean = (
    df.groupby("MB")["Base_GBps"]
    .mean()
    .reindex(sizes)
)


# Compute mean effective pipeline throughput separately for each
# compressibility mode at every tested input size.
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


# Create one figure showing both individual measurements and
# their corresponding mean throughput trends.
plt.figure(figsize=(12, 7))

# Use categorical x positions so detailed points from different
# modes can be offset slightly without overlapping completely.
x = np.arange(len(sizes))


# Plot every baseline measurement slightly to the left of the
# corresponding input-size position.
for i, mb in enumerate(sizes):

    df_mb = df[df["MB"] == mb]

    y_vals = df_mb["Base_GBps"].values

    x_vals = np.full(len(y_vals), i - 0.12)

    plt.scatter(
        x_vals,
        y_vals,
        alpha=0.35,
        marker="o"
    )


# Plot individual HIGH-compressibility pipeline runs.
for i, mb in enumerate(sizes):

    df_case = df[
        (df["Mode"] == "HIGH") &
        (df["MB"] == mb)
    ]

    y_vals = df_case["Pipe_Eff_GBps"].values

    x_vals = np.full(
        len(y_vals),
        i - 0.04
    )

    plt.scatter(
        x_vals,
        y_vals,
        alpha=0.35,
        marker="s"
    )


# Plot individual MEDIUM-compressibility pipeline runs.
for i, mb in enumerate(sizes):

    df_case = df[
        (df["Mode"] == "MEDIUM") &
        (df["MB"] == mb)
    ]

    y_vals = df_case["Pipe_Eff_GBps"].values

    x_vals = np.full(
        len(y_vals),
        i + 0.04
    )

    plt.scatter(
        x_vals,
        y_vals,
        alpha=0.35,
        marker="^"
    )


# Plot individual RANDOM-data pipeline runs.
for i, mb in enumerate(sizes):

    df_case = df[
        (df["Mode"] == "RANDOM") &
        (df["MB"] == mb)
    ]

    y_vals = df_case["Pipe_Eff_GBps"].values

    x_vals = np.full(
        len(y_vals),
        i + 0.12
    )

    plt.scatter(
        x_vals,
        y_vals,
        alpha=0.35,
        marker="x"
    )


# Overlay mean trend lines so the overall throughput behavior
# remains visible above the individual run-to-run measurements.
plt.plot(
    x,
    base_mean.values,
    marker="o",
    linewidth=2,
    label="BASE mean"
)

plt.plot(
    x,
    pipe_high_mean.values,
    marker="s",
    linewidth=2,
    label="PIPE HIGH mean"
)

plt.plot(
    x,
    pipe_medium_mean.values,
    marker="^",
    linewidth=2,
    label="PIPE MEDIUM mean"
)

plt.plot(
    x,
    pipe_random_mean.values,
    marker="x",
    linewidth=2,
    label="PIPE RANDOM mean"
)


# Display the actual benchmark input sizes at the categorical positions.
plt.xticks(x, sizes)

plt.xlabel("Data Size (MB)")
plt.ylabel("Throughput (GB/s)")
plt.title("Detailed Runs + Mean Trends in One Figure")

# A horizontal grid makes run-to-run throughput variation easier to compare.
plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.5
)

plt.legend()

plt.tight_layout()


# Save a high-resolution copy of the detailed benchmark visualization.
plt.savefig(
    output_path,
    dpi=300
)

print(
    "Saved:",
    output_path
)

plt.show()


# Run from the repository root:
#
# python3 <path-to-this-script>.py
#
# Input:
# results/lz4_nvcomp_pipeline/csv_file/
# parallel_cpu_lz4_nvcomp_full_pipeline_detailed_runs.csv
#
# Output:
# results/lz4_nvcomp_pipeline/graphs/detailed_runs.png