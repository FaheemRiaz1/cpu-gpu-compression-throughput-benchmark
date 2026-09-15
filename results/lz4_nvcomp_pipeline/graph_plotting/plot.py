import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


# Load the aggregated LZ4 + nvCOMP pipeline results together with
# the simpler RLE-based pipeline measurements used for comparison.
df_pipe = pd.read_csv(
    "results/lz4_nvcomp_pipeline/csv_file/"
    "parallel_cpu_lz4_nvcomp_full_pipeline_results.csv"
)

df_simple = pd.read_csv(
    "results/simple_pipeline/csv_file/simple_results.csv"
)

# Normalize CSV headers in case they contain leading or trailing whitespace.
df_pipe.columns = df_pipe.columns.str.strip()
df_simple.columns = df_simple.columns.str.strip()


# Select one compressibility mode from the full pipeline results.
mode = "HIGH"

df_pipe = df_pipe[df_pipe["Mode"] == mode].copy()

# Map the full-pipeline mode to the corresponding synthetic run length
# used by the simple RLE benchmark.
runlen_map = {
    "HIGH": 128,
    "MEDIUM": 32,
    "LOW": 2
}

df_simple = df_simple[
    df_simple["RunLen"] == runlen_map[mode]
].copy()


# Use the complete set of input sizes available in the full pipeline
# as the common x-axis for both datasets.
all_mb = sorted(df_pipe["MB"].unique())

# Align both result tables to the same input-size index.
# Sizes missing from the simple benchmark remain NaN after reindexing.
df_pipe = df_pipe.set_index("MB").reindex(all_mb)
df_simple = df_simple.set_index("MB").reindex(all_mb)

# Replace missing simple-pipeline measurements with zero only for
# plotting; the original NaN values are kept separately for N/A labels.
df_simple_filled = df_simple.fillna(0)


# Create equally spaced categorical x positions for the grouped bars.
x_labels = all_mb

x = np.arange(len(x_labels))
width = 0.25


# Plot the baseline, simple RLE pipeline, and LZ4 + nvCOMP pipeline
# side by side for each tested input size.
plt.figure(figsize=(12, 6))


# Plot the uncompressed baseline throughput with standard-deviation
# error bars from the full pipeline benchmark.
plt.bar(
    x - width,
    df_pipe["Base_GBps_Avg"],
    width,
    yerr=df_pipe["Base_GBps_StdDev"],
    capsize=6,
    error_kw={"ecolor": "red", "elinewidth": 2},
    label="BASE"
)


# Plot the simple RLE pipeline throughput.
bars_simple = plt.bar(
    x,
    df_simple_filled["CompressedGBs"],
    width,
    label="SIMPLE (RLE)"
)


# Plot effective throughput for the LZ4 + nvCOMP pipeline together
# with its measured standard-deviation error bars.
plt.bar(
    x + width,
    df_pipe["Pipe_Eff_GBps_Avg"],
    width,
    yerr=df_pipe["Pipe_Eff_GBps_StdDev"],
    capsize=6,
    error_kw={"ecolor": "red", "elinewidth": 2},
    label="PIPE (LZ4 + nvCOMP)"
)


# Mark input sizes for which the simple benchmark has no measurement.
for i, val in enumerate(df_simple["CompressedGBs"]):

    if pd.isna(val):
        plt.text(
            x[i],
            0.1,
            "N/A",
            ha="center",
            color="black",
            fontsize=9,
            rotation=90
        )


# Describe the compared throughput measurements and selected mode.
plt.xlabel("Data Size (MB)")
plt.ylabel("Throughput (GB/s)")
plt.title(f"BASE vs SIMPLE vs PIPE ({mode})")

plt.xticks(x, x_labels)

# Add a light horizontal grid to make throughput differences easier to read.
plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.5
)

plt.legend()


# Ensure the output directory exists before saving the comparison figure.
os.makedirs(
    "results/graphs",
    exist_ok=True
)

path = (
    f"results/lz4_nvcomp_pipeline/graphs/"
    f"{mode.lower()}_comparison.png"
)

# Save the selected-mode comparison at high resolution.
plt.savefig(
    path,
    dpi=300
)

print(
    "Saved:",
    path
)

plt.tight_layout()

plt.show()


# Run from the repository root:
#
# python3 <path-to-this-script>.py
#
# Inputs:
# results/lz4_nvcomp_pipeline/csv_file/
# parallel_cpu_lz4_nvcomp_full_pipeline_results.csv
#
# results/simple_pipeline/csv_file/simple_results.csv
#
# Output for mode="HIGH":
# results/lz4_nvcomp_pipeline/graphs/high_comparison.png