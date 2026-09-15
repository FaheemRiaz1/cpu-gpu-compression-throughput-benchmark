import pandas as pd
import matplotlib.pyplot as plt


# Load the simple pipeline benchmark results relative to this plotting script.
df = pd.read_csv("../csv_file/simple_results.csv")

# Keep measurements ordered by compressibility setting and input size.
df = df.sort_values(by=["RunLen", "MB"])


# Create the throughput comparison figure.
plt.figure(figsize=(10, 6))


# Use the RunLen=2 measurements as the uncompressed baseline series.
baseline = df[df["RunLen"] == 2]
baseline = baseline.sort_values(by="MB")

plt.plot(
    baseline["MB"],
    baseline["BaselineGBs"],
    linestyle="--",
    color="black",
    marker="o",
    label="Baseline"
)


# Plot compressed-pipeline throughput for three representative
# run lengths corresponding to different compressibility levels.
selected = [2, 32, 128]

for rl in selected:

    subset = df[df["RunLen"] == rl]
    subset = subset.sort_values(by="MB")

    plt.plot(
        subset["MB"],
        subset["CompressedGBs"],
        marker="o",
        label=f"RunLen={rl}"
    )


# Use a base-2 logarithmic x-axis because the tested input sizes
# span powers-of-two ranges.
plt.xscale("log", base=2)


# Describe the benchmark dimensions and throughput metric.
plt.xlabel("Data Size (MB)")
plt.ylabel("Throughput (GB/s)")
plt.title("Impact of Compressibility on Pipeline Performance")


# Show only the tested input sizes as explicit x-axis labels.
plt.xticks(
    [1, 4, 16, 64, 128],
    labels=[1, 4, 16, 64, 128]
)

plt.legend()

# Add a reference grid to make throughput differences easier to compare.
plt.grid()


# Save the generated figure relative to the plotting-script directory.
plt.savefig(
    "../graphs/simple_pipeline_graph_clean.png",
    dpi=300
)

plt.show()


# Run from the directory containing this plotting script:
#
# python3 <script_name>.py
#
# Input:
# ../csv_file/simple_results.csv
#
# Output:
# ../graphs/simple_pipeline_graph_clean.png