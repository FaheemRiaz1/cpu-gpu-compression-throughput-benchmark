import pandas as pd
import matplotlib.pyplot as plt


# Load benchmark results produced by simple_pipeline.
df = pd.read_csv("results/simple_pipeline/csv_file/simple_results.csv")

# Remove accidental leading/trailing whitespace from CSV column names.
df.columns = df.columns.str.strip()

# Keep rows ordered consistently by compressibility and input size.
df = df.sort_values(by=["RunLen", "MB"])

plt.figure(figsize=(10, 6))


# Use RunLen=2 as the source of the uncompressed baseline measurements.
# BaselineGBs represents pipeline throughput without compression.
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


# Compare three representative run lengths.
# Larger run lengths correspond to more compressible input patterns.
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


# Input sizes increase by powers of two, so a base-2 logarithmic
# x-axis keeps the spacing readable across the tested range.
plt.xscale("log", base=2)


# Describe the input size and measured pipeline throughput.
plt.xlabel("Data Size (MB)", fontsize=12)
plt.ylabel("Throughput (GB/s)", fontsize=12)
plt.title("Impact of Compressibility on Pipeline Performance", fontsize=14)


# Show only the benchmark sizes of interest as explicit x-axis ticks.
plt.xticks([1, 4, 16, 64, 128], labels=[1, 4, 16, 64, 128])


# Add a light reference grid without obscuring the measured curves.
plt.grid(True, linestyle="--", alpha=0.5)

plt.legend()


# Create the graph output directory if it does not already exist.
import os

os.makedirs("results/graphs", exist_ok=True)

# Save a high-resolution copy of the generated comparison figure.
plt.savefig("results/graphs/simple_pipeline_lines.png", dpi=300)

plt.tight_layout()
plt.show()


# Run from the repository root:
#
# python3 <path-to-this-script>.py
#
# Input:
# results/simple_pipeline/csv_file/simple_results.csv
#
# Output:
# results/graphs/simple_pipeline_lines.png