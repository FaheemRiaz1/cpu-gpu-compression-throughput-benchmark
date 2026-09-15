import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


# Benchmark input table containing measured GPU decompression throughput
# in GB/s for each system and dataset size.
# Additional systems can be appended later using the same schema.
data = [
    {
        "Dataset": "SF1 / 22.9 MiB",
        "System": "FastLanes-GPU bitpacking",
        "Throughput_GBps": 215.147,
    },
    {
        "Dataset": "SF1 / 22.9 MiB",
        "System": "LZ4_HC/nvCOMP LZ4",
        "Throughput_GBps": 1.793,
    },

    # FastLanes for the repeated 32x dataset can be updated later
    # by replacing or extending the measured value below if needed.
    {
        "Dataset": "32x / 732 MiB",
        "System": "FastLanes-GPU bitpacking",
        "Throughput_GBps": 212.112,
    },
    {
        "Dataset": "32x / 732 MiB",
        "System": "LZ4_HC/nvCOMP LZ4",
        "Throughput_GBps": 12.652,
    },
    {
        "Dataset": "224x / 5 GiB",
        "System": "FastLanes-GPU bitpacking",
        "Throughput_GBps": 214.467,
    },
    {
        "Dataset": "224x / 5 GiB",
        "System": "LZ4_HC/nvCOMP LZ4",
        "Throughput_GBps": 13.904,
    },
]

# Convert the in-script benchmark table into a DataFrame for export and plotting.
df = pd.DataFrame(data)


# Create the output directory used for both the raw comparison CSV
# and the rendered bar-chart figures.
out_dir = Path("results/baseline_comparison_graphs")
out_dir.mkdir(parents=True, exist_ok=True)

# Save the benchmark table so the plotted values are also available
# in a reusable tabular form.
csv_path = out_dir / "fastlanes_vs_lz4_nvcomp_decompression_comparison.csv"
df.to_csv(csv_path, index=False)

print("Saved CSV:", csv_path)
print(df)


# Preserve dataset order and system order as they first appear
# in the DataFrame.
datasets = list(df["Dataset"].drop_duplicates())
systems = list(df["System"].drop_duplicates())

# Use one categorical x-position per dataset and divide the available
# group width evenly across all compared systems.
x = np.arange(len(datasets))
width = 0.8 / len(systems)

fig, ax = plt.subplots(figsize=(11, 6))


# Plot one bar series per decompression system.
for i, system in enumerate(systems):

    values = []

    for dataset in datasets:

        row = df[
            (df["Dataset"] == dataset) &
            (df["System"] == system)
        ]

        # Missing measurements are represented as NaN so the grouped
        # layout still remains aligned across datasets.
        if row.empty:
            values.append(np.nan)
        else:
            values.append(float(row["Throughput_GBps"].iloc[0]))

    # Center the system bars around each dataset position.
    offset = (i - (len(systems) - 1) / 2) * width

    bars = ax.bar(
        x + offset,
        values,
        width,
        label=system
    )

    # Print the numeric throughput above each available bar.
    for bar, value in zip(bars, values):

        if not np.isnan(value):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
                rotation=0,
            )


# Label the chart using the measured decompression-throughput metric.
ax.set_title("GPU decompression throughput comparison")
ax.set_xlabel("Dataset size")
ax.set_ylabel("Decompression throughput (GB/s)")

ax.set_xticks(x)
ax.set_xticklabels(datasets)

ax.legend()

# Add a light horizontal grid to make bar-height comparison easier.
ax.grid(axis="y", linestyle="--", alpha=0.4)

fig.tight_layout()


# Save the chart in both PNG and PDF formats for reuse in reports,
# slides, or thesis figures.
png_path = out_dir / "fastlanes_vs_lz4_nvcomp_decompression_bar.png"
pdf_path = out_dir / "fastlanes_vs_lz4_nvcomp_decompression_bar.pdf"

fig.savefig(png_path, dpi=300)
fig.savefig(pdf_path)

print("Saved PNG:", png_path)
print("Saved PDF:", pdf_path)

plt.show()


# Run from the repository root:
#
# python3 <path-to-this-script>.py
#
# Outputs:
# results/baseline_comparison_graphs/fastlanes_vs_lz4_nvcomp_decompression_comparison.csv
# results/baseline_comparison_graphs/fastlanes_vs_lz4_nvcomp_decompression_bar.png
# results/baseline_comparison_graphs/fastlanes_vs_lz4_nvcomp_decompression_bar.pdf