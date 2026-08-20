import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ------------------------------------------------------------
# Input data: decompression-only throughput in GB/s
# Add more rows later for nvCOMP Cascaded, G-CasDec, etc.
# ------------------------------------------------------------
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

    # FastLanes 32x has not been measured yet.
    # Add it after running FastLanes on quantity_sf1_repeated_32x.bin:
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

df = pd.DataFrame(data)

# ------------------------------------------------------------
# Save raw comparison table
# ------------------------------------------------------------
out_dir = Path("results/baseline_comparison_graphs")
out_dir.mkdir(parents=True, exist_ok=True)

csv_path = out_dir / "fastlanes_vs_lz4_nvcomp_decompression_comparison.csv"
df.to_csv(csv_path, index=False)

print("Saved CSV:", csv_path)
print(df)

# ------------------------------------------------------------
# Grouped bar chart
# ------------------------------------------------------------
datasets = list(df["Dataset"].drop_duplicates())
systems = list(df["System"].drop_duplicates())

x = np.arange(len(datasets))
width = 0.8 / len(systems)

fig, ax = plt.subplots(figsize=(11, 6))

for i, system in enumerate(systems):
    values = []
    for dataset in datasets:
        row = df[(df["Dataset"] == dataset) & (df["System"] == system)]
        if row.empty:
            values.append(np.nan)
        else:
            values.append(float(row["Throughput_GBps"].iloc[0]))

    offset = (i - (len(systems) - 1) / 2) * width
    bars = ax.bar(x + offset, values, width, label=system)

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

ax.set_title("GPU decompression throughput comparison")
ax.set_xlabel("Dataset size")
ax.set_ylabel("Decompression throughput (GB/s)")
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.4)

fig.tight_layout()

png_path = out_dir / "fastlanes_vs_lz4_nvcomp_decompression_bar.png"
pdf_path = out_dir / "fastlanes_vs_lz4_nvcomp_decompression_bar.pdf"

fig.savefig(png_path, dpi=300)
fig.savefig(pdf_path)

print("Saved PNG:", png_path)
print("Saved PDF:", pdf_path)

plt.show()