from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Paths
# ============================================================
csv_path = Path("results/nvcomp_lz4_vs_fastlane/csv/final_grouped_plot_data.csv")
out_dir = Path("results/nvcomp_lz4_vs_fastlane/graphs")
out_dir.mkdir(parents=True, exist_ok=True)

# ============================================================
# Load data
# ============================================================
df = pd.read_csv(csv_path)
df.columns = [c.strip() for c in df.columns]

if "Include" not in df.columns:
    df["Include"] = "YES"

df["Include"] = df["Include"].astype(str).str.strip().str.upper()
df = df[df["Include"] == "YES"].copy()

df["nvCOMP_LZ4_GBs"] = pd.to_numeric(df["nvCOMP_LZ4_GBs"], errors="coerce")
df["FastLanes_GBs"] = pd.to_numeric(df["FastLanes_GBs"], errors="coerce")
df["nvCOMP_Reduction"] = pd.to_numeric(df["nvCOMP_Reduction"], errors="coerce")

datasets = df["Dataset"].astype(str).tolist()
nvcomp_vals = df["nvCOMP_LZ4_GBs"].to_numpy()
fastlanes_vals = df["FastLanes_GBs"].to_numpy()
nvcomp_reductions = df["nvCOMP_Reduction"].to_numpy()
fastlanes_labels = df["FastLanes_Label"].astype(str).tolist()

# ============================================================
# Plot setup
# ============================================================
x = np.arange(len(datasets))
width = 0.34

fig, ax = plt.subplots(figsize=(15, 7.5))

nvcomp_color = "#0B7D0B"
fastlanes_color = "#E60000"

bars_nvcomp = ax.bar(
    x - width / 2,
    np.nan_to_num(nvcomp_vals, nan=0.0),
    width,
    label="nvCOMP LZ4",
    color=nvcomp_color,
    edgecolor="black",
    linewidth=0.8,
)

bars_fastlanes = ax.bar(
    x + width / 2,
    np.nan_to_num(fastlanes_vals, nan=0.0),
    width,
    label="FastLanes-GPU",
    color=fastlanes_color,
    edgecolor="black",
    linewidth=0.8,
)

# ============================================================
# Titles and labels
# ============================================================
ax.set_title(
    "GPU Decompression Throughput: nvCOMP LZ4 vs FastLanes-GPU\n"
    "TPC-H SF10 Quantity Column",
    fontsize=18,
    fontweight="bold",
    pad=16,
)

ax.set_ylabel("Decompression throughput (GB/s)", fontsize=14, fontweight="bold")
ax.set_xlabel("Data layout / dataset size", fontsize=14, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=11, fontweight="bold", rotation=10, ha="right")

ax.tick_params(axis="y", labelsize=12)
ax.grid(axis="y", linestyle="--", alpha=0.35)

legend = ax.legend(fontsize=12, loc="upper left", frameon=True)
legend.get_frame().set_alpha(0.95)

# Y-axis height
all_real_values = []
for v in list(nvcomp_vals) + list(fastlanes_vals):
    if not np.isnan(v):
        all_real_values.append(v)

max_y = max(all_real_values) if all_real_values else 1.0
ax.set_ylim(0, max_y * 1.25)

# ============================================================
# Value labels + compression/encoding labels
# ============================================================
for bar, value, reduction in zip(bars_nvcomp, nvcomp_vals, nvcomp_reductions):
    x_pos = bar.get_x() + bar.get_width() / 2

    if np.isnan(value):
        ax.text(
            x_pos,
            max_y * 0.055,
            "N/A",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            rotation=90,
        )
    else:
        ax.text(
            x_pos,
            value + max_y * 0.020,
            f"{value:.1f} GB/s",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

        ax.text(
            x_pos,
            value + max_y * 0.075,
            f"LZ4 red. {reduction:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=nvcomp_color,
        )

for bar, value, label in zip(bars_fastlanes, fastlanes_vals, fastlanes_labels):
    x_pos = bar.get_x() + bar.get_width() / 2

    if np.isnan(value):
        ax.text(
            x_pos,
            max_y * 0.055,
            "FAILED",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            rotation=90,
        )
    else:
        ax.text(
            x_pos,
            value + max_y * 0.020,
            f"{value:.1f} GB/s",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

        ax.text(
            x_pos,
            value + max_y * 0.075,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=fastlanes_color,
        )

# ============================================================
# Bottom note
# ============================================================
note = (
    "Green: nvCOMP LZ4 resident decompression-only; percentage shows LZ4 compression reduction. "
    "Red: FastLanes-GPU unpack/decompression using packed representation. "
    "Sorted layouts preserve value distribution but are LZ4-friendly best-case layouts."
)

fig.text(0.5, 0.012, note, ha="center", fontsize=9.5)

plt.tight_layout(rect=[0, 0.065, 1, 1])

# ============================================================
# Save
# ============================================================
png_path = out_dir / "nvcomp_lz4_vs_fastlanes.png"
# pdf_path = out_dir / "nvcomp_lz4_vs_fastlanes_sf10_grouped_bar_with_reduction.pdf"

plt.savefig(png_path, dpi=300, bbox_inches="tight")
# plt.savefig(pdf_path, bbox_inches="tight")
plt.close()

print("Saved:")
print(png_path)
# print(pdf_path)
print()
print("CSV used:")
print(csv_path)