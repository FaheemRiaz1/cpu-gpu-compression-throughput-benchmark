import os
import pandas as pd
import matplotlib.pyplot as plt

# Input CSV file from final LZ4 + nvCOMP benchmark
input_csv = "results/lz4_nvcomp_pipeline/csv_file/parallel_cpu_lz4_nvcomp_full_pipeline_results.csv"

# Correct graph output folder
output_dir = "results/lz4_nvcomp_pipeline/graphs"
output_file = "base_vs_pipe_all_modes_line.png"
output_path = os.path.join(output_dir, output_file)

# Create output folder if it does not exist
os.makedirs(output_dir, exist_ok=True)

# Load results
df = pd.read_csv(input_csv)
df.columns = df.columns.str.strip()

modes = ["HIGH", "MEDIUM", "RANDOM"]

# Sort data
df = df.sort_values(["Mode", "MB"])

# All tested input sizes
all_mb = sorted(df["MB"].unique())

# ============================================================
df_base = (
    df.groupby("MB", as_index=False)
      .agg({
          "Base_GBps_Avg": "mean",
          "Base_GBps_StdDev": "mean"
      })
      .sort_values("MB")
)

# Plot
plt.figure(figsize=(12, 6))

# Baseline
plt.errorbar(
    df_base["MB"],
    df_base["Base_GBps_Avg"],
    yerr=df_base["Base_GBps_StdDev"],
    marker="o",
    linewidth=2,
    capsize=5,
    label="BASE"
)

# Pipeline lines for each data mode
for mode in modes:
    df_mode = df[df["Mode"] == mode].sort_values("MB")

    plt.errorbar(
        df_mode["MB"],
        df_mode["Pipe_Eff_GBps_Avg"],
        yerr=df_mode["Pipe_Eff_GBps_StdDev"],
        marker="o",
        linewidth=2,
        capsize=5,
        label=f"PIPE ({mode})"
    )

# Labels and layout
plt.xlabel("Data Size (MB)")
plt.ylabel("Effective Throughput (GB/s)")
plt.title("Baseline vs LZ4 + nvCOMP Pipeline Across Data Modes")

plt.xscale("log", base=2)
plt.xticks(all_mb, all_mb)

plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.legend()

# Save graph
plt.tight_layout()
plt.savefig(output_path, dpi=300)

print(f"Saved graph to: {output_path}")

plt.show()
