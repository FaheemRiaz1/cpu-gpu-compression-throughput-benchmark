
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../csv_file/simple_results.csv")
df = df.sort_values(by=["RunLen", "MB"])

plt.figure(figsize=(10,6))

# Baseline
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

# Pipeline curves
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

# 🔥 THIS FIXES YOUR SHAPE
plt.xscale('log', base=2)

plt.xlabel("Data Size (MB)")
plt.ylabel("Throughput (GB/s)")
plt.title("Impact of Compressibility on Pipeline Performance")

plt.xticks([1, 4, 16, 64, 128], labels=[1,4,16,64,128])

plt.legend()
plt.grid()

plt.savefig("../graphs/simple_pipeline_graph_clean.png", dpi=300)
plt.show()
