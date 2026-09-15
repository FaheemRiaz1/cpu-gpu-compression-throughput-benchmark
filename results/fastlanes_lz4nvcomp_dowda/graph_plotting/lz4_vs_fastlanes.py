#!/usr/bin/env python3
from __future__ import annotations

# Plot the final LZ4/nvCOMP vs FastLanes SPJA comparison from saved CSV results.
# The script only reads benchmark artifacts and writes figures; it does not rerun benchmarks.

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Repository-relative locations for the two result sources and generated figures.
BASE = Path("results/fastlanes_lz4nvcomp_dowda")

CSV_FILES = {
    "LZ4/nvCOMP": Path("results/spja_workload/csv/spja_lz4_nvcomp_hybrid_split_overlap_results.csv"),
    "FastLanes": BASE / "csv" / "fastlanes_spja_coproc_x40_results.csv",
}

OUT_DIR = BASE / "graphs"
OUT_EXEC = OUT_DIR / "lz4_vs_fastlanes_execution_time_stacked.png"
OUT_THROUGHPUT = OUT_DIR / "lz4_vs_fastlanes_throughput_stacked.png"

# Keep a fixed split order so both systems are plotted on the same x-axis.
SPLITS = [
    "100CPU/0GPU",
    "75CPU/25GPU",
    "50CPU/50GPU",
    "25CPU/75GPU",
    "0CPU/100GPU",
]

# Plotting order also determines the left/right position of each system within a split.
SYSTEMS = [
    "LZ4/nvCOMP",
    "FastLanes",
]

# Dark shades represent CPU components; lighter shades represent GPU components.
COLORS = {
    "LZ4/nvCOMP": ("#1f77b4", "#86c5f4"),
    "FastLanes": ("#ff7f0e", "#ffc36b"),
}


# Normalize CSV headers so small naming differences between benchmark outputs
# can be handled through a common set of candidate column names.
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        str(c).strip().lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("%", "percent")
        for c in df.columns
    ]
    return df


# Resolve a required column and fail with enough context to diagnose schema changes.
def find_col(df: pd.DataFrame, candidates: Iterable[str], label: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c

    raise KeyError(
        f"\nCould not find column for {label}.\n"
        f"Tried: {list(candidates)}\n"
        f"Available columns: {list(df.columns)}\n"
    )


# Resolve optional fields without rejecting older result CSV layouts.
def optional_col(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


# Accept the correctness labels used by the different benchmark CSV writers.
def truthy_match(value) -> bool:
    s = str(value).strip().upper()
    return s in {"YES", "Y", "TRUE", "1", "MATCH", "VALID"}


# Canonical split label used for filtering, sorting, and plotting.
def split_label(cpu_percent: float, gpu_percent: float) -> str:
    return f"{int(round(cpu_percent))}CPU/{int(round(gpu_percent))}GPU"


# Load one benchmark CSV and map its schema to the common plotting representation.
# Only configured CPU/GPU splits with valid query results are retained.
def load_csv_for_system(system_name: str, path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV for {system_name}: {path}")

    df = normalize_columns(pd.read_csv(path))

    # The LZ4/nvCOMP and FastLanes CSVs use slightly different column names,
    # so each logical field is resolved from a small set of accepted aliases.
    cpu_col = find_col(
        df,
        ["cpu_percent", "cpu_pct", "cpu"],
        f"{system_name} CPU percent",
    )

    gpu_col = find_col(
        df,
        ["gpu_percent", "gpu_pct", "gpu"],
        f"{system_name} GPU percent",
    )

    input_col = find_col(
        df,
        [
            "input_mib",
            "mib",
            "query_input_size_mib",
            "query_input_mib",
            "input_size_mib",
            "input_size",
        ],
        f"{system_name} input MiB",
    )

    cpu_ms_col = find_col(
        df,
        [
            "cpu_ms",
            "cpu_total_ms_avg",
            "cpu_total_ms",
            "cpu_time_ms",
            "cpu_ms_avg",
            "cpu_time_ms_avg",
        ],
        f"{system_name} CPU ms",
    )

    gpu_ms_col = find_col(
        df,
        [
            "gpu_ms",
            "gpu_total_ms_avg",
            "gpu_total_ms",
            "gpu_time_ms",
            "gpu_ms_avg",
            "gpu_time_ms_avg",
        ],
        f"{system_name} GPU ms",
    )

    total_ms_col = optional_col(
        df,
        [
            "total_ms_avg",
            "total_ms",
            "e2e_ms",
            "elapsed_ms",
            "end_to_end_ms",
            "wall_time_ms",
            "runtime_ms",
        ],
    )

    match_col = optional_col(
        df,
        ["match", "valid", "correct", "correctness"],
    )

    # Convert each accepted row to the same normalized structure used by both plots.
    rows = []

    for _, r in df.iterrows():
        cpu = float(r[cpu_col])
        gpu = float(r[gpu_col])

        label = split_label(cpu, gpu)

        if label not in SPLITS:
            continue

        # Do not visualize a timing result if the benchmark reported incorrect output.
        if match_col is not None and not truthy_match(r[match_col]):
            continue

        input_mib = float(r[input_col])
        input_gib = input_mib / 1024.0

        cpu_ms = float(r[cpu_ms_col])
        gpu_ms = float(r[gpu_ms_col])
        # Prefer measured end-to-end time. For legacy CSVs without that field,
        # the slower component is the closest overlap-aware fallback.
        total_ms = float(r[total_ms_col]) if total_ms_col is not None else max(cpu_ms, gpu_ms)

        # Component throughput is based on the logical fraction of the input assigned
        # to each processor, not on compressed physical bytes transferred.
        cpu_input_gib = input_gib * (cpu / 100.0)
        gpu_input_gib = input_gib * (gpu / 100.0)

        cpu_thr = cpu_input_gib / (cpu_ms / 1000.0) if cpu > 0 and cpu_ms > 0 else 0.0
        gpu_thr = gpu_input_gib / (gpu_ms / 1000.0) if gpu > 0 and gpu_ms > 0 else 0.0

        rows.append(
            {
                "system": system_name,
                "split": label,
                "cpu_percent": int(round(cpu)),
                "gpu_percent": int(round(gpu)),
                "input_mib": input_mib,
                "cpu_ms": cpu_ms,
                "gpu_ms": gpu_ms,
                "total_ms": total_ms,
                "cpu_thr": cpu_thr,
                "gpu_thr": gpu_thr,
            }
        )

    out = pd.DataFrame(rows)

    if out.empty:
        raise RuntimeError(f"No usable rows loaded for {system_name} from {path}")

    order = {s: i for i, s in enumerate(SPLITS)}
    out["split_order"] = out["split"].map(order)
    out = out.sort_values("split_order").reset_index(drop=True)

    return out


# Label sufficiently large stacked-bar segments without cluttering very small ones.
def label_inside(ax, rect, value: float, bottom: float, light_segment: bool, decimals: int | None = None):
    if value <= 0:
        return

    if value < 0.5:
        return

    if decimals is None:
        decimals = 2 if value < 10 else 1

    text = f"{value:.{decimals}f}"
    color = "black" if light_segment else "white"

    ax.text(
        rect.get_x() + rect.get_width() / 2,
        bottom + value / 2,
        text,
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        color=color,
    )


# Format the stacked CPU+GPU total for optional annotation above a bar.
def add_total_label(ax, x, total: float, ymax: float, metric: str):
    if total <= 0:
        return

    # Execution and throughput share the same plotting structure but use different
    # component columns, labels, and output files.
    if metric == "execution":
        text = f"Σ={total:.1f}"
    else:
        text = f"Σ={total:.2f}"

# Show the measured end-to-end runtime separately from the visual CPU+GPU stack.
# This distinction matters because CPU and GPU work can overlap in the benchmark.
def add_e2e_label(ax, x, stacked_total: float, e2e_ms: float, ymax: float):
    if e2e_ms <= 0:
        return

    ax.text(
        x,
        stacked_total + ymax * 0.012,
        f"E2E={e2e_ms:.1f} ms",
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        color="dimgray",
    )


# Estimate where CPU and GPU contributions are equal by looking for a sign change
# in CPU-GPU difference across adjacent measured split points.
def interpolate_balance_point(sub: pd.DataFrame, cpu_col: str, gpu_col: str) -> dict:
    sub = sub.set_index("split").reindex(SPLITS).reset_index()

    x = np.arange(len(SPLITS), dtype=float)
    cpu_vals = sub[cpu_col].fillna(0).to_numpy(dtype=float)
    gpu_vals = sub[gpu_col].fillna(0).to_numpy(dtype=float)
    cpu_perc = sub["cpu_percent"].fillna(0).to_numpy(dtype=float)
    gpu_perc = sub["gpu_percent"].fillna(0).to_numpy(dtype=float)

    # A zero difference is an exact measured balance point; a sign change indicates
    # that the balance lies between two neighboring measured splits.
    diff = cpu_vals - gpu_vals

    for i in range(len(SPLITS) - 1):
        d0 = diff[i]
        d1 = diff[i + 1]

        if d0 == 0:
            return {
                "x": float(x[i]),
                "cpu_percent": float(cpu_perc[i]),
                "gpu_percent": float(gpu_perc[i]),
                "cpu_value": float(cpu_vals[i]),
                "gpu_value": float(gpu_vals[i]),
                "estimated": False,
            }

        if d0 * d1 < 0:
            # Linear interpolation between measured split i and i+1.
            t = abs(d0) / (abs(d0) + abs(d1))

            x_est = x[i] + t * (x[i + 1] - x[i])
            cpu_percent_est = cpu_perc[i] + t * (cpu_perc[i + 1] - cpu_perc[i])
            gpu_percent_est = gpu_perc[i] + t * (gpu_perc[i + 1] - gpu_perc[i])
            cpu_value_est = cpu_vals[i] + t * (cpu_vals[i + 1] - cpu_vals[i])
            gpu_value_est = gpu_vals[i] + t * (gpu_vals[i + 1] - gpu_vals[i])

            return {
                "x": float(x_est),
                "cpu_percent": float(cpu_percent_est),
                "gpu_percent": float(gpu_percent_est),
                "cpu_value": float(cpu_value_est),
                "gpu_value": float(gpu_value_est),
                "estimated": True,
            }

    # Fallback: closest measured split where both sides have non-zero work if possible.
    both_active = (cpu_vals > 0) & (gpu_vals > 0)

    if np.any(both_active):
        search_diff = np.where(both_active, np.abs(diff), np.inf)
    else:
        search_diff = np.abs(diff)

    best_i = int(np.argmin(search_diff))

    return {
        "x": float(x[best_i]),
        "cpu_percent": float(cpu_perc[best_i]),
        "gpu_percent": float(gpu_perc[best_i]),
        "cpu_value": float(cpu_vals[best_i]),
        "gpu_value": float(gpu_vals[best_i]),
        "estimated": False,
    }


# Draw the CPU≈GPU balance marker and a callout without covering the bars.
def add_balance_line(
    ax,
    x_pos: float,
    system: str,
    metric: str,
    cpu_percent: float,
    gpu_percent: float,
    cpu_value: float,
    gpu_value: float,
    estimated: bool,
    ymax: float,
):
    """
    Add estimated CPU≈GPU balance line with a non-overlapping callout box.

    The box is deliberately placed in top whitespace/open area, not on top of bars.
    """
    if system == "LZ4/nvCOMP":
        line_color = "#d62728"   # red/crimson for LZ4
    else:
        line_color = "#2ca02c"   # green for FastLanes

    # Vertical balance line.
    ax.axvline(
        x=x_pos,
        color=line_color,
        linestyle="--",
        linewidth=2.2,
        alpha=0.95,
        zorder=5,
    )

    split_text = f"{cpu_percent:.0f}CPU/{gpu_percent:.0f}GPU"
    prefix = "Estimated" if estimated else "Closest measured"

    if metric == "execution":
        metric_text = "CPU≈GPU time"
        value_text = f"CPU {cpu_value:.1f} ms | GPU {gpu_value:.1f} ms"

        # Separate boxes clearly: LZ4 left/top, FastLanes right/top.
        if system == "LZ4/nvCOMP":
            box_x = 0.35
            ha = "left"
        else:
            box_x = 3.65
            ha = "right"

        box_y = ymax * 1.19
        arrow_y = ymax * 1.05

    else:
        metric_text = "CPU≈GPU throughput"
        value_text = f"CPU {cpu_value:.2f} GiB/s | GPU {gpu_value:.2f} GiB/s"

        # Throughput balance lines can lie between bars, so keep boxes in open
        # top area and separated to avoid overlap/distortion.
        if system == "LZ4/nvCOMP":
            box_x = 3.75
            ha = "right"
        else:
            box_x = 1.15
            ha = "left"

        box_y = ymax * 1.20
        arrow_y = ymax * 1.06

    ax.annotate(
        f"{prefix} {metric_text}\n{system}\n{split_text}\n{value_text}",
        xy=(x_pos, arrow_y),
        xytext=(box_x, box_y),
        ha=ha,
        va="center",
        fontsize=8.0,
        fontweight="bold",
        color="black",
        arrowprops=dict(
            arrowstyle="->",
            lw=1.15,
            color=line_color,
            shrinkA=3,
            shrinkB=3,
        ),
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            edgecolor=line_color,
            linewidth=1.2,
            alpha=0.96,
        ),
        zorder=8,
        clip_on=False,
    )


# Produce one side-by-side stacked comparison for either execution time or throughput.
def plot_stacked(df_all: pd.DataFrame, metric: str):
    assert metric in {"execution", "throughput"}

    x = np.arange(len(SPLITS), dtype=float)
    width = 0.34
    offsets = [-width / 2, width / 2]

    if metric == "execution":
        y_cpu_col = "cpu_ms"
        y_gpu_col = "gpu_ms"
        title = (
            "SPJA x40 Execution Time Breakdown (Stacked CPU + GPU)\n"
            "LZ4/nvCOMP vs FastLanes"
        )
        ylabel = "Measured Execution Time Component (ms)"
        footnote = (
            "Bars show CPU/GPU component times. E2E labels show real end-to-end time, "
            "because CPU and GPU execution may overlap."
        )
        out_path = OUT_EXEC
    else:
        y_cpu_col = "cpu_thr"
        y_gpu_col = "gpu_thr"
        title = (
            "SPJA x40 Throughput Breakdown (Stacked CPU + GPU)\n"
            "LZ4/nvCOMP vs FastLanes"
        )
        ylabel = "Component Throughput (GiB/s)"
        footnote = (
            "Bars show CPU/GPU component throughput contributions. "
            "Compression time is excluded."
        )
        out_path = OUT_THROUGHPUT

    fig, ax = plt.subplots(figsize=(16, 9))

    ymax = 0.0

    # Determine vertical scale from stacked component values before placing annotations.
    for system in SYSTEMS:
        sub = df_all[df_all["system"] == system].set_index("split").reindex(SPLITS)
        totals = sub[y_cpu_col].fillna(0).to_numpy() + sub[y_gpu_col].fillna(0).to_numpy()
        ymax = max(ymax, float(np.nanmax(totals)))

    balance_points = []

    # Draw one stacked bar per system at each split using a small horizontal offset.
    for system, offset in zip(SYSTEMS, offsets):
        sub = df_all[df_all["system"] == system].set_index("split").reindex(SPLITS)

        cpu_vals = sub[y_cpu_col].fillna(0).to_numpy(dtype=float)
        gpu_vals = sub[y_gpu_col].fillna(0).to_numpy(dtype=float)
        e2e_vals = sub["total_ms"].fillna(0).to_numpy(dtype=float)

        cpu_color, gpu_color = COLORS[system]

        b_cpu = ax.bar(
            x + offset,
            cpu_vals,
            width,
            label=f"{system} CPU",
            color=cpu_color,
            edgecolor="white",
            linewidth=0.6,
        )

        b_gpu = ax.bar(
            x + offset,
            gpu_vals,
            width,
            bottom=cpu_vals,
            label=f"{system} GPU",
            color=gpu_color,
            edgecolor="white",
            linewidth=0.6,
        )

        totals = cpu_vals + gpu_vals

        # Balance is estimated from component values, not from the end-to-end runtime.
        balance = interpolate_balance_point(
            df_all[df_all["system"] == system],
            y_cpu_col,
            y_gpu_col,
        )

        # Convert central split x to the actual side-by-side bar x.
        balance["x"] = balance["x"] + offset
        balance["system"] = system
        balance_points.append(balance)

        for i in range(len(SPLITS)):
            xpos = x[i] + offset
            stacked_total = float(totals[i])

            label_inside(ax, b_cpu[i], float(cpu_vals[i]), 0.0, light_segment=False)
            label_inside(ax, b_gpu[i], float(gpu_vals[i]), float(cpu_vals[i]), light_segment=True)
            add_total_label(ax, xpos, stacked_total, ymax, metric)

            # Add real end-to-end runtime labels on both graphs.
            add_e2e_label(ax, xpos, stacked_total, float(e2e_vals[i]), ymax)

    # Add estimated CPU≈GPU balance lines after bars.
    for bp in balance_points:
        add_balance_line(
            ax=ax,
            x_pos=bp["x"],
            system=bp["system"],
            metric=metric,
            cpu_percent=bp["cpu_percent"],
            gpu_percent=bp["gpu_percent"],
            cpu_value=bp["cpu_value"],
            gpu_value=bp["gpu_value"],
            estimated=bp["estimated"],
            ymax=ymax,
        )

    ax.set_title(title, fontsize=21, fontweight="bold", pad=12)
    ax.set_xlabel("CPU/GPU Split", fontsize=17, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=17, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(SPLITS, fontsize=12)
    ax.tick_params(axis="y", labelsize=12)

    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.set_ylim(0, ymax * 1.34)

    ax.legend(ncol=2, fontsize=10.5, loc="upper left", frameon=True)

    ax.text(
        0.5,
        -0.11,
        footnote,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10.5,
    )

    # Save publication-ready raster output while keeping annotations outside the axes visible.
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


# Load both systems once and generate the execution-time and throughput figures.
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    frames = []

    # Each source is normalized independently before concatenation.
    for system_name, path in CSV_FILES.items():
        frames.append(load_csv_for_system(system_name, path))

    df_all = pd.concat(frames, ignore_index=True)

    print("\nLoaded data:")
    print(
        df_all[
            [
                "system",
                "split",
                "cpu_ms",
                "gpu_ms",
                "total_ms",
                "cpu_thr",
                "gpu_thr",
            ]
        ].to_string(index=False)
    )

    # Generate both views from the same normalized measurement set.
    plot_stacked(df_all, "execution")
    plot_stacked(df_all, "throughput")

    print("\nSaved graphs:")
    print(f"  {OUT_EXEC}")
    print(f"  {OUT_THROUGHPUT}")


# Standard script entry point.
if __name__ == "__main__":
    main()


# Run from the repository root:
#
# python3 results/fastlanes_lz4nvcomp_dowda/graph_plotting/lz4_vs_fastlanes.py
#
# Required Python packages: numpy, pandas, matplotlib.
# The script reads the saved LZ4/nvCOMP and FastLanes CSV files above and
# rewrites only the two comparison PNG files in results/fastlanes_lz4nvcomp_dowda/graphs/.
