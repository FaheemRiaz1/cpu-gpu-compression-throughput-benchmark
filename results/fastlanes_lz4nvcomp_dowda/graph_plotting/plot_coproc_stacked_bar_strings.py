#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Input/result locations for the three string-predicate implementations.
# Keeping these paths centralized makes the plotting script reproducible from the repository root.
BASE = Path("results/fastlanes_lz4nvcomp_dowda")

CSV_FILES = {
    "LZ4/nvCOMP": Path("results/spja_workload/csv/spja_lz4_nvcomp_hybrid_split_overlap_strings_results.csv"),
    "FastLanes": BASE / "csv" / "fastlanes_spja_coproc_x40_strings_results.csv",
    "DPF-inspired fused": BASE / "csv" / "dpf_fused_spja_x40_common_timing_strings_results.csv",
}

OUT_DIR = BASE / "graphs"
OUT_EXEC = OUT_DIR / "execution_time_strings.png"
OUT_THROUGHPUT = OUT_DIR / "throughput_time_strings.png"

# Use one fixed split order so every system is aligned on the same x-axis positions.
SPLITS = ["100CPU/0GPU", "75CPU/25GPU", "50CPU/50GPU", "25CPU/75GPU", "0CPU/100GPU"]
SYSTEMS = ["LZ4/nvCOMP", "FastLanes", "DPF-inspired fused"]

# Dark/light pairs distinguish the CPU and GPU contribution of each system.
COLORS = {
    "LZ4/nvCOMP": ("#1f77b4", "#86c5f4"),
    "FastLanes": ("#ff7f0e", "#ffc36b"),
    "DPF-inspired fused": ("#2ca02c", "#98df8a"),
}


# Normalize CSV headers because the benchmark writers use slightly different naming conventions.
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


# Resolve one required logical field from the accepted header variants.
def find_col(df: pd.DataFrame, candidates: Iterable[str], label: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"\nCould not find column for {label}.\n"
        f"Tried: {list(candidates)}\n"
        f"Available columns: {list(df.columns)}\n"
    )


# Accept the correctness markers emitted by the different benchmark CSV formats.
def truthy_match(value) -> bool:
    s = str(value).strip().upper()
    return s in {"YES", "Y", "TRUE", "1", "MATCH", "VALID"}


# Convert numeric split percentages into the common label used by all plots.
def split_label(cpu_percent: float, gpu_percent: float) -> str:
    return f"{int(round(cpu_percent))}CPU/{int(round(gpu_percent))}GPU"


# Load one benchmark CSV into a common schema used by the comparison plots.
# Rows outside the five target splits, or rows marked incorrect, are excluded.
def load_csv_for_system(system_name: str, path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV for {system_name}: {path}")

    df = normalize_columns(pd.read_csv(path))

    cpu_col = find_col(df, ["cpu_percent", "cpu_pct", "cpu"], f"{system_name} CPU percent")
    gpu_col = find_col(df, ["gpu_percent", "gpu_pct", "gpu"], f"{system_name} GPU percent")

    input_col = find_col(
        df,
        ["input_mib", "mib", "query_input_size_mib", "query_input_mib", "input_size_mib"],
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

    # E2E is not used for stacked height, but we keep it for debugging/printing.
    e2e_col = None
    for c in [
        "e2e_ms",
        "total_ms_avg",
        "total_ms",
        "elapsed_ms",
        "diff_ms_avg",
        "diff_ms",
        "end_to_end_ms",
    ]:
        if c in df.columns:
            e2e_col = c
            break

    match_col = None
    for c in ["match", "valid", "correct", "correctness"]:
        if c in df.columns:
            match_col = c
            break

    rows = []
    for _, r in df.iterrows():
        cpu = float(r[cpu_col])
        gpu = float(r[gpu_col])

        label = split_label(cpu, gpu)
        if label not in SPLITS:
            continue

        if match_col is not None and not truthy_match(r[match_col]):
            continue

        input_mib = float(r[input_col])
        input_gib = input_mib / 1024.0

        cpu_ms = float(r[cpu_ms_col])
        gpu_ms = float(r[gpu_ms_col])
        e2e_ms = float(r[e2e_col]) if e2e_col is not None else max(cpu_ms, gpu_ms)

        # Component throughput is based only on the logical input fraction assigned to that processor.
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
                "e2e_ms": e2e_ms,
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


# Place component values inside visible stacked-bar segments without cluttering very small bars.
def label_inside(ax, rect, value: float, bottom: float, light_segment: bool, force_decimals: int | None = None):
    if value <= 0:
        return

    # Avoid unreadable labels for extremely tiny bars.
    if value < 0.5:
        return

    decimals = force_decimals if force_decimals is not None else (2 if value < 10 else 1)
    text = f"{value:.{decimals}f}"

    color = "black" if light_segment else "white"
    ax.text(
        rect.get_x() + rect.get_width() / 2,
        bottom + value / 2,
        text,
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        color=color,
    )


# Draw one grouped stacked chart for either execution-time components or component throughput.
# Stacking is visual only: CPU and GPU work may overlap, so execution-stack height is not E2E time.
def plot_stacked(df_all: pd.DataFrame, metric: str):
    assert metric in {"execution", "throughput"}

    x = np.arange(len(SPLITS), dtype=float)
    width = 0.22
    offsets = [-width, 0.0, width]

    if metric == "execution":
        y_cpu_col = "cpu_ms"
        y_gpu_col = "gpu_ms"
        title = (
            "SPJA x40 Dictionary-Encoded String Predicate: Execution Time Breakdown\n"
            "Query: quantity > 25 AND customer_mktsegment = BUILDING"
        )
        ylabel = "Measured Execution Time Component (ms)"
        footnote = (
            "Compression time is excluded. String predicate is evaluated via dictionary code BUILDING -> 1. "
            "CPU and GPU component times are stacked only for visual comparison; stacked height is not end-to-end time."
        )
        out_path = OUT_EXEC
    else:
        y_cpu_col = "cpu_thr"
        y_gpu_col = "gpu_thr"
        title = (
            "SPJA x40 Dictionary-Encoded String Predicate: Throughput Breakdown\n"
            "Query: quantity > 25 AND customer_mktsegment = BUILDING"
        )
        ylabel = "Component Throughput (GiB/s)"
        footnote = (
            "Component throughput uses CPU-owned and GPU-owned input fractions separately. "
            "String predicate is dictionary-encoded as BUILDING -> 1; stacked segments are not an additive total."
        )
        out_path = OUT_THROUGHPUT

    # Three narrow bars per split keep LZ4/nvCOMP, FastLanes, and DPF directly comparable.
    fig, ax = plt.subplots(figsize=(18, 10))

    # Compute ymax first for label padding.
    ymax = 0.0
    for system in SYSTEMS:
        sub = df_all[df_all["system"] == system].set_index("split").reindex(SPLITS)
        stack_heights = sub[y_cpu_col].fillna(0).to_numpy() + sub[y_gpu_col].fillna(0).to_numpy()
        ymax = max(ymax, float(np.nanmax(stack_heights)))

    # Reindex each system to the canonical split order before plotting its CPU/GPU components.
    for system, offset in zip(SYSTEMS, offsets):
        sub = df_all[df_all["system"] == system].set_index("split").reindex(SPLITS)
        cpu_vals = sub[y_cpu_col].fillna(0).to_numpy()
        gpu_vals = sub[y_gpu_col].fillna(0).to_numpy()

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


        for i in range(len(SPLITS)):
            label_inside(ax, b_cpu[i], float(cpu_vals[i]), 0.0, light_segment=False)
            label_inside(ax, b_gpu[i], float(gpu_vals[i]), float(cpu_vals[i]), light_segment=True)

    ax.set_title(title, fontsize=22, fontweight="bold", pad=12)
    ax.set_xlabel("CPU/GPU Split", fontsize=18, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=18, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(SPLITS, fontsize=13)
    ax.tick_params(axis="y", labelsize=12)

    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.set_ylim(0, ymax * 1.16)

    ax.legend(ncol=3, fontsize=10.5, loc="upper left", frameon=True)

    ax.text(
        0.5,
        -0.12,
        footnote,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=11,
    )

    fig.tight_layout()
    # Write publication-quality PNG output into the repository results directory.
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


# Load all three systems once, then generate both comparison figures from the same normalized data.
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    frames = []
    for system_name, path in CSV_FILES.items():
        frames.append(load_csv_for_system(system_name, path))

    df_all = pd.concat(frames, ignore_index=True)

    print("\nLoaded string-predicate data:")
    print(
        df_all[
            [
                "system",
                "split",
                "cpu_ms",
                "gpu_ms",
                "e2e_ms",
                "cpu_thr",
                "gpu_thr",
            ]
        ].to_string(index=False)
    )

    plot_stacked(df_all, "execution")
    plot_stacked(df_all, "throughput")

    print("\nSaved graphs:")
    print(f"  {OUT_EXEC}")
    print(f"  {OUT_THROUGHPUT}")


if __name__ == "__main__":
    main()

# Run from the repository root:
# python3 results/fastlanes_lz4nvcomp_dowda/graph_plotting/plot_coproc_stacked_bar_strings.py
# The script reads the three CSV files above and rewrites the two PNG outputs in OUT_DIR.
