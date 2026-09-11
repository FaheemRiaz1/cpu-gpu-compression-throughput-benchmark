#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE = Path("results/fastlanes_lz4nvcomp_dowda")

CSV_FILES = {
    "LZ4/nvCOMP":  Path("results/spja_workload/csv/spja_lz4_nvcomp_hybrid_split_overlap_results.csv"),
    "FastLanes": BASE / "csv" / "fastlanes_spja_coproc_x40_results.csv",
    "DPF-inspired fused": BASE / "csv" / "dpf_fused_spja_x40_common_timing_results.csv",
}

OUT_DIR = BASE / "graphs"
OUT_EXEC = OUT_DIR / "execution_time.png"
OUT_THROUGHPUT = OUT_DIR / "throughput_time.png"

SPLITS = ["100CPU/0GPU", "75CPU/25GPU", "50CPU/50GPU", "25CPU/75GPU", "0CPU/100GPU"]
SYSTEMS = ["LZ4/nvCOMP", "FastLanes", "DPF-inspired fused"]

COLORS = {
    "LZ4/nvCOMP": ("#1f77b4", "#86c5f4"),
    "FastLanes": ("#ff7f0e", "#ffc36b"),
    "DPF-inspired fused": ("#2ca02c", "#98df8a"),
}


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


def find_col(df: pd.DataFrame, candidates: Iterable[str], label: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"\nCould not find column for {label}.\n"
        f"Tried: {list(candidates)}\n"
        f"Available columns: {list(df.columns)}\n"
    )


def truthy_match(value) -> bool:
    s = str(value).strip().upper()
    return s in {"YES", "Y", "TRUE", "1", "MATCH", "VALID"}


def split_label(cpu_percent: float, gpu_percent: float) -> str:
    return f"{int(round(cpu_percent))}CPU/{int(round(gpu_percent))}GPU"


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


def add_total_label(ax, x, total: float, ymax: float):
    if total <= 0:
        return
    text = f"Σ={total:.2f}" if total < 100 else f"Σ={total:.1f}"
    ax.text(
        x,
        total + ymax * 0.012,
        text,
        ha="center",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        color="black",
    )


def plot_stacked(df_all: pd.DataFrame, metric: str):
    assert metric in {"execution", "throughput"}

    x = np.arange(len(SPLITS), dtype=float)
    width = 0.22
    offsets = [-width, 0.0, width]

    if metric == "execution":
        y_cpu_col = "cpu_ms"
        y_gpu_col = "gpu_ms"
        title = (
            "SPJA x40 Execution Time Breakdown (Stacked CPU + GPU)\n"
            "Same x40 data and same SPJA query for all three systems"
        )
        ylabel = "Measured Execution Time Component (ms)"
        footnote = (
            "Compression time is excluded. Each bar is stacked from CPU-owned work time "
            "and GPU-owned work time for that split."
        )
        out_path = OUT_EXEC
    else:
        y_cpu_col = "cpu_thr"
        y_gpu_col = "gpu_thr"
        title = (
            "SPJA x40 Throughput Breakdown (Stacked CPU + GPU)\n"
            "Same x40 data and same SPJA query for all three systems"
        )
        ylabel = "Component Throughput (GiB/s)"
        footnote = (
            "Component throughput is computed from the CPU-owned and GPU-owned input fractions separately, "
            "then shown as stacked breakdown values."
        )
        out_path = OUT_THROUGHPUT

    fig, ax = plt.subplots(figsize=(18, 10))

    # Compute ymax first for label padding.
    ymax = 0.0
    for system in SYSTEMS:
        sub = df_all[df_all["system"] == system].set_index("split").reindex(SPLITS)
        totals = sub[y_cpu_col].fillna(0).to_numpy() + sub[y_gpu_col].fillna(0).to_numpy()
        ymax = max(ymax, float(np.nanmax(totals)))

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

        totals = cpu_vals + gpu_vals

        for i in range(len(SPLITS)):
            label_inside(ax, b_cpu[i], float(cpu_vals[i]), 0.0, light_segment=False)
            label_inside(ax, b_gpu[i], float(gpu_vals[i]), float(cpu_vals[i]), light_segment=True)
            add_total_label(ax, x[i] + offset, float(totals[i]), ymax)

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
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    frames = []
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
