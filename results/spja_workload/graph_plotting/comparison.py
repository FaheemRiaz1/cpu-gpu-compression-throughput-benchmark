#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_COMPRESSION_CSV = Path(
    "results/spja_workload/csv/"
    "spja_compressed_vs_uncompressed_results.csv"
)

DEFAULT_H2D_CSV = Path(
    "results/spja_workload/csv/"
    "spja_h2d_vs_no_h2d_results.csv"
)

DEFAULT_OUTPUT_DIR = Path(
    "results/spja_workload/graphs/key_findings"
)

BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
LIGHT_ORANGE = "#fdb462"

REQUIRED_COLUMNS = {
    "Experiment",
    "Mode",
    "Input_MiB",
    "CPU_Percent",
    "GPU_Percent",
    "Assignment_Trials",
    "Timed_Runs_Per_Assignment",
    "CPU_Rows_Avg",
    "GPU_Rows_Avg",
    "Timed_H2D_Bytes_Avg",
    "Offline_Compression_ms_Excluded",
    "E2E_ms_Avg",
    "E2E_ms_StdDev",
    "Effective_GiBps_Avg",
    "Effective_GiBps_StdDev",
    "CPU_Path_ms_Avg",
    "CPU_Path_ms_StdDev",
    "GPU_Path_ms_Avg",
    "GPU_Path_ms_StdDev",
    "Final_Result",
    "Reference_Result",
    "Valid",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate four independent compression/H2D graphs."
    )
    parser.add_argument(
        "--compression-csv",
        type=Path,
        default=DEFAULT_COMPRESSION_CSV,
    )
    parser.add_argument(
        "--h2d-csv",
        type=Path,
        default=DEFAULT_H2D_CSV,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
        }
    )


def load_experiment(
    path: Path,
    expected_experiment: str,
    expected_modes: tuple[str, str],
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    dataframe = pd.read_csv(path)
    dataframe.columns = dataframe.columns.str.strip()

    missing = REQUIRED_COLUMNS.difference(dataframe.columns)
    if missing:
        raise ValueError(
            f"{path} is missing columns: " + ", ".join(sorted(missing))
        )

    numeric_columns = sorted(REQUIRED_COLUMNS.difference({"Experiment", "Mode", "Valid"}))
    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")

    bad_rows = dataframe.index[
        dataframe[numeric_columns].isna().any(axis=1)
    ].tolist()
    if bad_rows:
        raise ValueError(f"Non-numeric or missing values in {path}, rows {bad_rows}")

    dataframe["Experiment"] = dataframe["Experiment"].astype(str).str.strip()
    dataframe["Mode"] = dataframe["Mode"].astype(str).str.strip()
    dataframe["Valid"] = dataframe["Valid"].astype(str).str.strip().str.upper()

    if set(dataframe["Experiment"].unique()) != {expected_experiment}:
        raise ValueError(
            f"{path} does not contain only experiment {expected_experiment}."
        )

    if set(dataframe["Mode"].unique()) != set(expected_modes):
        raise ValueError(
            f"{path} modes are {sorted(dataframe['Mode'].unique())}; "
            f"expected {sorted(expected_modes)}."
        )

    if not dataframe["Valid"].isin({"YES", "TRUE", "1"}).all():
        raise ValueError(f"At least one row in {path} is not valid.")

    if not np.allclose(
        dataframe["Final_Result"].to_numpy(float),
        dataframe["Reference_Result"].to_numpy(float),
    ):
        raise ValueError(f"At least one result in {path} mismatches the reference.")

    if not np.allclose(
        (dataframe["CPU_Percent"] + dataframe["GPU_Percent"]).to_numpy(float),
        100.0,
    ):
        raise ValueError(f"At least one split in {path} does not sum to 100.")

    duplicates = dataframe.duplicated(
        subset=["Mode", "CPU_Percent", "GPU_Percent"]
    )
    if duplicates.any():
        raise ValueError(f"Duplicate mode/split rows found in {path}.")

    expected_gpu_splits = [0, 25, 50, 75, 100]
    for mode in expected_modes:
        splits = (
            dataframe.loc[dataframe["Mode"] == mode, "GPU_Percent"]
            .astype(int)
            .sort_values()
            .tolist()
        )
        if splits != expected_gpu_splits:
            raise ValueError(
                f"Mode {mode} has GPU splits {splits}; "
                f"expected {expected_gpu_splits}."
            )

    # Verify throughput arithmetic: logical GiB / measured E2E seconds.
    expected_throughput = (
        dataframe["Input_MiB"].to_numpy(float) / 1024.0
    ) / (dataframe["E2E_ms_Avg"].to_numpy(float) / 1000.0)

    reported_throughput = dataframe["Effective_GiBps_Avg"].to_numpy(float)
    tolerance = np.maximum(0.005, np.abs(expected_throughput) * 0.005)
    if np.any(np.abs(reported_throughput - expected_throughput) > tolerance):
        raise ValueError(
            f"Throughput in {path} is inconsistent with Input_MiB/E2E_ms_Avg."
        )

    return dataframe.sort_values(["Mode", "GPU_Percent"]).reset_index(drop=True)


def mode_frame(dataframe: pd.DataFrame, mode: str) -> pd.DataFrame:
    return (
        dataframe[dataframe["Mode"] == mode]
        .sort_values("GPU_Percent")
        .reset_index(drop=True)
    )


def split_labels(frame: pd.DataFrame) -> list[str]:
    return [
        f"{int(cpu)}/{int(gpu)}"
        for cpu, gpu in zip(
            frame["CPU_Percent"],
            frame["GPU_Percent"],
            strict=True,
        )
    ]


def save_figure(
    figure: plt.Figure,
    output_directory: Path,
    filename: str,
    dpi: int,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_directory / f"{filename}.png", dpi=dpi)
    figure.savefig(output_directory / f"{filename}.pdf")
    plt.close(figure)


def add_labels(
    axis: plt.Axes,
    bars,
    values: np.ndarray,
    decimals: int,
) -> None:
    offset = axis.get_ylim()[1] * 0.018
    for bar, value in zip(bars, values, strict=True):
        if value <= 0.0:
            continue
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + offset,
            f"{value:,.{decimals}f}",
            ha="center",
            va="bottom",
            fontsize=8.7,
            fontweight="bold",
            clip_on=False,
        )


def grouped_plot(
    *,
    labels: list[str],
    left_values: np.ndarray,
    right_values: np.ndarray,
    left_std: np.ndarray,
    right_std: np.ndarray,
    left_label: str,
    right_label: str,
    left_color: str,
    right_color: str,
    title: str,
    ylabel: str,
    footer: str,
    filename: str,
    decimals: int,
    output_directory: Path,
    dpi: int,
) -> None:
    x = np.arange(len(labels), dtype=float)
    width = 0.36
    figure, axis = plt.subplots(figsize=(11.8, 6.8))

    left_bars = axis.bar(
        x - width / 2.0,
        left_values,
        width,
        yerr=left_std,
        capsize=3,
        color=left_color,
        edgecolor="black",
        linewidth=0.55,
        label=left_label,
        zorder=3,
    )

    right_bars = axis.bar(
        x + width / 2.0,
        right_values,
        width,
        yerr=right_std,
        capsize=3,
        color=right_color,
        edgecolor="black",
        linewidth=0.55,
        label=right_label,
        zorder=3,
    )

    highest = max(
        float(np.max(left_values + left_std)),
        float(np.max(right_values + right_std)),
    )
    axis.set_ylim(0.0, highest * 1.24 if highest > 0.0 else 1.0)
    axis.set_xticks(x)
    axis.set_xticklabels(labels)
    axis.set_xlabel("CPU/GPU Split (%)")
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35, zorder=0)

    add_labels(axis, left_bars, left_values, decimals)
    add_labels(axis, right_bars, right_values, decimals)

    handles, legend_labels = axis.get_legend_handles_labels()
    figure.suptitle(title, y=0.975, fontsize=15)
    figure.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=2,
        frameon=False,
    )
    figure.text(0.5, 0.035, footer, ha="center", va="center", fontsize=9)
    figure.subplots_adjust(top=0.80, bottom=0.17, left=0.105, right=0.985)

    save_figure(figure, output_directory, filename, dpi)


def plot_compression_experiment(
    dataframe: pd.DataFrame,
    output_directory: Path,
    dpi: int,
) -> pd.DataFrame:
    uncompressed = mode_frame(dataframe, "UNCOMPRESSED")
    compressed = mode_frame(dataframe, "COMPRESSED_LZ4_NVCOMP")
    labels = split_labels(uncompressed)

    grouped_plot(
        labels=labels,
        left_values=uncompressed["E2E_ms_Avg"].to_numpy(float),
        right_values=compressed["E2E_ms_Avg"].to_numpy(float),
        left_std=uncompressed["E2E_ms_StdDev"].to_numpy(float),
        right_std=compressed["E2E_ms_StdDev"].to_numpy(float),
        left_label="Uncompressed pipeline",
        right_label="LZ4/nvCOMP compressed pipeline",
        left_color=BLUE,
        right_color=ORANGE,
        title="Compressed vs Uncompressed SPJA: End-to-End Runtime",
        ylabel="System End-to-End Runtime (ms)",
        footer=(
            "Independent compression experiment: H2D is timed in both modes. "
            "Offline compression and allocation are excluded."
        ),
        filename="kf1_compression_execution_time_cpu_gpu",
        decimals=1,
        output_directory=output_directory,
        dpi=dpi,
    )

    grouped_plot(
        labels=labels,
        left_values=uncompressed["Effective_GiBps_Avg"].to_numpy(float),
        right_values=compressed["Effective_GiBps_Avg"].to_numpy(float),
        left_std=uncompressed["Effective_GiBps_StdDev"].to_numpy(float),
        right_std=compressed["Effective_GiBps_StdDev"].to_numpy(float),
        left_label="Uncompressed pipeline",
        right_label="LZ4/nvCOMP compressed pipeline",
        left_color=BLUE,
        right_color=ORANGE,
        title="Compressed vs Uncompressed SPJA: Effective Throughput",
        ylabel="System Effective Throughput (GiB/s)",
        footer=(
            "Independent compression experiment: throughput uses the same "
            "logical input size divided by each measured E2E runtime."
        ),
        filename="kf1_compression_throughput_cpu_gpu",
        decimals=3,
        output_directory=output_directory,
        dpi=dpi,
    )

    return pd.DataFrame(
        {
            "CPU_GPU_Split": labels,
            "Uncompressed_E2E_ms": uncompressed["E2E_ms_Avg"],
            "Compressed_E2E_ms": compressed["E2E_ms_Avg"],
            "Uncompressed_GiBps": uncompressed["Effective_GiBps_Avg"],
            "Compressed_GiBps": compressed["Effective_GiBps_Avg"],
        }
    )


def plot_h2d_experiment(
    dataframe: pd.DataFrame,
    output_directory: Path,
    dpi: int,
) -> pd.DataFrame:
    with_h2d = mode_frame(dataframe, "WITH_H2D")
    without_h2d = mode_frame(dataframe, "WITHOUT_H2D")
    labels = split_labels(with_h2d)

    # Do not alter values. Warn when noise reverses the expected direction.
    active_gpu = with_h2d["GPU_Percent"].to_numpy(float) > 0.0
    time_difference = (
        without_h2d["E2E_ms_Avg"].to_numpy(float)
        - with_h2d["E2E_ms_Avg"].to_numpy(float)
    )
    material_tolerance = np.maximum(
        2.0,
        with_h2d["E2E_ms_Avg"].to_numpy(float) * 0.02,
    )
    material_reversal = (time_difference > material_tolerance) & active_gpu
    if np.any(material_reversal):
        splits = [
            labels[index]
            for index in np.where(material_reversal)[0]
        ]
        print(
            "WARNING: WITHOUT_H2D is materially slower at splits "
            + ", ".join(splits)
            + ". Review/re-run those measurements; values were not modified."
        )

    grouped_plot(
        labels=labels,
        left_values=with_h2d["E2E_ms_Avg"].to_numpy(float),
        right_values=without_h2d["E2E_ms_Avg"].to_numpy(float),
        left_std=with_h2d["E2E_ms_StdDev"].to_numpy(float),
        right_std=without_h2d["E2E_ms_StdDev"].to_numpy(float),
        left_label="Compressed pipeline with timed H2D",
        right_label="Compressed pipeline without timed H2D",
        left_color=ORANGE,
        right_color=LIGHT_ORANGE,
        title="Compressed SPJA H2D Ablation: End-to-End Runtime",
        ylabel="System End-to-End Runtime (ms)",
        footer=(
            "Independent H2D experiment: both modes use LZ4/nvCOMP. "
            "WITHOUT_H2D preloads only the GPU-owned compressed bytes; "
            "CPU work, nvCOMP, SPJA, reduction, synchronization, and scalar D2H remain timed."
        ),
        filename="kf2_h2d_execution_time_cpu_gpu",
        decimals=1,
        output_directory=output_directory,
        dpi=dpi,
    )

    grouped_plot(
        labels=labels,
        left_values=with_h2d["Effective_GiBps_Avg"].to_numpy(float),
        right_values=without_h2d["Effective_GiBps_Avg"].to_numpy(float),
        left_std=with_h2d["Effective_GiBps_StdDev"].to_numpy(float),
        right_std=without_h2d["Effective_GiBps_StdDev"].to_numpy(float),
        left_label="Compressed pipeline with timed H2D",
        right_label="Compressed pipeline without timed H2D",
        left_color=ORANGE,
        right_color=LIGHT_ORANGE,
        title="Compressed SPJA H2D Ablation: Effective Throughput",
        ylabel="System Effective Throughput (GiB/s)",
        footer=(
            "Independent H2D experiment: both throughputs use the same logical input size "
            "and their separately measured E2E runtimes."
        ),
        filename="kf2_h2d_throughput_cpu_gpu",
        decimals=3,
        output_directory=output_directory,
        dpi=dpi,
    )

    return pd.DataFrame(
        {
            "CPU_GPU_Split": labels,
            "With_H2D_E2E_ms": with_h2d["E2E_ms_Avg"],
            "Without_H2D_E2E_ms": without_h2d["E2E_ms_Avg"],
            "With_H2D_GiBps": with_h2d["Effective_GiBps_Avg"],
            "Without_H2D_GiBps": without_h2d["Effective_GiBps_Avg"],
        }
    )


def main() -> int:
    arguments = parse_args()
    configure_matplotlib()

    try:
        compression_data = load_experiment(
            arguments.compression_csv,
            "COMPRESSION",
            ("UNCOMPRESSED", "COMPRESSED_LZ4_NVCOMP"),
        )
        h2d_data = load_experiment(
            arguments.h2d_csv,
            "H2D",
            ("WITH_H2D", "WITHOUT_H2D"),
        )

        arguments.output_dir.mkdir(parents=True, exist_ok=True)

        compression_audit = plot_compression_experiment(
            compression_data,
            arguments.output_dir,
            arguments.dpi,
        )
        h2d_audit = plot_h2d_experiment(
            h2d_data,
            arguments.output_dir,
            arguments.dpi,
        )

        compression_audit.to_csv(
            arguments.output_dir / "compression_plot_audit.csv",
            index=False,
        )
        h2d_audit.to_csv(
            arguments.output_dir / "h2d_plot_audit.csv",
            index=False,
        )

    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"Error: {error}")
        return 1

    print("Generated four figures in:")
    print(f"  {arguments.output_dir}")
    print("Generated audit CSVs:")
    print(f"  {arguments.output_dir / 'compression_plot_audit.csv'}")
    print(f"  {arguments.output_dir / 'h2d_plot_audit.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
