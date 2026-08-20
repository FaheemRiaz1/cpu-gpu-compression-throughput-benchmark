# CPU–GPU Compression Throughput Benchmark

This repository contains the CUDA/C++ benchmark implementations, scripts, selected results, and documentation used for a Master's thesis on compressed CPU–GPU analytical query processing.

The project started with basic CPU/GPU memory-transfer measurements and was extended into a TPC-H-derived columnar benchmark covering:

- CPU LZ4 compression/decompression
- GPU nvCOMP LZ4 decompression
- CPU-only, GPU-only, and hybrid CPU/GPU execution
- SPJA-style analytical query processing
- FastLanes-GPU comparison baselines
- fused decode-and-query experimental baselines
- string-workload variants
- throughput, compression, overlap, and correctness evaluation

The main experimental direction is:

```text
TPC-H-derived columnar data
        ↓
chunk-wise compression / encoding
        ↓
CPU and/or GPU processing
        ↓
decompression / decoding
        ↓
SPJA-style query execution
        ↓
correctness validation
        ↓
throughput and timing measurements
```

---

## 1. Project Goals

The main goals of this project are:

1. Measure baseline CPU memory-copy and CPU–GPU transfer throughput.
2. Evaluate compressed data processing with LZ4 and nvCOMP.
3. Compare CPU-only, GPU-only, and hybrid CPU/GPU execution.
4. Study how compression changes effective end-to-end query throughput.
5. Evaluate CPU/GPU workload splits and overlap behavior.
6. Compare general-purpose LZ4/nvCOMP processing with FastLanes-GPU.
7. Evaluate a fused decode-and-query processing design.
8. Extend the comparison to string-aware SPJA workloads.
9. Preserve correctness while comparing different compressed-processing designs.

---

## 2. Repository Structure

Important project folders and files:

```text
.
├── src/
│   ├── benchmark/
│   │   ├── baseline_benchmark.cu
│   │   ├── averaged_benchmark.cu
│   │   ├── simple_pipeline.cu
│   │   ├── parallel_cpu_lz4_nvcomp_full_pipeline.cu
│   │   ├── nvcomp_lz4_vs_fastlane.cu
│   │   ├── spja_lz4_nvcomp_split_overlap.cu
│   │   ├── spja_lz4_nvcomp_split_overlap_strings.cu
│   │   ├── spja_lz4_nvcomp_four_modes.cu
│   │   ├── spja_dpf_fused_x40.cu
│   │   └── spja_dpf_fused_x40_strings.cu
│   └── tools/
│       └── generate_customer_mktsegment_code.py
│
├── external/
│   └── baseline/
│       └── fastlanes_gpu/
│           ├── coproc_fastlanes.cu
│           ├── spja_coproc_fastlanes.cu
│           └── spja_coproc_fastlanes_strings.cu
│
├── scripts/
│   ├── tpch_columnar/
│   │   ├── convert_dbgen_tbl_to_bin.py
│   │   ├── generate_tpch_columnar.py
│   │   ├── check_columnar_lz4_ratio.cpp
│   │   └── check_columnar_lz4_delta_ratio.cpp
│   ├── run_fastlanes_coproc_quantity.sh
│   ├── run_fastlanes_spja_x25.sh
│   ├── run_fastlanes_spja_x40.sh
│   └── run_fastlanes_spja_x40_strings.sh
│
├── docs/
│   ├── averaged_benchmark_notes.txt
│   ├── benchmark_notes.txt
│   ├── fastlanes_gpu.txt
│   ├── fused_dpf_goda.txt
│   ├── lz4_nvcomp.txt
│   ├── rle_simple_pipeline.txt
│   ├── spja.txt
│   └── tpch_data_generation.txt
│
├── results/
│   ├── baseline_comparison_graphs/
│   ├── baseline_fastlanes_gpu/
│   ├── fastlanes_lz4nvcomp_dowda/
│   ├── lz4_nvcomp_pipeline/
│   ├── sf1_backup_csv/
│   ├── sf1_backup_graphs/
│   ├── sf10_csv/
│   ├── sf10_graphs/
│   ├── sf20_graphs/
│   ├── simple_pipeline/
│   └── spja_workload/
│
├── data/
├── tools/
├── setup_dirs.sh
├── thesis_environment_info.sh
├── .gitignore
└── README.md
```

Generated data, build directories, external tools, and selected large intermediate outputs are intentionally excluded from Git.

---

## 3. Requirements

The project is intended for a Linux machine with an NVIDIA GPU.

Required software and libraries include:

- CUDA Toolkit with `nvcc`
- NVIDIA GPU and compatible CUDA driver
- nvCOMP
- LZ4 development library
- C++17-compatible compiler
- Python 3
- pandas
- matplotlib
- NumPy

### 3.1 Install System Dependencies

On Ubuntu/Debian:

```bash
sudo apt update
sudo apt install -y build-essential cmake python3 python3-pip liblz4-dev
```

Install Python packages:

```bash
python3 -m pip install --user pandas matplotlib numpy
```

Or use a virtual environment:

```bash
python3 -m venv benchmark_env
source benchmark_env/bin/activate
python3 -m pip install pandas matplotlib numpy
```

### 3.2 nvCOMP

The thesis environment used the Python-distributed nvCOMP package:

```bash
python3 -m venv nvcomp_env
source nvcomp_env/bin/activate
pip install nvidia-nvcomp-cu12
```

Example runtime library path:

```bash
export LD_LIBRARY_PATH=~/nvcomp_env/lib/python3.12/site-packages/nvidia/libnvcomp/lib64:$LD_LIBRARY_PATH
```

If nvCOMP is installed elsewhere, update the include, library, and runtime paths accordingly.

Check CUDA:

```bash
nvcc --version
```

---

## 4. Basic Baseline Benchmarks

The original baseline programs are:

```text
src/benchmark/baseline_benchmark.cu
src/benchmark/averaged_benchmark.cu
```

Compile:

```bash
mkdir -p bin

nvcc src/benchmark/baseline_benchmark.cu \
  -o bin/baseline_benchmark

nvcc src/benchmark/averaged_benchmark.cu \
  -o bin/averaged_benchmark
```

Run:

```bash
./bin/baseline_benchmark
./bin/averaged_benchmark
```

These programs measure basic CPU memory and CPU–GPU transfer behavior before compression is introduced.

---

## 5. LZ4 + nvCOMP Single-Pipeline Benchmark

The single-file compressed pipeline benchmark is:

```text
src/benchmark/parallel_cpu_lz4_nvcomp_full_pipeline.cu
```

Pipeline:

```text
CPU LZ4 Compression
        ↓
H2D Compressed Transfer
        ↓
GPU nvCOMP Decompression
        ↓
GPU Compute
        ↓
D2H
```

Compile from the repository root:

```bash
mkdir -p bin

nvcc src/benchmark/parallel_cpu_lz4_nvcomp_full_pipeline.cu \
  -o bin/final_pipeline \
  -I ~/nvcomp_env/lib/python3.12/site-packages/nvidia/libnvcomp/include \
  -L ~/nvcomp_env/lib/python3.12/site-packages/nvidia/libnvcomp/lib64 \
  -lnvcomp -llz4
```

Run:

```bash
./bin/final_pipeline
```

Selected result files are stored under:

```text
results/lz4_nvcomp_pipeline/
```

For additional details, see:

```text
docs/lz4_nvcomp.txt
```

---

## 6. TPC-H Data Preparation

The SPJA benchmarks operate on TPC-H-derived binary column files.

Expected generated column data is stored under:

```text
data/tpch_columnar/
```

Examples include:

```text
orderkey_sf1.bin
quantity_sf1.bin
extendedprice_sf1.bin
order_custkey_sf1.bin
customer_nation_sf1.bin
partkey_sf1.bin
part_category_sf1.bin
part_factor_sf1.bin
```

### 6.1 Generate TPC-H Tables

TPC-H `.tbl` files are generated using `tpch-dbgen`.

A typical command for scale factor 1 is:

```bash
./dbgen -s 1
```

A typical local source layout is:

```text
data/tpch_real/sf1/
```

For larger scale factors, use corresponding folders such as:

```text
data/tpch_real/sf10/
```

### 6.2 Convert TPC-H Tables to Binary Columns

Run:

```bash
python3 scripts/tpch_columnar/convert_dbgen_tbl_to_bin.py
```

Then inspect the generated data:

```bash
ls -lh data/tpch_columnar/
```

Additional preparation utilities are available under:

```text
scripts/tpch_columnar/
```

For a more complete description, see:

```text
docs/tpch_data_generation.txt
```

---

## 7. Main LZ4 + nvCOMP SPJA Benchmark

The main compressed CPU/GPU SPJA implementation is:

```text
src/benchmark/spja_lz4_nvcomp_split_overlap.cu
```

The benchmark includes:

- TPC-H-derived column loading
- chunk-wise LZ4_HC compression
- CPU LZ4 decompression
- GPU nvCOMP LZ4 decompression
- SPJA-style query execution
- CPU-only execution
- GPU-only execution
- hybrid CPU/GPU execution
- deterministic workload splitting
- correctness validation
- multiple timing trials
- throughput reporting
- compression-statistics output

### 7.1 Compile

```bash
mkdir -p bin

nvcc -std=c++17 -O3 \
  src/benchmark/spja_lz4_nvcomp_split_overlap.cu \
  -o bin/spja_lz4_nvcomp_split_overlap \
  -llz4 \
  -lnvcomp
```

If nvCOMP is not installed in a default system location:

```bash
nvcc -std=c++17 -O3 \
  -I ~/nvcomp_env/lib/python3.12/site-packages/nvidia/libnvcomp/include \
  -L ~/nvcomp_env/lib/python3.12/site-packages/nvidia/libnvcomp/lib64 \
  src/benchmark/spja_lz4_nvcomp_split_overlap.cu \
  -o bin/spja_lz4_nvcomp_split_overlap \
  -llz4 \
  -lnvcomp
```

### 7.2 Run

```bash
export LD_LIBRARY_PATH=~/nvcomp_env/lib/python3.12/site-packages/nvidia/libnvcomp/lib64:$LD_LIBRARY_PATH

./bin/spja_lz4_nvcomp_split_overlap
```

Important outputs are stored under:

```text
results/spja_workload/csv/
results/spja_workload/graphs/
results/spja_workload/graph_plotting/
```

Representative result files include:

```text
results/spja_workload/csv/spja_lz4_nvcomp_hybrid_split_overlap_results.csv
results/spja_workload/csv/spja_lz4_nvcomp_hybrid_split_overlap_detailed_trials.csv
results/spja_workload/csv/spja_lz4_nvcomp_compression_stats.csv
results/spja_workload/csv/spja_lz4_nvcomp_summary.txt
```

Additional SPJA experiment outputs in the repository include:

```text
spja_compressed_vs_uncompressed_results.csv
spja_h2d_vs_no_h2d_results.csv
spja_lz4_nvcomp_four_mode_results.csv
spja_lz4_nvcomp_graph_data.csv
spja_lz4_nvcomp_key_findings_results.csv
```

For implementation notes, see:

```text
docs/spja.txt
```

---

## 8. String SPJA Variant

The string-aware LZ4 + nvCOMP SPJA implementation is:

```text
src/benchmark/spja_lz4_nvcomp_split_overlap_strings.cu
```

Associated string benchmark metadata and results are stored under:

```text
results/spja_workload/csv/
```

Examples include:

```text
spja_lz4_nvcomp_hybrid_split_overlap_strings_results.csv
spja_lz4_nvcomp_hybrid_split_overlap_strings_detailed_trials.csv
spja_lz4_nvcomp_strings_compression_stats.csv
spja_lz4_nvcomp_strings_summary.txt
spja_lz4_nvcomp_strings_benchmark_metadata.txt
```

These experiments extend the SPJA evaluation beyond the earlier integer-only processing path.

---

## 9. Four-Mode SPJA Comparison

The four-mode implementation is:

```text
src/benchmark/spja_lz4_nvcomp_four_modes.cu
```

Its corresponding result file is:

```text
results/spja_workload/csv/spja_lz4_nvcomp_four_mode_results.csv
```

This benchmark is retained as part of the broader comparison of different processing configurations.

---

## 10. FastLanes-GPU Baseline

FastLanes-GPU is used as a specialized comparison baseline.

Relevant source files are:

```text
external/baseline/fastlanes_gpu/coproc_fastlanes.cu
external/baseline/fastlanes_gpu/spja_coproc_fastlanes.cu
external/baseline/fastlanes_gpu/spja_coproc_fastlanes_strings.cu
```

Available execution scripts include:

```text
scripts/run_fastlanes_coproc_quantity.sh
scripts/run_fastlanes_spja_x25.sh
scripts/run_fastlanes_spja_x40.sh
scripts/run_fastlanes_spja_x40_strings.sh
```

Example:

```bash
bash scripts/run_fastlanes_spja_x40_strings.sh
```

Selected FastLanes comparison results are stored under:

```text
results/fastlanes_lz4nvcomp_dowda/
```

Important CSV files include:

```text
fastlanes_quantity_coproc_results.csv
fastlanes_spja_coproc_x25_results.csv
fastlanes_spja_coproc_x40_results.csv
fastlanes_spja_coproc_x40_strings_results.csv
```

Comparison plotting scripts are stored under:

```text
results/fastlanes_lz4nvcomp_dowda/graph_plotting/
```

FastLanes is a specialized vector-oriented packed decoding approach, while nvCOMP LZ4 is a general-purpose LZ4 decompression implementation. Therefore, these experiments should be interpreted as a comparison between different compressed-processing designs rather than as a one-to-one comparison of identical codecs.

For additional details, see:

```text
docs/fastlanes_gpu.txt
```

---

## 11. Fused Decode + SPJA Baseline

The repository also contains fused decode-and-query experimental baselines:

```text
src/benchmark/spja_dpf_fused_x40.cu
src/benchmark/spja_dpf_fused_x40_strings.cu
```

The fused design investigates a different processing strategy:

```text
Encoded Input
        ↓
Decode Required Values
        ↓
Immediately Apply Query Logic
        ↓
Produce Result
```

This differs from a fully separated pipeline where decoded data is first materialized and then consumed by another query stage.

Selected fused result files are stored under:

```text
results/fastlanes_lz4nvcomp_dowda/csv/
```

Examples include:

```text
dpf_fused_spja_x40_results.csv
dpf_fused_spja_x40_common_timing_results.csv
dpf_fused_spja_x40_common_timing_strings_results.csv
```

The fused implementation is an experimental thesis baseline and should not be interpreted as a full reproduction of an external database system.

For additional details, see:

```text
docs/fused_dpf_goda.txt
```

---

## 12. nvCOMP LZ4 vs FastLanes Comparison Utility

An earlier comparison benchmark is retained at:

```text
src/benchmark/nvcomp_lz4_vs_fastlane.cu
```

Related plotting code includes:

```text
results/baseline_fastlanes_gpu/comparison_fastlane.py
```

Run the plotting script with:

```bash
python3 results/baseline_fastlanes_gpu/comparison_fastlane.py
```

This comparison provides supporting context for the later full SPJA FastLanes experiments.

---

## 13. Result Organization

The repository intentionally retains selected CSV files, graphs, metadata, summaries, and plotting scripts used during thesis evaluation.

Important result folders include:

```text
results/baseline_comparison_graphs/
results/baseline_fastlanes_gpu/
results/fastlanes_lz4nvcomp_dowda/
results/lz4_nvcomp_pipeline/
results/spja_workload/
results/sf1_backup_csv/
results/sf1_backup_graphs/
results/sf10_csv/
results/sf10_graphs/
results/sf20_graphs/
results/simple_pipeline/
```

The result tree contains:

- raw benchmark CSV outputs
- detailed trial CSV files
- benchmark metadata
- compression statistics
- summary text files
- plotting scripts
- generated thesis graphs
- selected backup/reference results

Some plotting scripts contain experiment-specific paths. Check the input path inside a script before regenerating a graph.

---

## 14. Reproducing the Main Workflow

A typical fresh workflow is:

```bash
# 1. Clone repository
git clone <repository-url>
cd gpu_benchmark_clean

# 2. Install system dependencies
sudo apt update
sudo apt install -y build-essential cmake python3 python3-pip liblz4-dev

# 3. Install Python packages
python3 -m pip install --user pandas matplotlib numpy

# 4. Prepare nvCOMP
python3 -m venv nvcomp_env
source nvcomp_env/bin/activate
pip install nvidia-nvcomp-cu12

# 5. Set nvCOMP runtime path
export LD_LIBRARY_PATH=~/nvcomp_env/lib/python3.12/site-packages/nvidia/libnvcomp/lib64:$LD_LIBRARY_PATH

# 6. Generate TPC-H .tbl files with tpch-dbgen
# Example:
# ./dbgen -s 1

# 7. Convert TPC-H tables to binary columns
python3 scripts/tpch_columnar/convert_dbgen_tbl_to_bin.py

# 8. Compile the main SPJA benchmark
mkdir -p bin

nvcc -std=c++17 -O3 \
  -I ~/nvcomp_env/lib/python3.12/site-packages/nvidia/libnvcomp/include \
  -L ~/nvcomp_env/lib/python3.12/site-packages/nvidia/libnvcomp/lib64 \
  src/benchmark/spja_lz4_nvcomp_split_overlap.cu \
  -o bin/spja_lz4_nvcomp_split_overlap \
  -llz4 \
  -lnvcomp

# 9. Run
./bin/spja_lz4_nvcomp_split_overlap

# 10. Inspect results
ls -lh results/spja_workload/csv/
```

For FastLanes experiments, use the corresponding scripts in:

```text
scripts/run_fastlanes_*.sh
```

---

## 15. Git and Large-File Handling

Generated benchmark data and build artifacts are intentionally excluded from version control.

Important ignored categories include:

```text
archive/
bin/
build/
build_fmt/
data/
tools/tpch-dbgen/

*.o
*.out
*.exe
*.ptx
*.cubin
*.log
*.tmp
*.dat
*.bin
```

Large generated FastLanes datasets are also ignored, including:

```text
results/baseline_fastlanes_gpu/quantity_sf10/
results/baseline_fastlanes_gpu/quantity_sf10_sorted_lz4best/
results/baseline_fastlanes_gpu/quantity_sf10_sorted_lz4best_23x/
results/baseline_fastlanes_gpu/quantity_sf10_sorted_lz4best_45x/
```

The repository keeps selected graphs, CSV files, metadata, scripts, and documentation that are useful for understanding the thesis experiments while excluding large reproducible inputs and build products.

---

## 16. Environment Information

The repository contains:

```text
thesis_environment_info.sh
```

This file records thesis-environment information useful for reproducing the benchmark setup.

The experiments were developed and evaluated on an NVIDIA RTX A2000 with 12 GB of GPU memory.

Large-input experiments, especially some FastLanes configurations, can be constrained by available GPU memory.

---

## 17. Documentation

Additional implementation-specific notes are available under `docs/`:

```text
docs/averaged_benchmark_notes.txt
docs/benchmark_notes.txt
docs/fastlanes_gpu.txt
docs/fused_dpf_goda.txt
docs/lz4_nvcomp.txt
docs/rle_simple_pipeline.txt
docs/spja.txt
docs/tpch_data_generation.txt
```

These files provide more focused descriptions of individual benchmark stages, dependencies, execution paths, and experimental design decisions.

---

## 18. Thesis Context

This repository supports the thesis investigation of compressed CPU/GPU analytical query processing.

The main implementation evaluates:

- LZ4_HC preprocessing compression
- CPU LZ4 decompression
- GPU nvCOMP LZ4 decompression
- multi-threaded CPU processing
- GPU processing
- deterministic CPU/GPU workload assignment
- CPU-only, GPU-only, and hybrid execution
- overlapping CPU/GPU processing
- multiple timing trials
- correctness validation
- compression statistics
- throughput measurements

The broader evaluation additionally includes:

- FastLanes-GPU co-processing
- x25 and x40 FastLanes SPJA experiments
- string-aware SPJA processing
- fused decode-and-query baselines
- large-input and sensitivity experiments
- comparison graphs and key-finding summaries

The goal is not only to measure individual decompression kernels, but to evaluate the end-to-end trade-off between reduced data movement and the extra processing required to reconstruct compressed data.

---

## 19. Reproducibility Notes

To reproduce a specific thesis result:

1. Identify the corresponding source file or experiment script.
2. Prepare the required TPC-H-derived column data.
3. Verify CUDA, LZ4, and nvCOMP dependencies.
4. Compile the required benchmark or execute its script.
5. Run the same workload configuration.
6. Inspect the corresponding CSV and metadata files under `results/`.
7. Use the associated plotting script where available.

Exact experiment details may differ between the baseline, SPJA, FastLanes, string, and fused implementations. The source code, scripts, documentation, selected raw results, and graphs retained in this repository provide the reference implementation for the thesis experiments.

---

## 20. License and External Components

This repository contains thesis benchmark code together with selected external-baseline integration files.

External libraries and tools such as CUDA, nvCOMP, LZ4, TPC-H/dbgen, and FastLanes remain subject to their own licenses and distribution terms.

Large external source trees and generated benchmark datasets are intentionally not vendored into this repository unless required for the retained thesis baseline integration.

---

## 21. Repository Status

This repository has been cleaned for thesis submission and archival.

It retains:

- benchmark source code
- required FastLanes baseline source files
- experiment scripts
- thesis documentation
- selected CSV result files
- selected plots and graph-generation scripts
- environment information

It excludes:

- generated TPC-H binary data
- compiled binaries
- build directories
- temporary files
- large reproducible benchmark datasets
- unnecessary external source trees
- obsolete experimental archives
