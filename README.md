# CPU–GPU Compression Throughput Benchmark

This repository contains CUDA/C++ benchmark implementations for a Master's thesis on CPU–GPU data-transfer throughput, compression, decompression, and CPU/GPU query execution.

The project started with basic CPU/GPU memory-transfer measurements and was extended into a compressed TPC-H columnar benchmark using CPU LZ4 decompression, GPU nvCOMP LZ4 decompression, and FastLanes-GPU baseline comparison.

---

## 1. Project Goals

The main goals of this project are:

1. Measure baseline CPU memory-copy and CPU–GPU transfer throughput.
2. Evaluate compressed data processing with LZ4/nvCOMP.
3. Compare CPU-only, GPU-only, and hybrid CPU/GPU execution.
4. Study how compression affects effective query throughput.
5. Compare general-purpose GPU decompression with a specialized integer-decoding baseline such as FastLanes-GPU.

The central benchmark direction is:

```text
TPC-H columnar data
        ↓
chunk-wise LZ4_HC compression
        ↓
CPU LZ4 decompression and/or GPU nvCOMP LZ4 decompression
        ↓
SPJA-style query execution
        ↓
correctness validation and throughput measurement
```

---

## 2. Repository Structure

Important visible project folders:

```text
.
├── src/
│   └── benchmark/
│       ├── baseline_benchmark.cu
│       ├── averaged_benchmark.cu
│       ├── simple_pipeline.cu
│       ├── parallel_cpu_lz4_nvcomp_full_pipeline.cu
│       ├── spja_lz4_nvcomp_split_overlap.cu
│       └── nvcomp_lz4_vs_fastlane.cu
│
├── scripts/
│   └── tpch_columnar/
│       ├── convert_dbgen_tbl_to_bin.py
│       ├── generate_tpch_columnar.py
│       ├── check_columnar_lz4_ratio.cpp
│       └── check_columnar_lz4_delta_ratio.cpp
│
├── docs/
│   ├── averaged_benchmark_notes.txt
│   ├── benchmark_notes.txt
│   ├── lz4_nvcomp.txt
│   ├── rle_simple_pipeline.txt
│   └── spja.txt
│
├── results/
│   ├── baseline_comparison_graphs/
│   ├── baseline_fastlanes_gpu/
│   │   └── comparison_fastlane.py
│   ├── lz4_nvcomp_pipeline/
│   ├── nvcomp_lz4_vs_fastlane/
│   ├── spja_workload/
│   └── other result/backup folders
│
├── data/
├── external/
├── tools/
├── setup_dirs.sh
├── .gitignore
└── README.md
```

Some generated folders contain large local data files and are intentionally ignored by Git. See [Section 10](#10-git-and-large-file-handling).

---

## 3. Requirements

The project is intended for a Linux machine with an NVIDIA GPU.

Required tools/libraries:

- CUDA Toolkit with `nvcc`
- NVIDIA GPU and CUDA driver
- nvCOMP
- LZ4 development library
- C++17 compatible compiler
- Python 3
- pandas and matplotlib for plotting scripts

### 3.1 Installing Dependencies

On Ubuntu/Debian systems, install system-level build tools and LZ4 headers with `apt`:

```bash
sudo apt update
sudo apt install -y build-essential cmake python3 python3-pip liblz4-dev
```

Install Python packages with `pip`:

```bash
python3 -m pip install --user pandas matplotlib numpy
```

If the system uses a virtual environment, activate it first and install without `--user`:

```bash
python3 -m venv benchmark_env
source benchmark_env/bin/activate
python3 -m pip install pandas matplotlib numpy
```

If `sudo apt install` is not available on the target machine, ask the system administrator to install the system packages, or use an existing module/environment. Python packages can usually still be installed locally with:

```bash
python3 -m pip install --user pandas matplotlib numpy
```

Note: `pip` can install Python packages, but it does not replace system libraries such as CUDA, nvCOMP, or the LZ4 development headers required for compiling the C++/CUDA benchmarks.

The nvCOMP library path must be available at runtime. On the current thesis machine, this was done with:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/nvcomp_env/lib/python3.12/site-packages/nvidia/libnvcomp/lib64
```

If nvCOMP is installed somewhere else, update the path accordingly.

---

## 4. Basic Baseline Benchmarks

The original baseline benchmarks are located in:

```text
src/benchmark/baseline_benchmark.cu
src/benchmark/averaged_benchmark.cu
```

Compile:

```bash
cd ~/gpu_benchmark_clean

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

These programs measure basic memory and transfer behavior, including CPU memory copy, host-to-device transfer, device-to-host transfer, and GPU memory throughput.

---

## 5. TPC-H Data Preparation

The SPJA benchmark expects binary column files under:

```text
data/tpch_columnar/
```

The expected files include:

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

### 5.1 Generate TPC-H `.tbl` Files

TPC-H `.tbl` files should first be generated using `tpch-dbgen`.

A typical expected source folder is:

```text
data/tpch_real/sf1/
```

Example files:

```text
data/tpch_real/sf1/lineitem.tbl
data/tpch_real/sf1/orders.tbl
data/tpch_real/sf1/customer.tbl
data/tpch_real/sf1/part.tbl
```

For larger scale factors, use folders such as:

```text
data/tpch_real/sf10/
```

The exact `dbgen` command depends on the local TPC-H/dbgen setup. A typical example is:

```bash
./dbgen -s 1
```

Then move the generated `.tbl` files into:

```text
data/tpch_real/sf1/
```

### 5.2 Convert `.tbl` Files to Binary Column Files

Use the converter script under:

```text
scripts/tpch_columnar/convert_dbgen_tbl_to_bin.py
```

Run from the project root:

```bash
cd ~/gpu_benchmark_clean

python3 scripts/tpch_columnar/convert_dbgen_tbl_to_bin.py
```

After conversion, check:

```bash
ls -lh data/tpch_columnar/
```

The benchmark paths are currently hard-coded for SF=1 binary column files in `data/tpch_columnar/`. If a different scale factor is used, either update the file names in the benchmark source code or create matching symlinks.

---

## 6. Main Compressed SPJA CPU/GPU Benchmark

The main thesis benchmark is:

```text
src/benchmark/spja_lz4_nvcomp_split_overlap.cu
```

This benchmark performs:

- loading of TPC-H-derived binary column files,
- chunk-wise LZ4_HC compression,
- CPU LZ4 decompression,
- GPU nvCOMP LZ4 decompression,
- SPJA-style query execution,
- deterministic fair CPU/GPU chunk assignment,
- CPU-only, GPU-only, and hybrid CPU/GPU split evaluation,
- correctness validation against CPU reference result,
- CSV and summary output generation.

The current CPU side has been updated to use multiple CPU worker threads for CPU-assigned chunks, instead of using only one CPU thread.

### 6.1 Compile Main SPJA Benchmark

```bash
cd ~/gpu_benchmark_clean

mkdir -p bin

nvcc -std=c++17 -O3 \
  src/benchmark/spja_lz4_nvcomp_split_overlap.cu \
  -o bin/spja_lz4_nvcomp_split_overlap \
  -llz4 \
  -lnvcomp
```

If nvCOMP headers/libraries are not found automatically, add include/library paths, for example:

```bash
nvcc -std=c++17 -O3 \
  -I~/nvcomp_env/lib/python3.12/site-packages/nvidia/libnvcomp/include \
  -L~/nvcomp_env/lib/python3.12/site-packages/nvidia/libnvcomp/lib64 \
  src/benchmark/spja_lz4_nvcomp_split_overlap.cu \
  -o bin/spja_lz4_nvcomp_split_overlap \
  -llz4 \
  -lnvcomp
```

### 6.2 Run Main SPJA Benchmark

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/nvcomp_env/lib/python3.12/site-packages/nvidia/libnvcomp/lib64

./bin/spja_lz4_nvcomp_split_overlap
```

Expected output files are written under:

```text
results/spja_workload/csv/
results/spja_workload/graphs/
```

Important CSV files include:

```text
results/spja_workload/csv/spja_lz4_nvcomp_hybrid_split_overlap_results.csv
results/spja_workload/csv/spja_lz4_nvcomp_hybrid_split_overlap_detailed_trials.csv
results/spja_workload/csv/spja_lz4_nvcomp_compression_stats.csv
results/spja_workload/csv/spja_lz4_nvcomp_summary.txt
```

---

## 7. nvCOMP LZ4 vs FastLanes-GPU Comparison

The comparison benchmark source is:

```text
src/benchmark/nvcomp_lz4_vs_fastlane.cu
```

This benchmark is used to compare nvCOMP LZ4 decompression throughput with FastLanes-GPU style integer decoding/decompression results.

FastLanes-related local results/scripts are under:

```text
results/baseline_fastlanes_gpu/
```

The plotting script is:

```text
results/baseline_fastlanes_gpu/comparison_fastlane.py
```

Run the comparison plotting script:

```bash
cd ~/gpu_benchmark_clean

python3 results/baseline_fastlanes_gpu/comparison_fastlane.py
```

Generated comparison graphs/CSV files are stored under result folders such as:

```text
results/baseline_comparison_graphs/
results/nvcomp_lz4_vs_fastlane/
```

Important note: FastLanes-GPU is a specialized packed integer decoding/decompression baseline, whereas nvCOMP LZ4 is a general-purpose LZ4 decompression library. Therefore, the comparison should be presented as a baseline comparison between different compression/decompression approaches, not as a one-to-one identical codec comparison.

---

## 8. Plotting and Result Files

Result folders contain CSV files, summaries, and generated graphs.

Useful folders:

```text
results/spja_workload/
results/baseline_comparison_graphs/
results/nvcomp_lz4_vs_fastlane/
results/lz4_nvcomp_pipeline/
```

Example plotting scripts may be located in result folders or under scripts, depending on the experiment.

Before regenerating graphs, check the CSV path inside the plotting script. Some plotting scripts use hard-coded CSV paths.

---

## 9. Reproducing the Main Workflow

A fresh user can reproduce the main workflow approximately as follows:

```bash
# 1. Clone repository
git clone <repository-url>
cd gpu_benchmark_clean

# 2. Install dependencies
sudo apt update
sudo apt install -y build-essential cmake python3 python3-pip liblz4-dev
pip install pandas matplotlib numpy

# 3. Set nvCOMP runtime path
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/nvcomp_env/lib/python3.12/site-packages/nvidia/libnvcomp/lib64

# 4. Prepare TPC-H .tbl files manually using tpch-dbgen
# Put them under:
# data/tpch_real/sf1/

# 5. Convert TPC-H .tbl files to binary column files
python3 scripts/tpch_columnar/convert_dbgen_tbl_to_bin.py

# 6. Compile main benchmark
mkdir -p bin

nvcc -std=c++17 -O3 \
  src/benchmark/spja_lz4_nvcomp_split_overlap.cu \
  -o bin/spja_lz4_nvcomp_split_overlap \
  -llz4 \
  -lnvcomp

# 7. Run benchmark
./bin/spja_lz4_nvcomp_split_overlap

# 8. Check output
ls -lh results/spja_workload/csv/
```

If nvCOMP include/library paths are not globally available, use the longer compile command shown in Section 6.1.

---

## 10. Git and Large File Handling

Large generated benchmark data should not be committed to Git.

The following folders are ignored because they contain large local FastLanes benchmark data:

```text
results/baseline_fastlanes_gpu/quantity_sf10/
results/baseline_fastlanes_gpu/quantity_sf10_sorted_lz4best/
results/baseline_fastlanes_gpu/quantity_sf10_sorted_lz4best_23x/
results/baseline_fastlanes_gpu/quantity_sf10_sorted_lz4best_45x/
```

Other ignored patterns include:

```text
results/nvcomp_lz4_vs_fastlane/
*.dat
*.bin
build/
build_fmt/
```

## 11. Current Thesis Context

This repository supports the thesis investigation of compressed CPU/GPU query processing. The current main benchmark direction is a compressed TPC-H SPJA workload with:

- LZ4_HC preprocessing compression,
- CPU LZ4 decompression,
- GPU nvCOMP LZ4 decompression,
- multi-threaded CPU processing for CPU-owned chunks,
- fair deterministic CPU/GPU chunk assignment,
- multiple timed runs and assignment trials,
- correctness validation,
- throughput and compression-statistics reporting.

The FastLanes-GPU comparison is used as an external/specialized baseline for integer-oriented decompression throughput.

---