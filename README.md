# Evaluating the Impact of Compression and Decompression on CPU–GPU Data Transfer Throughput

This repository contains CUDA/C++ benchmark implementations for a Master's thesis on CPU–GPU data-transfer throughput, compression, decompression, decoding, and CPU/GPU analytical query execution.

The project started with basic CPU/GPU memory-transfer measurements and was extended into a compressed TPC-H-derived columnar benchmark using CPU LZ4 decompression, GPU nvCOMP LZ4 decompression, FastLanes-GPU comparison, and a DPF-inspired fused decode-query baseline.

---

## 1. Project Goals

The main goals of this project are:

1. Measure baseline CPU memory-copy and CPU–GPU transfer throughput.
2. Evaluate compressed data processing with LZ4/nvCOMP.
3. Compare CPU-only, GPU-only, and hybrid CPU/GPU execution.
4. Study how compression affects effective analytical query throughput.
5. Compare the LZ4/nvCOMP pipeline with the specialized FastLanes-GPU integer decoding approach.
6. Evaluate a DPF-inspired fused decode-query baseline using a block-based Frame-of-Reference and bit-packed integer representation.
7. Evaluate the systems using a common TPC-H-derived SPJA workload and common CPU/GPU split configurations.
8. Validate correctness against the same CPU reference result.

The main experimental direction is:

```text
TPC-H-derived columnar data
        ↓
compressed / encoded representation
        ↓
CPU/GPU workload distribution
        ↓
CPU path                    GPU path
        ↓                       ↓
decompression / decoding    transfer + decompression / decoding
        ↓                       ↓
SPJA query processing       SPJA query processing
        ↓                       ↓
        partial results
              ↓
      result combination
              ↓
 correctness + timing validation
```

The three principal thesis implementations are:

```text
LZ4/nvCOMP
FastLanes-GPU
DPF-inspired fused baseline
```

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
│   │   ├── spja_lz4_nvcomp_four_modes.cu
│   │   ├── spja_lz4_nvcomp_split_overlap.cu
│   │   ├── spja_lz4_nvcomp_split_overlap_strings.cu
│   │   ├── spja_dpf_fused_x40.cu
│   │   ├── spja_dpf_fused_x40_strings.cu
│   │   └── nvcomp_lz4_vs_fastlane.cu
│   │
│   └── tools/
│       ├── compress_tbl_lz4.cpp
│       └── generate_customer_mktsegment_code.py
│
├── scripts/
│   ├── run_fastlanes_coproc_quantity.sh
│   ├── run_fastlanes_spja_x25.sh
│   ├── run_fastlanes_spja_x40.sh
│   ├── run_fastlanes_spja_x40_strings.sh
│   │
│   └── tpch_columnar/
│       ├── convert_dbgen_tbl_to_bin.py
│       ├── check_columnar_lz4_ratio.cpp
│       └── check_columnar_lz4_delta_ratio.cpp
│
├── tools/
│   └── convert_required_tpch_to_binary.py
│
├── docs/
│   ├── averaged_benchmark_notes.txt
│   ├── benchmark_notes.txt
│   ├── lz4_nvcomp.txt
│   ├── rle_simple_pipeline.txt
│   ├── spja.txt
│   ├── fastlanes_gpu.txt
│   ├── fused_dpf_goda.txt
│   └── tpch_data_generation.txt
│
├── external/
│   └── baseline/
│       └── fastlanes_gpu/
│           ├── coproc_fastlanes.cu
│           ├── spja_coproc_fastlanes.cu
│           └── spja_coproc_fastlanes_strings.cu
│
├── results/
│   ├── baseline_comparison_graphs/
│   ├── baseline_fastlanes_gpu/
│   ├── fastlanes_lz4nvcomp_dowda/
│   │   ├── csv/
│   │   ├── graph_plotting/
│   │   ├── graphs/
│   │   ├── spja_x25_fastlanes/
│   │   └── spja_x40_fastlanes/
│   ├── lz4_nvcomp_pipeline/
│   ├── sf10_csv/
│   ├── sf10_graphs/
│   ├── sf1_backup_csv/
│   ├── sf1_backup_graphs/
│   ├── sf20_graphs/
│   ├── simple_pipeline/
│   └── spja_workload/
│
├── data/
├── setup_dirs.sh
├── thesis_environment_info.sh
├── .gitignore
└── README.md
```

Large generated datasets, binaries, build folders, and other machine-generated artifacts are intentionally excluded from Git.

---

## 3. Requirements

The project is intended for a Linux machine with an NVIDIA GPU.

The thesis experiments were executed using an NVIDIA RTX A2000 with 12 GB GPU memory.

Required tools and libraries include:

- CUDA Toolkit with `nvcc`
- NVIDIA GPU and compatible CUDA driver
- nvCOMP
- LZ4 development library
- C++17 or newer compatible compiler
- Python 3
- NumPy
- pandas
- matplotlib
- FastLanes-GPU dependencies for the FastLanes implementation

### 3.1 Installing Common Dependencies

On Ubuntu/Debian systems:

```bash
sudo apt update
sudo apt install -y build-essential cmake python3 python3-pip liblz4-dev
```

Install Python packages:

```bash
python3 -m pip install --user numpy pandas matplotlib lz4
```

If a virtual environment is preferred:

```bash
python3 -m venv benchmark_env
source benchmark_env/bin/activate
python3 -m pip install numpy pandas matplotlib lz4
```

Python packages do not replace CUDA, nvCOMP, the NVIDIA driver, or system-level LZ4 development headers.

### 3.2 nvCOMP Runtime Path

The nvCOMP library path must be available at runtime.

On the thesis machine:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/nvcomp_env/lib/python3.12/site-packages/nvidia/libnvcomp/lib64
```

If nvCOMP is installed elsewhere, update the path accordingly.

### 3.3 Environment Information

The repository contains:

```text
thesis_environment_info.sh
```

This helper can be used to record relevant software and hardware environment information for reproducibility.

---

## 4. Basic Baseline Benchmarks

The initial CPU/GPU throughput benchmarks are:

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

These benchmarks measure basic CPU and GPU memory behavior, including:

- CPU memory copy
- host-to-device transfer
- device-to-host transfer
- GPU memory throughput

The averaged benchmark extends the initial baseline with repeated measurements to reduce sensitivity to individual-run variation.

---

## 5. TPC-H Data Preparation

The thesis benchmarks use TPC-H-derived columnar data.

The main SPJA workload uses information derived from:

```text
LINEITEM
ORDERS
CUSTOMER
```

The primary fact-side columns are:

```text
orderkey
quantity
extendedprice
```

Join lookup information is derived from:

```text
ORDERS
CUSTOMER
```

The main x40 workload contains:

```text
240,048,600 fact rows
```

with three 32-bit fact columns and approximately:

```text
2.683 GiB logical fact-column input
```

### 5.1 Generate TPC-H `.tbl` Files

TPC-H source tables should first be generated with a TPC-H `dbgen` implementation.

A typical development layout is:

```text
data/tpch_real/sf1/
```

Expected source tables include:

```text
data/tpch_real/sf1/lineitem.tbl
data/tpch_real/sf1/orders.tbl
data/tpch_real/sf1/customer.tbl
```

A typical generation workflow is:

```bash
cd tools/tpch-dbgen
make
./dbgen -s 1 -f
```

The generated `.tbl` files should then be placed under:

```text
data/tpch_real/sf1/
```

The external `tpch-dbgen` source itself is not retained as part of the thesis implementation.

### 5.2 Generate Required SPJA Binary Columns

The repository contains a converter for the required SPJA columns:

```text
tools/convert_required_tpch_to_binary.py
```

For SF1, run:

```bash
python3 tools/convert_required_tpch_to_binary.py \
  --sf 1 \
  --output-dir data/tpch_columnar
```

This generates the required integer columns:

```text
data/tpch_columnar/orderkey_sf1.bin
data/tpch_columnar/quantity_sf1.bin
data/tpch_columnar/extendedprice_sf1.bin
data/tpch_columnar/order_custkey_sf1.bin
data/tpch_columnar/customer_nation_sf1.bin
```

The fact columns are stored as 32-bit integers.

`extendedprice` is represented as integer cents.

### 5.3 Additional Quantity-Only Converter

The repository also contains:

```text
scripts/tpch_columnar/convert_dbgen_tbl_to_bin.py
```

This is a quantity-oriented helper used for additional compression/FastLanes experiments.

It generates quantity representations such as:

```text
quantity_sf10.bin
quantity_sf10_u8.bin
```

This script should not be confused with the complete SPJA column converter in:

```text
tools/convert_required_tpch_to_binary.py
```

### 5.4 SPJA x40 Input

The final cross-system benchmark uses the larger x40 TPC-H-derived workload.

The main implementations expect files such as:

```text
data/tpch_columnar/orderkey_sfx40.bin
data/tpch_columnar/quantity_sfx40.bin
data/tpch_columnar/extendedprice_sfx40.bin
data/tpch_columnar/order_custkey_sfx40.bin
data/tpch_columnar/customer_nation_sfx40.bin
```

The x40 workload was produced by scaling the TPC-H-derived column data while preserving the same logical SPJA structure.

Generated binary datasets are intentionally excluded from Git because they are large.

The dataset-preparation design is documented further in:

```text
docs/tpch_data_generation.txt
```

### 5.5 String / Categorical Variant

The string-oriented experiments additionally use the CUSTOMER market-segment attribute.

The helper:

```text
src/tools/generate_customer_mktsegment_code.py
```

converts the TPC-H market-segment strings into integer dictionary codes.

The mapping used by the script is:

```text
AUTOMOBILE -> 0
BUILDING   -> 1
FURNITURE  -> 2
MACHINERY  -> 3
HOUSEHOLD  -> 4
```

The x40 output is:

```text
data/tpch_columnar/customer_mktsegment_code_sfx40.bin
```

This allows the string/categorical predicate to participate in the CPU/GPU comparison using a compact integer representation.

---

## 6. Main LZ4/nvCOMP SPJA CPU/GPU Benchmark

The principal LZ4/nvCOMP thesis benchmark is:

```text
src/benchmark/spja_lz4_nvcomp_split_overlap.cu
```

A string/categorical variant is available as:

```text
src/benchmark/spja_lz4_nvcomp_split_overlap_strings.cu
```

The main benchmark performs:

- loading of TPC-H-derived binary columns
- chunk-wise LZ4_HC compression
- CPU LZ4 decompression
- GPU nvCOMP LZ4 decompression
- SPJA-style analytical query execution
- deterministic fair CPU/GPU chunk assignment
- concurrent CPU and GPU processing
- CPU-only, GPU-only, and hybrid CPU/GPU evaluation
- correctness validation against a CPU reference
- end-to-end runtime measurement
- CSV and summary generation

The CPU side uses multiple worker threads for CPU-assigned chunks.

### 6.1 LZ4 Configuration

The primary LZ4/nvCOMP experiments use:

```text
Chunk size:       512 KiB
Compression:      LZ4_HC
Compression level: 8
GPU batch size:   120 chunks
```

Compression is treated as preprocessing rather than part of the timed analytical query execution.

### 6.2 CPU/GPU Splits

The common split configurations are:

```text
100% CPU /   0% GPU
 75% CPU /  25% GPU
 50% CPU /  50% GPU
 25% CPU /  75% GPU
  0% CPU / 100% GPU
```

CPU and GPU execution paths are concurrent.

Therefore, CPU-path and GPU-path timing components should not be added together to estimate end-to-end runtime.

The end-to-end runtime is measured directly.

### 6.3 Fair Chunk Assignment

The LZ4/nvCOMP benchmark uses deterministic fair chunk distribution.

For each assignment trial, chunk IDs are reordered deterministically and assigned according to the requested CPU/GPU split.

This avoids repeatedly assigning the same consecutive chunk region to one processor while preserving the requested workload ratio.

### 6.4 Compile Main LZ4/nvCOMP Benchmark

```bash
mkdir -p bin

nvcc -std=c++17 -O3 \
  -I ~/nvcomp_env/lib/python3.12/site-packages/nvidia/libnvcomp/include \
  src/benchmark/spja_lz4_nvcomp_split_overlap.cu \
  -L ~/nvcomp_env/lib/python3.12/site-packages/nvidia/libnvcomp/lib64 \
  -lnvcomp \
  -llz4 \
  -o bin/spja_lz4_nvcomp_split_overlap
```

If nvCOMP is available through the default compiler paths, the explicit include/library paths may not be required.

### 6.5 Run Main LZ4/nvCOMP Benchmark

```bash
export LD_LIBRARY_PATH=~/nvcomp_env/lib/python3.12/site-packages/nvidia/libnvcomp/lib64:$LD_LIBRARY_PATH

./bin/spja_lz4_nvcomp_split_overlap
```

### 6.6 Main Result Files

The LZ4/nvCOMP benchmark writes results under:

```text
results/spja_workload/csv/
```

Important files include:

```text
spja_lz4_nvcomp_hybrid_split_overlap_results.csv
spja_lz4_nvcomp_hybrid_split_overlap_detailed_trials.csv
spja_lz4_nvcomp_compression_stats.csv
spja_lz4_nvcomp_benchmark_metadata.txt
spja_lz4_nvcomp_summary.txt
```

Graph-related outputs are stored under:

```text
results/spja_workload/graphs/
```

---

## 7. Cross-System Comparison: LZ4/nvCOMP, FastLanes-GPU, and DPF-Inspired Baseline

The final thesis comparison evaluates three implementations using the common SPJA workload:

```text
LZ4/nvCOMP
FastLanes-GPU
DPF-inspired fused baseline
```

The purpose is not to claim that all three approaches implement the same compression format.

Instead, the comparison evaluates different compressed/encoded processing strategies under a common analytical workload, common CPU/GPU splits, common correctness validation, and comparable end-to-end timing methodology.

### 7.1 Final SPJA x40 Implementations

The main LZ4/nvCOMP implementation is:

```text
src/benchmark/spja_lz4_nvcomp_split_overlap.cu
```

The main FastLanes-GPU implementation is:

```text
external/baseline/fastlanes_gpu/spja_coproc_fastlanes.cu
```

The main DPF-inspired implementation is:

```text
src/benchmark/spja_dpf_fused_x40.cu
```

The corresponding string/categorical variants are:

```text
src/benchmark/spja_lz4_nvcomp_split_overlap_strings.cu

external/baseline/fastlanes_gpu/spja_coproc_fastlanes_strings.cu

src/benchmark/spja_dpf_fused_x40_strings.cu
```

The final x40 comparison result CSVs are retained under:

```text
results/fastlanes_lz4nvcomp_dowda/csv/
```

Important files include:

```text
lz4_nvcomp_spja_x40_results.csv
fastlanes_spja_coproc_x40_results.csv
fastlanes_spja_coproc_x40_strings_results.csv
dpf_fused_spja_x40_results.csv
dpf_fused_spja_x40_common_timing_results.csv
dpf_fused_spja_x40_common_timing_strings_results.csv
```

An earlier LZ4-vs-FastLanes comparison benchmark is also retained as:

```text
src/benchmark/nvcomp_lz4_vs_fastlane.cu
```

This benchmark is useful for historical/decompression-focused comparison, but the final thesis SPJA evaluation uses the complete x40 implementations listed above.

### 7.2 FastLanes-GPU Upstream Reference

The FastLanes-GPU comparison is based on the official FastLanesGPU reference implementation:

https://github.com/cwida/FastLanesGPU

Reference commit inspected for reproducibility:

```text
bf668e99662d0692faa3ad3e57b10ad5d890c722
```

The thesis implementation adapts FastLanes-GPU to the common SPJA workload, CPU/GPU split configurations, timing methodology, and correctness validation used in this thesis.

The thesis-specific FastLanes files retained in this repository are:

```text
external/baseline/fastlanes_gpu/coproc_fastlanes.cu
external/baseline/fastlanes_gpu/spja_coproc_fastlanes.cu
external/baseline/fastlanes_gpu/spja_coproc_fastlanes_strings.cu
```

FastLanes-GPU is a specialized packed integer decoding approach, whereas LZ4/nvCOMP uses a general-purpose LZ4 compression/decompression pipeline.

The comparison should therefore be interpreted as a system/design baseline comparison rather than as an identical-codec comparison.

The FastLanes implementation and experimental setup are documented in:

```text
docs/fastlanes_gpu.txt
```

#### Running the FastLanes x40 SPJA Benchmark

The repository includes:

```text
scripts/run_fastlanes_spja_x40.sh
```

Run:

```bash
bash scripts/run_fastlanes_spja_x40.sh
```

For the string/categorical variant:

```bash
bash scripts/run_fastlanes_spja_x40_strings.sh
```

These scripts expect a locally configured FastLanesGPU build environment and use the thesis-specific source files retained in this repository.

### 7.3 DPF-Inspired Fused Baseline

The DPF-inspired comparison is based on the design and reference artifact of DPFProto.

Official reference repository:

https://github.com/dbc-utokyoiis/DPFProto

Reference commit inspected for reproducibility:

```text
697fe24fc500db52c60c17bb021d3dcfd38289e5
```

The thesis implementation is a simplified DPF-inspired fused decode-query baseline.

It uses a block-based Frame-of-Reference and bit-packed integer representation together with selective decoding during analytical query execution.

The implementation uses a DPF-style packed layout with:

```text
block size: 128 values
miniblocks: 4
block reference/minimum value
bit-width metadata
bit-packed values
```

The purpose of the fused design is to reduce unnecessary intermediate materialization by consuming reconstructed values directly during query processing where possible.

The implementation does not reproduce the complete DPFProto architecture.

In particular, it does not reproduce the full BaM-based GPU-initiated storage I/O path of DPFProto.

The thesis implementations are:

```text
src/benchmark/spja_dpf_fused_x40.cu
src/benchmark/spja_dpf_fused_x40_strings.cu
```

Further documentation is available in:

```text
docs/fused_dpf_goda.txt
```

#### Compile DPF-Inspired x40 Benchmark

```bash
mkdir -p bin

nvcc -O3 -std=c++17 -arch=sm_86 \
  src/benchmark/spja_dpf_fused_x40.cu \
  -o bin/spja_dpf_fused_x40
```

Run:

```bash
./bin/spja_dpf_fused_x40
```

For the string/categorical variant:

```bash
nvcc -O3 -std=c++17 -arch=sm_86 \
  src/benchmark/spja_dpf_fused_x40_strings.cu \
  -o bin/spja_dpf_fused_x40_strings
```

Run:

```bash
./bin/spja_dpf_fused_x40_strings
```

### 7.4 Common SPJA Query

The common analytical workload is based on Selection, Projection, Join, and Aggregation.

The integer-oriented workload uses:

```text
LINEITEM.orderkey
LINEITEM.quantity
LINEITEM.extendedprice
ORDERS customer lookup
CUSTOMER nation lookup
```

The main predicates include:

```text
quantity > 25
customer nation = target nation
```

The aggregate is based on qualifying:

```text
extendedprice
```

For the primary integer workload, all three systems are validated against the same CPU reference result:

```text
27571934343560
```

### 7.5 Timing Methodology

End-to-end runtime is measured directly.

CPU and GPU path measurements are retained as diagnostic timing components.

Because CPU and GPU work execute concurrently, those individual timing components should not be summed to obtain end-to-end runtime.

Conceptually:

```text
End-to-end runtime ≈ max(CPU path, GPU path) + coordination overhead
```

but the benchmark does not rely on that expression to generate the reported end-to-end runtime.

The actual end-to-end runtime is directly measured.

Effective throughput is computed from:

```text
logical input size / measured end-to-end runtime
```

This allows implementations with different compressed or encoded representations to be compared using the same logical input size.

---

## 8. Plotting and Result Files

Benchmark outputs include:

- CSV result files
- detailed timing files
- compression statistics
- correctness information
- summary files
- plotting scripts
- generated graphs

Important result folders include:

```text
results/spja_workload/
results/fastlanes_lz4nvcomp_dowda/
results/baseline_comparison_graphs/
results/baseline_fastlanes_gpu/
results/lz4_nvcomp_pipeline/
```

The final three-system plotting utilities are under:

```text
results/fastlanes_lz4nvcomp_dowda/graph_plotting/
```

These scripts use result files from:

```text
LZ4/nvCOMP
FastLanes-GPU
DPF-inspired fused
```

The plotting scripts should be checked for input CSV paths before regeneration because some experimental scripts use fixed repository-relative paths.

The repository retains selected CSV files and graphs required to document the thesis experiments.

Large temporary outputs and generated datasets are excluded.

---

## 9. Reproducing the Main Workflow

A fresh environment can reproduce the main workflow conceptually as follows.

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd cpu-gpu-compression-throughput-benchmark
```

### Step 2: Install Common Dependencies

```bash
sudo apt update
sudo apt install -y build-essential cmake python3 python3-pip liblz4-dev

python3 -m pip install --user numpy pandas matplotlib lz4
```

CUDA, an NVIDIA driver, and nvCOMP must also be installed separately.

### Step 3: Generate TPC-H Data

Generate the required TPC-H tables and place them under:

```text
data/tpch_real/sf1/
```

Required tables include:

```text
lineitem.tbl
orders.tbl
customer.tbl
```

### Step 4: Generate Required SPJA Columns

```bash
python3 tools/convert_required_tpch_to_binary.py \
  --sf 1 \
  --output-dir data/tpch_columnar
```

### Step 5: Prepare x40 Input

The final thesis benchmark expects the prepared x40 files:

```text
data/tpch_columnar/orderkey_sfx40.bin
data/tpch_columnar/quantity_sfx40.bin
data/tpch_columnar/extendedprice_sfx40.bin
data/tpch_columnar/order_custkey_sfx40.bin
data/tpch_columnar/customer_nation_sfx40.bin
```

Large generated x40 binary data is not committed to Git.

See:

```text
docs/tpch_data_generation.txt
```

for the dataset-preparation design and configuration.

### Step 6: Prepare String/Categorical Lookup if Required

For the string/categorical experiments:

```bash
python3 src/tools/generate_customer_mktsegment_code.py
```

Expected output:

```text
data/tpch_columnar/customer_mktsegment_code_sfx40.bin
```

### Step 7: Configure nvCOMP

```bash
export LD_LIBRARY_PATH=~/nvcomp_env/lib/python3.12/site-packages/nvidia/libnvcomp/lib64:$LD_LIBRARY_PATH
```

### Step 8: Compile LZ4/nvCOMP

```bash
mkdir -p bin

nvcc -std=c++17 -O3 \
  -I ~/nvcomp_env/lib/python3.12/site-packages/nvidia/libnvcomp/include \
  src/benchmark/spja_lz4_nvcomp_split_overlap.cu \
  -L ~/nvcomp_env/lib/python3.12/site-packages/nvidia/libnvcomp/lib64 \
  -lnvcomp \
  -llz4 \
  -o bin/spja_lz4_nvcomp_split_overlap
```

### Step 9: Run LZ4/nvCOMP

```bash
./bin/spja_lz4_nvcomp_split_overlap
```

### Step 10: Run FastLanes-GPU

After configuring the FastLanesGPU build environment locally:

```bash
bash scripts/run_fastlanes_spja_x40.sh
```

For the string/categorical experiment:

```bash
bash scripts/run_fastlanes_spja_x40_strings.sh
```

### Step 11: Compile and Run DPF-Inspired Baseline

```bash
nvcc -O3 -std=c++17 -arch=sm_86 \
  src/benchmark/spja_dpf_fused_x40.cu \
  -o bin/spja_dpf_fused_x40

./bin/spja_dpf_fused_x40
```

### Step 12: Check Results

LZ4/nvCOMP:

```text
results/spja_workload/csv/
```

FastLanes/DPF comparison data:

```text
results/fastlanes_lz4nvcomp_dowda/csv/
```

Plotting scripts:

```text
results/fastlanes_lz4nvcomp_dowda/graph_plotting/
```

---

## 10. Git and Large File Handling

Large generated benchmark data should not be committed to Git.

The repository excludes generated datasets and build artifacts using `.gitignore`.

Typical excluded content includes:

```text
data/
*.bin
*.dat
bin/
build/
build_fmt/
```

Large local FastLanes data folders are also excluded where appropriate.

Generated binary TPC-H data is intentionally omitted because the base TPC-H column data can be recreated from the source tables and retained preparation utilities. The larger x40 workload used in the final experiments is generated from the TPC-H-derived columns as described in the accompanying data-generation documentation.

Selected result CSVs, graphs, scripts, and documentation required for the thesis are retained.

External development-only components such as a local TPC-H `dbgen` checkout are also not required to be committed as thesis source code.

---

## 11. Current Thesis Context

This repository supports the Master's thesis investigation of compressed CPU/GPU analytical query processing.

The final evaluation focuses on a TPC-H-derived SPJA workload and three principal implementations:

```text
1. LZ4/nvCOMP
2. FastLanes-GPU
3. DPF-inspired fused decode-query baseline
```

The LZ4/nvCOMP implementation evaluates:

- LZ4_HC preprocessing compression
- CPU LZ4 decompression
- GPU nvCOMP LZ4 decompression
- multi-threaded CPU processing
- deterministic fair CPU/GPU chunk assignment
- concurrent CPU/GPU execution
- multiple CPU/GPU workload splits
- directly measured end-to-end runtime
- correctness validation
- compression statistics
- effective throughput

The FastLanes-GPU implementation provides a specialized packed integer decoding comparison adapted to the same SPJA workload and CPU/GPU split methodology.

The DPF-inspired implementation provides an additional fused decode-query baseline for comparing a lightweight packed representation with the LZ4/nvCOMP and FastLanes approaches.

The final cross-system comparison therefore studies not only decompression speed, but complete analytical processing behavior including:

```text
data representation
CPU/GPU workload distribution
host-to-device transfer where applicable
decompression / decoding
SPJA execution
concurrent CPU/GPU processing
result combination
correctness validation
end-to-end runtime
effective throughput
```

The primary CPU/GPU split configurations are:

```text
100/0
75/25
50/50
25/75
0/100
```

where the first value is the CPU percentage and the second value is the GPU percentage.

The main x40 integer workload contains:

```text
240,048,600 rows
approximately 2.683 GiB logical fact-column input
```

All primary implementations are checked against the same CPU reference result.

---

## Research Artifacts

The thesis research artifacts include:

- CUDA/C++ benchmark source code
- data-preparation utilities
- experiment scripts
- FastLanes thesis adaptations
- DPF-inspired implementation
- string/categorical variants
- selected result CSV files
- plotting scripts
- experiment documentation
- reproducibility information

University GitLab repository:

https://gitlab.studium.uni-bamberg.de/faheem.riaz/cpu-gpu-compression-throughput-benchmark

GitHub mirror:

https://github.com/FaheemRiaz1/cpu-gpu-compression-throughput-benchmark