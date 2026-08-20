#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <lz4.h>
#include <lz4hc.h>
#include <nvcomp/lz4.h>

#define CUDA_CHECK(x)                                                          \
  do {                                                                         \
    cudaError_t err__ = (x);                                                   \
    if (err__ != cudaSuccess) {                                                \
      std::cerr << "CUDA error: " << cudaGetErrorString(err__)                 \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;        \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

#define NVCOMP_CHECK(x)                                                        \
  do {                                                                         \
    nvcompStatus_t st__ = (x);                                                 \
    if (st__ != nvcompSuccess) {                                               \
      std::cerr << "nvCOMP error: status=" << static_cast<int>(st__)           \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;        \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

static double bytes_to_mib(size_t bytes)
{
  return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

static double bytes_to_gib(size_t bytes)
{
  return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
}

static double bytes_to_gb(size_t bytes)
{
  return static_cast<double>(bytes) / 1e9;
}

static double mean(const std::vector<double>& xs)
{
  if (xs.empty()) return 0.0;
  double s = 0.0;
  for (double x : xs) s += x;
  return s / static_cast<double>(xs.size());
}

static double stddev_sample(const std::vector<double>& xs)
{
  if (xs.size() < 2) return 0.0;
  const double m = mean(xs);
  double v = 0.0;
  for (double x : xs) {
    const double d = x - m;
    v += d * d;
  }
  v /= static_cast<double>(xs.size() - 1);
  return std::sqrt(v);
}

static double compression_reduction(size_t original, size_t compressed)
{
  if (original == 0) return 0.0;
  return (1.0 - static_cast<double>(compressed) /
                    static_cast<double>(original)) * 100.0;
}

static size_t file_size_bytes(const std::string& path)
{
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("Could not open file: " + path);
  return static_cast<size_t>(f.tellg());
}

static bool file_exists(const std::string& path)
{
  std::ifstream f(path, std::ios::binary);
  return static_cast<bool>(f);
}

static std::vector<int> read_int_column(const std::string& path)
{
  const size_t bytes = file_size_bytes(path);
  if (bytes % sizeof(int) != 0) {
    throw std::runtime_error("Invalid int32 column size: " + path);
  }

  std::vector<int> v(bytes / sizeof(int));
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("Could not open file: " + path);

  f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(bytes));
  if (static_cast<size_t>(f.gcount()) != bytes) {
    throw std::runtime_error("Could not read full file: " + path);
  }

  return v;
}

static unsigned long long cpu_sum_quantity(const int* q, size_t n)
{
  unsigned long long s = 0ULL;
  for (size_t i = 0; i < n; ++i) {
    s += static_cast<unsigned long long>(q[i]);
  }
  return s;
}

__global__ void sum_int_kernel(
    const int* __restrict__ q,
    size_t n,
    unsigned long long* __restrict__ block_sums)
{
  extern __shared__ unsigned long long smem[];

  const unsigned int tid = threadIdx.x;
  const size_t global = static_cast<size_t>(blockIdx.x) * blockDim.x + tid;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

  unsigned long long local = 0ULL;
  for (size_t i = global; i < n; i += stride) {
    local += static_cast<unsigned long long>(q[i]);
  }

  smem[tid] = local;
  __syncthreads();

  for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (tid < s) smem[tid] += smem[tid + s];
    __syncthreads();
  }

  if (tid == 0) block_sums[blockIdx.x] = smem[0];
}

__global__ void reduce_ull_kernel(
    const unsigned long long* __restrict__ block_sums,
    size_t n,
    unsigned long long* __restrict__ result)
{
  extern __shared__ unsigned long long smem[];

  const unsigned int tid = threadIdx.x;
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + tid;

  unsigned long long local = 0ULL;
  if (i < n) local = block_sums[i];

  smem[tid] = local;
  __syncthreads();

  for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (tid < s) smem[tid] += smem[tid + s];
    __syncthreads();
  }

  if (tid == 0) atomicAdd(result, smem[0]);
}

struct CompressedColumn {
  std::vector<size_t> uncomp_sizes;
  std::vector<size_t> comp_sizes;
  std::vector<size_t> comp_offsets;
  std::vector<std::vector<char>> comp_chunks;
  size_t total_comp_bytes = 0;
};

static CompressedColumn compress_column_lz4_hc(
    const int* data,
    size_t n_rows,
    size_t chunk_rows,
    int lz4_hc_level)
{
  CompressedColumn out;

  const size_t total_chunks = (n_rows + chunk_rows - 1) / chunk_rows;
  out.uncomp_sizes.resize(total_chunks);
  out.comp_sizes.resize(total_chunks);
  out.comp_offsets.resize(total_chunks);
  out.comp_chunks.resize(total_chunks);

  const char* src = reinterpret_cast<const char*>(data);

  for (size_t c = 0; c < total_chunks; ++c) {
    const size_t row0 = c * chunk_rows;
    const size_t rows = std::min(chunk_rows, n_rows - row0);
    const size_t uncomp_bytes = rows * sizeof(int);
    const size_t byte0 = row0 * sizeof(int);

    if (uncomp_bytes > static_cast<size_t>(std::numeric_limits<int>::max())) {
      throw std::runtime_error("Chunk too large for LZ4 int API.");
    }

    const int max_comp = LZ4_compressBound(static_cast<int>(uncomp_bytes));
    out.comp_chunks[c].resize(static_cast<size_t>(max_comp));

    const int comp = LZ4_compress_HC(
        src + byte0,
        out.comp_chunks[c].data(),
        static_cast<int>(uncomp_bytes),
        max_comp,
        lz4_hc_level);

    if (comp <= 0) throw std::runtime_error("LZ4_HC compression failed.");

    out.uncomp_sizes[c] = uncomp_bytes;
    out.comp_sizes[c] = static_cast<size_t>(comp);
    out.comp_offsets[c] = out.total_comp_bytes;
    out.total_comp_bytes += static_cast<size_t>(comp);
    out.comp_chunks[c].resize(static_cast<size_t>(comp));
  }

  return out;
}

struct HostBatchPlan {
  size_t first_chunk = 0;
  size_t count = 0;
  size_t rows = 0;
  size_t uncomp_bytes = 0;
  size_t comp_bytes = 0;

  std::vector<size_t> comp_sizes;
  std::vector<size_t> uncomp_sizes;
  std::vector<size_t> local_row_offsets;
  std::vector<size_t> global_row_offsets;
  std::vector<size_t> global_comp_offsets;
};

struct DeviceBatchPlan {
  size_t first_chunk = 0;
  size_t count = 0;
  size_t rows = 0;
  size_t uncomp_bytes = 0;
  size_t comp_bytes = 0;

  void** d_comp_ptrs = nullptr;
  void** d_decomp_ptrs = nullptr;
  size_t* d_comp_sizes = nullptr;
  size_t* d_uncomp_sizes = nullptr;
  size_t* d_actual_uncomp_sizes = nullptr;
  nvcompStatus_t* d_statuses = nullptr;

  size_t temp_bytes = 0;
};

static HostBatchPlan build_host_batch_plan(
    size_t first_chunk,
    size_t count,
    size_t n_rows,
    size_t chunk_rows,
    const CompressedColumn& comp)
{
  HostBatchPlan p;
  p.first_chunk = first_chunk;
  p.count = count;
  p.comp_sizes.resize(count);
  p.uncomp_sizes.resize(count);
  p.local_row_offsets.resize(count);
  p.global_row_offsets.resize(count);
  p.global_comp_offsets.resize(count);

  size_t local_rows = 0;
  size_t local_uncomp_bytes = 0;
  size_t local_comp_bytes = 0;

  for (size_t k = 0; k < count; ++k) {
    const size_t c = first_chunk + k;
    const size_t global_row0 = c * chunk_rows;
    const size_t rows = std::min(chunk_rows, n_rows - global_row0);
    const size_t uncomp_bytes = rows * sizeof(int);

    p.comp_sizes[k] = comp.comp_sizes[c];
    p.uncomp_sizes[k] = uncomp_bytes;
    p.local_row_offsets[k] = local_rows;
    p.global_row_offsets[k] = global_row0;
    p.global_comp_offsets[k] = comp.comp_offsets[c];

    local_rows += rows;
    local_uncomp_bytes += uncomp_bytes;
    local_comp_bytes += comp.comp_sizes[c];
  }

  p.rows = local_rows;
  p.uncomp_bytes = local_uncomp_bytes;
  p.comp_bytes = local_comp_bytes;

  return p;
}

static float elapsed_ms(cudaEvent_t a, cudaEvent_t b)
{
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
  return ms;
}

static void destroy_device_batch(DeviceBatchPlan& b)
{
  if (b.d_comp_ptrs) CUDA_CHECK(cudaFree(b.d_comp_ptrs));
  if (b.d_decomp_ptrs) CUDA_CHECK(cudaFree(b.d_decomp_ptrs));
  if (b.d_comp_sizes) CUDA_CHECK(cudaFree(b.d_comp_sizes));
  if (b.d_uncomp_sizes) CUDA_CHECK(cudaFree(b.d_uncomp_sizes));
  if (b.d_actual_uncomp_sizes) CUDA_CHECK(cudaFree(b.d_actual_uncomp_sizes));
  if (b.d_statuses) CUDA_CHECK(cudaFree(b.d_statuses));
  b = DeviceBatchPlan{};
}

struct FastLanesResult {
  bool available = false;
  double time_ms = 0.0;
  double speed_rows_per_s = 0.0;
  double throughput_gbps = 0.0;
  double bandwidth_gbps = 0.0;
  double bench = 0.0;
  std::string status = "N/A";
};

static std::vector<std::string> split_csv_line(const std::string& line)
{
  std::vector<std::string> out;
  std::stringstream ss(line);
  std::string item;
  while (std::getline(ss, item, ',')) out.push_back(item);
  return out;
}

static FastLanesResult read_fastlanes_benchmark_txt(const std::string& path)
{
  FastLanesResult r;

  std::ifstream f(path);
  if (!f) {
    r.status = "MISSING";
    return r;
  }

  std::string header;
  std::string values;
  std::getline(f, header);
  std::getline(f, values);

  const auto cols = split_csv_line(values);
  if (cols.size() < 5) {
    r.status = "PARSE_FAILED";
    return r;
  }

  r.time_ms = std::stod(cols[0]);
  r.speed_rows_per_s = std::stod(cols[1]);
  r.throughput_gbps = std::stod(cols[2]);
  r.bandwidth_gbps = std::stod(cols[3]);
  r.bench = std::stod(cols[4]);
  r.available = true;
  r.status = "OK";
  return r;
}

static void print_line(char ch, int n = 118)
{
  for (int i = 0; i < n; ++i) std::cout << ch;
  std::cout << "\n";
}

int main()
{
  try {
    CUDA_CHECK(cudaSetDevice(0));
    std::system("mkdir -p bin");
    std::system("mkdir -p results/nvcomp_lz4_vs_fastlane/csv");
    std::system("mkdir -p results/nvcomp_lz4_vs_fastlane/summary");

    // ============================================================
    // Main experiment configuration
    // ============================================================
    const std::string experiment_name =
        "sorted_sf10_quantity_repeated_23x_5gib";

    const std::string quantity_path =
        "data/baseline_quantity_sf10_sorted_lz4best_repeated/" 
        "quantity_sf10_sorted_lz4best_repeated_23x_5gib.bin"; //update this path later

    // Saved FastLanes output for the same 23x sorted dataset.
    const std::string fastlanes_5gib_benchmark_path =
        "results/baseline_fastlanes_gpu/quantity_sf10_sorted_lz4best_5gib/"
        "fastlanes_gpu/decompress_benchmark.txt";

    // 10 GiB FastLanes status is added to the comparison table intentionally.
    // It failed on RTX A2000 12GB, while nvCOMP LZ4 completed using batching.
    const std::string fastlanes_10gib_status =
        "FAILED: CUDA out-of-memory / illegal memory access on RTX A2000 12GB";

    const double nvcomp_10gib_gbps = 241.764;
    const double nvcomp_10gib_reduction = 99.605;
    const double nvcomp_10gib_ms = 44.661;

    // Previously measured original-order/no-layout-tweak SF10 quantity results.
    // These are included only for comparison in the printed table and CSV.
    const double nvcomp_original_ms = 25.178;
    const double nvcomp_original_gbps = 9.530;
    const double nvcomp_original_reduction = 62.641;

    const double fastlanes_original_ms = 1.119;
    const double fastlanes_original_gbps = 214.340;

    // Terminal colors only. CSV files remain plain text.
    const std::string GREEN = "[32m";
    const std::string RED = "[31m";
    const std::string RESET = "[0m";

    const size_t chunk_bytes = 1ULL << 19;      // 512 KiB
    const size_t chunk_rows = chunk_bytes / sizeof(int);
    const size_t gpu_batch_chunks = 4096;       // 2 GiB decompressed per batch

    const int lz4_hc_level = 6;
    const int warmup = 3;
    const int iterations = 10;

    const std::string nvcomp_csv_path =
        "results/nvcomp_lz4_vs_fastlane/csv/"
        "nvcomp_lz4_vs_fastlane_nvcomp_results.csv";

    const std::string comparison_csv_path =
        "results/nvcomp_lz4_vs_fastlane/csv/"
        "nvcomp_lz4_vs_fastlane_comparison.csv";

    const std::string summary_path =
        "results/nvcomp_lz4_vs_fastlane/summary/"
        "nvcomp_lz4_vs_fastlane_summary.txt";

    std::cout << std::fixed << std::setprecision(3);

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    // ============================================================
    // Load input and compute CPU reference
    // ============================================================
    std::vector<int> host_quantity = read_int_column(quantity_path);
    if (host_quantity.empty()) {
      throw std::runtime_error("Quantity column is empty.");
    }

    const size_t n_rows = host_quantity.size();
    const size_t input_bytes = n_rows * sizeof(int);
    const double input_mib = bytes_to_mib(input_bytes);
    const double input_gib = bytes_to_gib(input_bytes);
    const double input_gb = bytes_to_gb(input_bytes);

    const size_t total_chunks = (n_rows + chunk_rows - 1) / chunk_rows;
    const size_t total_batches =
        (total_chunks + gpu_batch_chunks - 1) / gpu_batch_chunks;

    const unsigned long long reference_sum =
        cpu_sum_quantity(host_quantity.data(), n_rows);

    // ============================================================
    // Preprocess compression. Not counted in timed benchmark.
    // ============================================================
    auto c0 = std::chrono::high_resolution_clock::now();
    CompressedColumn comp = compress_column_lz4_hc(
        host_quantity.data(), n_rows, chunk_rows, lz4_hc_level);
    auto c1 = std::chrono::high_resolution_clock::now();

    const double compress_ms =
        std::chrono::duration<double, std::milli>(c1 - c0).count();

    const double compressed_mib = bytes_to_mib(comp.total_comp_bytes);
    const double compression_pct =
        compression_reduction(input_bytes, comp.total_comp_bytes);

    // ============================================================
    // Flatten compressed chunks once outside timing
    // ============================================================
    std::vector<char> h_comp_flat(comp.total_comp_bytes);
    for (size_t c = 0; c < total_chunks; ++c) {
      std::memcpy(
          h_comp_flat.data() + comp.comp_offsets[c],
          comp.comp_chunks[c].data(),
          comp.comp_sizes[c]);
    }

    for (auto& v : comp.comp_chunks) {
      std::vector<char>().swap(v);
    }

    // ============================================================
    // Build host batch plans
    // ============================================================
    std::vector<HostBatchPlan> host_batches;
    host_batches.reserve(total_batches);

    size_t max_batch_rows = 0;
    size_t max_batch_count = 0;
    size_t max_batch_uncomp_bytes = 0;

    for (size_t first = 0; first < total_chunks; first += gpu_batch_chunks) {
      const size_t count = std::min(gpu_batch_chunks, total_chunks - first);
      HostBatchPlan p =
          build_host_batch_plan(first, count, n_rows, chunk_rows, comp);

      max_batch_rows = std::max(max_batch_rows, p.rows);
      max_batch_count = std::max(max_batch_count, p.count);
      max_batch_uncomp_bytes = std::max(max_batch_uncomp_bytes, p.uncomp_bytes);

      host_batches.push_back(std::move(p));
    }

    // ============================================================
    // Allocate device buffers
    // ============================================================
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));

    char* d_comp_all = nullptr;
    int* d_quantity_batch = nullptr;
    unsigned long long* d_result = nullptr;
    unsigned long long* d_block_sums = nullptr;
    void* d_temp = nullptr;

    CUDA_CHECK(cudaMalloc(&d_comp_all, comp.total_comp_bytes));
    CUDA_CHECK(cudaMalloc(&d_quantity_batch, max_batch_rows * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(unsigned long long)));

    const int block_size = 256;
    const int max_sum_grid = static_cast<int>(
        std::min<size_t>((max_batch_rows + block_size - 1) / block_size, 65535));
    CUDA_CHECK(cudaMalloc(&d_block_sums, max_sum_grid * sizeof(unsigned long long)));

    CUDA_CHECK(cudaMemcpyAsync(
        d_comp_all,
        h_comp_flat.data(),
        comp.total_comp_bytes,
        cudaMemcpyHostToDevice,
        stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<char>().swap(h_comp_flat);

    // ============================================================
    // Build all device batch metadata once outside timing
    // ============================================================
    const nvcompBatchedLZ4DecompressOpts_t opts =
        nvcompBatchedLZ4DecompressDefaultOpts;

    std::vector<DeviceBatchPlan> device_batches(host_batches.size());
    size_t global_temp_bytes = 0;

    for (size_t b = 0; b < host_batches.size(); ++b) {
      const HostBatchPlan& hb = host_batches[b];
      DeviceBatchPlan& db = device_batches[b];

      db.first_chunk = hb.first_chunk;
      db.count = hb.count;
      db.rows = hb.rows;
      db.uncomp_bytes = hb.uncomp_bytes;
      db.comp_bytes = hb.comp_bytes;

      std::vector<void*> h_comp_ptrs(hb.count);
      std::vector<void*> h_decomp_ptrs(hb.count);

      for (size_t k = 0; k < hb.count; ++k) {
        h_comp_ptrs[k] = d_comp_all + hb.global_comp_offsets[k];
        h_decomp_ptrs[k] = d_quantity_batch + hb.local_row_offsets[k];
      }

      CUDA_CHECK(cudaMalloc(&db.d_comp_ptrs, hb.count * sizeof(void*)));
      CUDA_CHECK(cudaMalloc(&db.d_decomp_ptrs, hb.count * sizeof(void*)));
      CUDA_CHECK(cudaMalloc(&db.d_comp_sizes, hb.count * sizeof(size_t)));
      CUDA_CHECK(cudaMalloc(&db.d_uncomp_sizes, hb.count * sizeof(size_t)));
      CUDA_CHECK(cudaMalloc(&db.d_actual_uncomp_sizes, hb.count * sizeof(size_t)));
      CUDA_CHECK(cudaMalloc(&db.d_statuses, hb.count * sizeof(nvcompStatus_t)));

      CUDA_CHECK(cudaMemcpyAsync(
          db.d_comp_ptrs,
          h_comp_ptrs.data(),
          hb.count * sizeof(void*),
          cudaMemcpyHostToDevice,
          stream));
      CUDA_CHECK(cudaMemcpyAsync(
          db.d_decomp_ptrs,
          h_decomp_ptrs.data(),
          hb.count * sizeof(void*),
          cudaMemcpyHostToDevice,
          stream));
      CUDA_CHECK(cudaMemcpyAsync(
          db.d_comp_sizes,
          hb.comp_sizes.data(),
          hb.count * sizeof(size_t),
          cudaMemcpyHostToDevice,
          stream));
      CUDA_CHECK(cudaMemcpyAsync(
          db.d_uncomp_sizes,
          hb.uncomp_sizes.data(),
          hb.count * sizeof(size_t),
          cudaMemcpyHostToDevice,
          stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));

      size_t temp_candidate = 0;
      NVCOMP_CHECK(nvcompBatchedLZ4DecompressGetTempSizeSync(
          (const void* const* const)db.d_comp_ptrs,
          db.d_comp_sizes,
          db.count,
          chunk_bytes,
          &temp_candidate,
          db.uncomp_bytes,
          opts,
          db.d_statuses,
          stream));

      db.temp_bytes = temp_candidate;
      global_temp_bytes = std::max(global_temp_bytes, temp_candidate);
    }

    CUDA_CHECK(cudaMalloc(&d_temp, global_temp_bytes));

    // ============================================================
    // Benchmark A: resident decompression only
    // ============================================================
    cudaEvent_t ev_start{}, ev_end{};
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_end));

    std::vector<double> decomp_ms_runs;

    for (int it = 0; it < warmup + iterations; ++it) {
      CUDA_CHECK(cudaEventRecord(ev_start, stream));

      for (DeviceBatchPlan& db : device_batches) {
        NVCOMP_CHECK(nvcompBatchedLZ4DecompressAsync(
            (const void* const*)db.d_comp_ptrs,
            db.d_comp_sizes,
            db.d_uncomp_sizes,
            db.d_actual_uncomp_sizes,
            db.count,
            d_temp,
            global_temp_bytes,
            db.d_decomp_ptrs,
            opts,
            db.d_statuses,
            stream));
      }

      CUDA_CHECK(cudaEventRecord(ev_end, stream));
      CUDA_CHECK(cudaEventSynchronize(ev_end));

      const double ms = static_cast<double>(elapsed_ms(ev_start, ev_end));
      if (it >= warmup) decomp_ms_runs.push_back(ms);
    }

    const double decomp_ms_avg = mean(decomp_ms_runs);
    const double decomp_ms_std = stddev_sample(decomp_ms_runs);
    const double decomp_gibps = input_gib / (decomp_ms_avg / 1000.0);
    const double decomp_gbps = input_gb / (decomp_ms_avg / 1000.0);

    // ============================================================
    // Benchmark B: resident decompression + GPU SUM
    // ============================================================
    std::vector<double> decomp_sum_ms_runs;
    unsigned long long last_sum = 0ULL;

    for (int it = 0; it < warmup + iterations; ++it) {
      CUDA_CHECK(cudaMemsetAsync(d_result, 0, sizeof(unsigned long long), stream));
      CUDA_CHECK(cudaEventRecord(ev_start, stream));

      for (DeviceBatchPlan& db : device_batches) {
        NVCOMP_CHECK(nvcompBatchedLZ4DecompressAsync(
            (const void* const*)db.d_comp_ptrs,
            db.d_comp_sizes,
            db.d_uncomp_sizes,
            db.d_actual_uncomp_sizes,
            db.count,
            d_temp,
            global_temp_bytes,
            db.d_decomp_ptrs,
            opts,
            db.d_statuses,
            stream));

        const int grid = static_cast<int>(
            std::min<size_t>((db.rows + block_size - 1) / block_size, 65535));

        sum_int_kernel<<<grid, block_size, block_size * sizeof(unsigned long long), stream>>>(
            d_quantity_batch,
            db.rows,
            d_block_sums);
        CUDA_CHECK(cudaGetLastError());

        const int reduce_grid =
            static_cast<int>((static_cast<size_t>(grid) + block_size - 1) / block_size);

        reduce_ull_kernel<<<reduce_grid, block_size,
                            block_size * sizeof(unsigned long long), stream>>>(
            d_block_sums,
            static_cast<size_t>(grid),
            d_result);
        CUDA_CHECK(cudaGetLastError());
      }

      CUDA_CHECK(cudaMemcpyAsync(
          &last_sum,
          d_result,
          sizeof(unsigned long long),
          cudaMemcpyDeviceToHost,
          stream));

      CUDA_CHECK(cudaEventRecord(ev_end, stream));
      CUDA_CHECK(cudaEventSynchronize(ev_end));

      const double ms = static_cast<double>(elapsed_ms(ev_start, ev_end));
      if (it >= warmup) decomp_sum_ms_runs.push_back(ms);
    }

    const double decomp_sum_ms_avg = mean(decomp_sum_ms_runs);
    const double decomp_sum_ms_std = stddev_sample(decomp_sum_ms_runs);
    const double decomp_sum_gibps = input_gib / (decomp_sum_ms_avg / 1000.0);
    const double decomp_sum_gbps = input_gb / (decomp_sum_ms_avg / 1000.0);
    const bool sum_match = (last_sum == reference_sum);

    // ============================================================
    // Exact correctness pass outside timing
    // ============================================================
    bool exact_match = true;
    size_t mismatch_batch = static_cast<size_t>(-1);
    size_t mismatch_row = static_cast<size_t>(-1);

    std::vector<int> verify_buffer(max_batch_rows);

    for (size_t b = 0; b < device_batches.size(); ++b) {
      DeviceBatchPlan& db = device_batches[b];
      const HostBatchPlan& hb = host_batches[b];

      NVCOMP_CHECK(nvcompBatchedLZ4DecompressAsync(
          (const void* const*)db.d_comp_ptrs,
          db.d_comp_sizes,
          db.d_uncomp_sizes,
          db.d_actual_uncomp_sizes,
          db.count,
          d_temp,
          global_temp_bytes,
          db.d_decomp_ptrs,
          opts,
          db.d_statuses,
          stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));

      CUDA_CHECK(cudaMemcpy(
          verify_buffer.data(),
          d_quantity_batch,
          db.rows * sizeof(int),
          cudaMemcpyDeviceToHost));

      for (size_t i = 0; i < db.rows; ++i) {
        const size_t global_row = hb.global_row_offsets[0] + i;
        if (verify_buffer[i] != host_quantity[global_row]) {
          exact_match = false;
          mismatch_batch = b;
          mismatch_row = global_row;
          break;
        }
      }

      if (!exact_match) break;
    }

    // ============================================================
    // Read saved FastLanes result
    // ============================================================
    FastLanesResult fl_5gib =
        read_fastlanes_benchmark_txt(fastlanes_5gib_benchmark_path);

    // ============================================================
    // Clean terminal output
    // ============================================================
    print_line('=');
    std::cout << "nvCOMP LZ4 vs FastLanes-GPU Baseline Comparison\n";
    print_line('=');
    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "Experiment: " << experiment_name << "\n";
    std::cout << "Input: " << quantity_path << "\n";
    std::cout << "Rows: " << n_rows << "\n";
    std::cout << "Input MiB: " << input_mib << "\n";
    std::cout << "Chunks: " << total_chunks << "\n";
    std::cout << "Chunk KiB: " << static_cast<double>(chunk_bytes) / 1024.0 << "\n";
    std::cout << "GPU batch chunks: " << gpu_batch_chunks << "\n";
    std::cout << "Batches: " << total_batches << "\n";
    std::cout << "LZ4_HC level: " << lz4_hc_level << "\n";
    std::cout << "Warmup: " << warmup << ", timed iterations: " << iterations << "\n\n";

    std::cout << "Compression:\n";
    std::cout << std::left
              << std::setw(18) << "Original MiB"
              << std::setw(18) << "Compressed MiB"
              << std::setw(16) << "Reduction %"
              << std::setw(18) << "Compress ms"
              << "\n";
    std::cout << std::left
              << std::setw(18) << input_mib
              << std::setw(18) << compressed_mib
              << std::setw(16) << compression_pct
              << std::setw(18) << compress_ms
              << "\n\n";

    std::cout << "Own nvCOMP results:\n";
    std::cout << std::left
              << std::setw(30) << "Benchmark"
              << std::setw(12) << "Avg ms"
              << std::setw(12) << "Std ms"
              << std::setw(12) << "GiB/s"
              << std::setw(12) << "GB/s"
              << std::setw(10) << "Correct"
              << "\n";
    std::cout << std::left
              << std::setw(30) << "Decompression only"
              << std::setw(12) << decomp_ms_avg
              << std::setw(12) << decomp_ms_std
              << std::setw(12) << decomp_gibps
              << std::setw(12) << decomp_gbps
              << std::setw(10) << (exact_match ? "YES" : "NO")
              << "\n";
    std::cout << std::left
              << std::setw(30) << "Decompression + SUM"
              << std::setw(12) << decomp_sum_ms_avg
              << std::setw(12) << decomp_sum_ms_std
              << std::setw(12) << decomp_sum_gibps
              << std::setw(12) << decomp_sum_gbps
              << std::setw(10) << (sum_match ? "YES" : "NO")
              << "\n\n";

    std::cout << "FastLanes saved result:\n";
    std::cout << "  File: " << fastlanes_5gib_benchmark_path << "\n";
    if (fl_5gib.available) {
      std::cout << "  Time ms: " << fl_5gib.time_ms << "\n";
      std::cout << "  Throughput GB/s: " << fl_5gib.throughput_gbps << "\n";
      std::cout << "  Bandwidth GB/s: " << fl_5gib.bandwidth_gbps << "\n\n";
    } else {
      std::cout << "  Status: " << fl_5gib.status << "\n\n";
    }

    auto print_comparison_row = [&](const std::string& color,
                                    const std::string& dataset,
                                    const std::string& layout,
                                    const std::string& system,
                                    const std::string& avg_ms,
                                    const std::string& gbps,
                                    const std::string& reduction,
                                    const std::string& status) {
      std::cout << color << std::left
                << std::setw(32) << dataset
                << std::setw(16) << layout
                << std::setw(18) << system
                << std::setw(14) << avg_ms
                << std::setw(14) << gbps
                << std::setw(16) << reduction
                << std::setw(24) << status
                << RESET << std::endl;
    };

    auto dstr = [](double x) {
      std::ostringstream os;
      os << std::fixed << std::setprecision(3) << x;
      return os.str();
    };

    std::cout << "Comparison table:" << std::endl;
    std::cout << std::left
              << std::setw(32) << "Dataset"
              << std::setw(16) << "Layout"
              << std::setw(18) << "System"
              << std::setw(14) << "Avg ms"
              << std::setw(14) << "GB/s"
              << std::setw(16) << "Reduction %"
              << std::setw(24) << "Status"
              << std::endl;
    print_line('-', 134);

    print_comparison_row(GREEN, "SF10 quantity 1x", "No tweak", "nvCOMP LZ4",
                         dstr(nvcomp_original_ms), dstr(nvcomp_original_gbps),
                         dstr(nvcomp_original_reduction), "OK");

    print_comparison_row(RED, "SF10 quantity 1x", "No tweak", "FastLanes-GPU",
                         dstr(fastlanes_original_ms), dstr(fastlanes_original_gbps),
                         "packed", "OK");

    print_comparison_row(GREEN, "SF10 quantity 23x", "Sorted", "nvCOMP LZ4",
                         dstr(decomp_ms_avg), dstr(decomp_gbps),
                         dstr(compression_pct), exact_match ? "OK" : "CHECK_FAILED");

    print_comparison_row(RED, "SF10 quantity 23x", "Sorted", "FastLanes-GPU",
                         fl_5gib.available ? dstr(fl_5gib.time_ms) : "N/A",
                         fl_5gib.available ? dstr(fl_5gib.throughput_gbps) : "N/A",
                         "packed", fl_5gib.status);

    print_comparison_row(GREEN, "SF10 quantity 45x", "Sorted", "nvCOMP LZ4",
                         dstr(nvcomp_10gib_ms), dstr(nvcomp_10gib_gbps),
                         dstr(nvcomp_10gib_reduction), "OK");

    print_comparison_row(RED, "SF10 quantity 45x", "Sorted", "FastLanes-GPU",
                         "N/A", "N/A", "packed", "FAILED on 12GB GPU");

    print_line('-');

    if (!exact_match) {
      std::cout << "First mismatch batch: " << mismatch_batch
                << ", global row: " << mismatch_row << "\n";
    }

    // ============================================================
    // CSV output: own nvCOMP result
    // ============================================================
    std::ofstream csv(nvcomp_csv_path);
    csv << "Experiment,Input_Path,Input_MiB,Input_GiB,Input_GB,Rows,Chunks,Chunk_KiB,"
        << "GPU_Batch_Chunks,Batches,LZ4_HC_Level,Compressed_MiB,Compression_Reduction_Percent,"
        << "Preprocess_Compress_ms_Not_Counted,Resident_Decomp_ms_Avg,Resident_Decomp_ms_StdDev,"
        << "Resident_Decomp_GiBps,Resident_Decomp_GBps,Resident_Decomp_SUM_ms_Avg,"
        << "Resident_Decomp_SUM_ms_StdDev,Resident_Decomp_SUM_GiBps,Resident_Decomp_SUM_GBps,"
        << "Reference_SUM,GPU_SUM,SUM_Match,Exact_Match\n";

    csv << experiment_name << ","
        << quantity_path << ","
        << input_mib << ","
        << input_gib << ","
        << input_gb << ","
        << n_rows << ","
        << total_chunks << ","
        << static_cast<double>(chunk_bytes) / 1024.0 << ","
        << gpu_batch_chunks << ","
        << total_batches << ","
        << lz4_hc_level << ","
        << compressed_mib << ","
        << compression_pct << ","
        << compress_ms << ","
        << decomp_ms_avg << ","
        << decomp_ms_std << ","
        << decomp_gibps << ","
        << decomp_gbps << ","
        << decomp_sum_ms_avg << ","
        << decomp_sum_ms_std << ","
        << decomp_sum_gibps << ","
        << decomp_sum_gbps << ","
        << reference_sum << ","
        << last_sum << ","
        << (sum_match ? "YES" : "NO") << ","
        << (exact_match ? "YES" : "NO") << "\n";
    csv.close();

    // ============================================================
    // CSV output: combined comparison table
    // ============================================================
    std::ofstream comparison_csv(comparison_csv_path);
    comparison_csv << "Dataset,Layout,System,Avg_ms,Throughput_GBps,Compression_or_Encoding,Reduction_Percent,Status,Source";
    comparison_csv << "SF10 quantity 1x,No tweak,nvCOMP LZ4,"
                   << nvcomp_original_ms << ","
                   << nvcomp_original_gbps << ",LZ4_HC_level_6,"
                   << nvcomp_original_reduction << ",OK,previous original-order SF10 run";
    comparison_csv << "SF10 quantity 1x,No tweak,FastLanes-GPU,"
                   << fastlanes_original_ms << ","
                   << fastlanes_original_gbps << ",packed,,OK,previous original-order SF10 FastLanes run";
    comparison_csv << "Sorted SF10 quantity 23x,Sorted,nvCOMP LZ4,"
                   << decomp_ms_avg << ","
                   << decomp_gbps << ",LZ4_HC_level_" << lz4_hc_level << ","
                   << compression_pct << ",OK," << nvcomp_csv_path << "\n";
    comparison_csv << "Sorted SF10 quantity 23x,Sorted,FastLanes-GPU,"
                   << (fl_5gib.available ? std::to_string(fl_5gib.time_ms) : "") << ","
                   << (fl_5gib.available ? std::to_string(fl_5gib.throughput_gbps) : "") << ",packed,"
                   << "," << fl_5gib.status << "," << fastlanes_5gib_benchmark_path << "\n";
    comparison_csv << "Sorted SF10 quantity 45x,Sorted,nvCOMP LZ4,"
                   << nvcomp_10gib_ms << ","
                   << nvcomp_10gib_gbps << ",LZ4_HC_level_6,"
                   << nvcomp_10gib_reduction << ",OK,previous successful 10GiB run\n";
    comparison_csv << "Sorted SF10 quantity 45x,Sorted,FastLanes-GPU,,,,packed,,"
                   << fastlanes_10gib_status << ",manual observed run failure\n";
    comparison_csv.close();

    // ============================================================
    // Summary text output
    // ============================================================
    std::ofstream summary(summary_path);
    summary << "nvCOMP LZ4 vs FastLanes-GPU Baseline Comparison\n";
    summary << "================================================\n";
    summary << "GPU: " << prop.name << "\n";
    summary << "Experiment: " << experiment_name << "\n";
    summary << "Input: " << quantity_path << "\n";
    summary << "Rows: " << n_rows << "\n";
    summary << "Input MiB: " << input_mib << "\n";
    summary << "Compressed MiB: " << compressed_mib << "\n";
    summary << "Compression reduction %: " << compression_pct << "\n";
    summary << "nvCOMP decompression-only avg ms: " << decomp_ms_avg << "\n";
    summary << "nvCOMP decompression-only GB/s: " << decomp_gbps << "\n";
    summary << "nvCOMP decompression+SUM avg ms: " << decomp_sum_ms_avg << "\n";
    summary << "nvCOMP decompression+SUM GB/s: " << decomp_sum_gbps << "\n";
    summary << "FastLanes 23x benchmark file: " << fastlanes_5gib_benchmark_path << "\n";
    summary << "FastLanes 23x status: " << fl_5gib.status << "\n";
    if (fl_5gib.available) {
      summary << "FastLanes 23x avg ms: " << fl_5gib.time_ms << "\n";
      summary << "FastLanes 23x throughput GB/s: " << fl_5gib.throughput_gbps << "\n";
    }
    summary << "FastLanes 45x/10GiB status: " << fastlanes_10gib_status << "\n";
    summary << "SUM match: " << (sum_match ? "YES" : "NO") << "\n";
    summary << "Exact byte/int match: " << (exact_match ? "YES" : "NO") << "\n";
    summary.close();

    std::cout << "\nCSV files written:\n";
    std::cout << "  " << nvcomp_csv_path << "\n";
    std::cout << "  " << comparison_csv_path << "\n";
    std::cout << "Summary written:\n";
    std::cout << "  " << summary_path << "\n";
    print_line('=');

    // ============================================================
    // Cleanup
    // ============================================================
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_end));

    for (auto& db : device_batches) destroy_device_batch(db);

    CUDA_CHECK(cudaFree(d_comp_all));
    CUDA_CHECK(cudaFree(d_quantity_batch));
    CUDA_CHECK(cudaFree(d_result));
    CUDA_CHECK(cudaFree(d_block_sums));
    CUDA_CHECK(cudaFree(d_temp));
    CUDA_CHECK(cudaStreamDestroy(stream));

    return exact_match && sum_match ? 0 : 2;
  }
  catch (const std::exception& e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    return 1;
  }
}


// d=results/baseline_fastlanes_gpu/quantity_sf10_sorted_lz4best_23x; rm -rf "$d/fastlanes_gpu"; bash external/scripts/fastlanes_gpu/compress.sh "$d"; bash external/scripts/fastlanes_gpu/decompress.sh "$d"; rm -f "$d/fastlanes_gpu/decompressed.dat"
// cat results/baseline_fastlanes_gpu/quantity_sf10_sorted_lz4best_23x/fastlanes_gpu/decompress_benchmark.txt

// nvcc -std=c++17 -O3   -I ~/nvcomp_env/lib/python3.12/site-packages/nvidia/libnvcomp/include   src/benchmark/nvcomp_lz4_vs_fastlane.cu   -L ~/nvcomp_env/lib/python3.12/site-packages/nvidia/libnvcomp/lib64   -lnvcomp -llz4   -o bin/nvcomp_lz4_vs_fastlane
// LD_LIBRARY_PATH=~/nvcomp_env/lib/python3.12/site-packages/nvidia/libnvcomp/lib64:$LD_LIBRARY_PATH ./bin/nvcomp_lz4_vs_fastlane