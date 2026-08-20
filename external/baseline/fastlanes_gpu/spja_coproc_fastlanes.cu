#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "fls_gen/unpack/unpack.cuh"
#include "./common.hpp"

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " -> "   \
                << cudaGetErrorString(err__) << std::endl;                     \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

namespace {

constexpr int kDecodeThreads = 32;
constexpr int kDecodeItemsPerThread = 32;
constexpr int kGpuQueryThreads = 256;

static_assert(kDecodeThreads * kDecodeItemsPerThread == kVecSize,
              "Expected FastLanes vector size 1024.");

struct ColumnData {
  std::string name;
  std::filesystem::path root_dir;
  Metadata meta;
  std::vector<uint32_t> compressed;
  size_t raw_bytes = 0;
  size_t compressed_bytes = 0;
};

struct SplitResult {
  int cpu_percent = 0;
  int gpu_percent = 0;

  size_t cpu_vecs = 0;
  size_t gpu_vecs = 0;
  size_t cpu_rows = 0;
  size_t gpu_rows = 0;

  double cpu_ms = 0.0;
  double gpu_ms = 0.0;
  double total_ms = 0.0;
  double eff_gib_s = 0.0;

  unsigned long long cpu_sum = 0ULL;
  unsigned long long gpu_sum = 0ULL;
  unsigned long long total_sum = 0ULL;
  unsigned long long reference_sum = 0ULL;

  bool valid = false;
};

double bytes_to_mib(size_t bytes) {
  return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

double bytes_to_gib(size_t bytes) {
  return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
}

std::vector<unsigned char> read_binary_file(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Could not open file: " + path.string());
  }

  file.seekg(0, std::ios::end);
  const size_t bytes = static_cast<size_t>(file.tellg());
  file.seekg(0, std::ios::beg);

  std::vector<unsigned char> buffer(bytes);
  if (bytes > 0) {
    file.read(reinterpret_cast<char*>(buffer.data()),
              static_cast<std::streamsize>(bytes));
  }

  if (static_cast<size_t>(file.gcount()) != bytes) {
    throw std::runtime_error("Could not read full file: " + path.string());
  }

  return buffer;
}

std::vector<int> read_int_file(const std::filesystem::path& path) {
  auto bytes = read_binary_file(path);

  if (bytes.size() % sizeof(int) != 0) {
    throw std::runtime_error("Invalid int file size: " + path.string());
  }

  std::vector<int> out(bytes.size() / sizeof(int));

  if (!out.empty()) {
    std::memcpy(out.data(), bytes.data(), bytes.size());
  }

  return out;
}

std::vector<uint32_t> read_u32_file(const std::filesystem::path& path) {
  auto bytes = read_binary_file(path);

  if (bytes.size() % sizeof(uint32_t) != 0) {
    throw std::runtime_error("Invalid uint32 file size: " + path.string());
  }

  std::vector<uint32_t> out(bytes.size() / sizeof(uint32_t));

  if (!out.empty()) {
    std::memcpy(out.data(), bytes.data(), bytes.size());
  }

  return out;
}

ColumnData load_column(const std::string& name,
                       const std::filesystem::path& root_dir) {
  ColumnData col;
  col.name = name;
  col.root_dir = root_dir;
  col.meta = Metadata::read(root_dir / "fastlanes_gpu" / "metadata.txt");
  col.compressed = read_u32_file(root_dir / "fastlanes_gpu" / "data.dat");
  col.raw_bytes = std::filesystem::file_size(root_dir / "raw.dat");
  col.compressed_bytes = col.compressed.size() * sizeof(uint32_t);

  if (col.meta.scheme != Metadata::unpack) {
    throw std::runtime_error(
        "Only FastLanes unpack scheme is supported in this SPJA version.");
  }

  return col;
}

uint32_t fastlanes_unpack_lane_value_cpu(const uint32_t* encoded_vec,
                                         int bitwidth,
                                         int lane,
                                         int item) {
  if (bitwidth == 0) {
    return 0;
  }

  if (bitwidth == 32) {
    return encoded_vec[static_cast<size_t>(item) * 32ULL +
                       static_cast<size_t>(lane)];
  }

  const size_t bit_pos =
      static_cast<size_t>(item) * static_cast<size_t>(bitwidth);
  const size_t word = bit_pos / 32ULL;
  const int shift = static_cast<int>(bit_pos % 32ULL);

  uint64_t packed =
      static_cast<uint64_t>(
          encoded_vec[word * 32ULL + static_cast<size_t>(lane)]);

  if (shift + bitwidth > 32) {
    packed |=
        static_cast<uint64_t>(
            encoded_vec[(word + 1ULL) * 32ULL + static_cast<size_t>(lane)])
        << 32ULL;
  }

  const uint64_t mask =
      (1ULL << static_cast<unsigned>(bitwidth)) - 1ULL;

  return static_cast<uint32_t>((packed >> shift) & mask);
}

int fastlanes_value_cpu(const ColumnData& col, size_t global_row) {
  const size_t vec = global_row / kVecSize;
  const size_t local = global_row % kVecSize;

  const int item = static_cast<int>(local / 32ULL);
  const int lane = static_cast<int>(local % 32ULL);

  const uint32_t* encoded_vec =
      col.compressed.data() +
      vec * static_cast<size_t>(col.meta.bitwidth) * 32ULL;

  return static_cast<int>(
      fastlanes_unpack_lane_value_cpu(
          encoded_vec,
          col.meta.bitwidth,
          lane,
          item));
}

unsigned long long spja_cpu_raw_reference(
    const int* orderkey,
    const int* quantity,
    const int* extendedprice,
    const int* order_custkey,
    const int* customer_nation,
    size_t n_rows,
    int order_count,
    int customer_count,
    int target_nation) {
  unsigned long long sum = 0ULL;

  for (size_t i = 0; i < n_rows; ++i) {
    const int ok = orderkey[i];

    if (ok <= 0 || ok >= order_count) {
      continue;
    }

    const int custkey = order_custkey[ok];

    if (custkey <= 0 || custkey >= customer_count) {
      continue;
    }

    const int nation = customer_nation[custkey];

    if (quantity[i] > 25 && nation == target_nation) {
      sum += static_cast<unsigned long long>(extendedprice[i]);
    }
  }

  return sum;
}

unsigned long long spja_cpu_fastlanes_range_single_thread(
    const ColumnData& orderkey_col,
    const ColumnData& quantity_col,
    const ColumnData& extendedprice_col,
    const int* order_custkey,
    const int* customer_nation,
    size_t start_row,
    size_t end_row,
    int order_count,
    int customer_count,
    int target_nation) {
  unsigned long long sum = 0ULL;

  for (size_t i = start_row; i < end_row; ++i) {
    const int ok = fastlanes_value_cpu(orderkey_col, i);

    if (ok <= 0 || ok >= order_count) {
      continue;
    }

    const int custkey = order_custkey[ok];

    if (custkey <= 0 || custkey >= customer_count) {
      continue;
    }

    const int nation = customer_nation[custkey];

    if (nation != target_nation) {
      continue;
    }

    if (fastlanes_value_cpu(quantity_col, i) > 25) {
      sum += static_cast<unsigned long long>(
          fastlanes_value_cpu(extendedprice_col, i));
    }
  }

  return sum;
}

unsigned long long spja_cpu_fastlanes_range_parallel(
    const ColumnData& orderkey_col,
    const ColumnData& quantity_col,
    const ColumnData& extendedprice_col,
    const int* order_custkey,
    const int* customer_nation,
    size_t start_vec,
    size_t num_vecs,
    size_t n_rows,
    int order_count,
    int customer_count,
    int target_nation,
    int worker_threads) {
  const size_t start_row = start_vec * static_cast<size_t>(kVecSize);
  const size_t end_row =
      std::min(n_rows,
               (start_vec + num_vecs) * static_cast<size_t>(kVecSize));

  if (start_row >= end_row) {
    return 0ULL;
  }

  const size_t total_rows = end_row - start_row;
  const int threads = std::max(1, std::min(worker_threads,
                                           static_cast<int>(total_rows)));

  std::vector<std::thread> workers;
  std::vector<unsigned long long> partial(threads, 0ULL);

  workers.reserve(threads);

  for (int t = 0; t < threads; ++t) {
    const size_t local_start = start_row +
        (total_rows * static_cast<size_t>(t)) / static_cast<size_t>(threads);
    const size_t local_end = start_row +
        (total_rows * static_cast<size_t>(t + 1)) / static_cast<size_t>(threads);

    workers.emplace_back([&, t, local_start, local_end]() {
      partial[t] =
          spja_cpu_fastlanes_range_single_thread(
              orderkey_col,
              quantity_col,
              extendedprice_col,
              order_custkey,
              customer_nation,
              local_start,
              local_end,
              order_count,
              customer_count,
              target_nation);
    });
  }

  for (auto& worker : workers) {
    worker.join();
  }

  return std::accumulate(partial.begin(), partial.end(), 0ULL);
}

template <int BlockThreads, int ItemsPerThread>
__global__ void decode_fastlanes_unpack_kernel(
    const uint32_t* __restrict__ encoded,
    size_t num_vecs,
    int bitwidth,
    int* __restrict__ out) {
  const size_t vec = blockIdx.x;

  if (vec >= num_vecs) {
    return;
  }

  const size_t encoded_offset =
      vec * static_cast<size_t>(bitwidth) * 32ULL;

  const size_t tile_offset =
      vec * static_cast<size_t>(kVecSize);

  unpack_device(
      encoded + encoded_offset,
      reinterpret_cast<uint32_t*>(out + tile_offset),
      bitwidth);
}

__global__ void spja_gpu_kernel(
    const int* orderkey,
    const int* quantity,
    const int* extendedprice,
    const int* order_custkey,
    const int* customer_nation,
    size_t n_rows,
    int order_count,
    int customer_count,
    int target_nation,
    unsigned long long* block_sums) {
  extern __shared__ unsigned long long shared_sum[];

  const unsigned int tid = threadIdx.x;
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  unsigned long long local_sum = 0ULL;

  if (i < n_rows) {
    const int ok = orderkey[i];

    if (ok > 0 && ok < order_count) {
      const int custkey = order_custkey[ok];

      if (custkey > 0 && custkey < customer_count) {
        const int nation = customer_nation[custkey];

        if (quantity[i] > 25 && nation == target_nation) {
          local_sum = static_cast<unsigned long long>(extendedprice[i]);
        }
      }
    }
  }

  shared_sum[tid] = local_sum;
  __syncthreads();

  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared_sum[tid] += shared_sum[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    block_sums[blockIdx.x] = shared_sum[0];
  }
}

template <typename T>
void cuda_malloc_bytes(T** ptr, size_t bytes) {
  if (bytes == 0) {
    *ptr = nullptr;
    return;
  }

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(ptr), bytes));
}

size_t column_words_for_vecs(const ColumnData& col, size_t vecs) {
  return vecs * static_cast<size_t>(col.meta.bitwidth) * 32ULL;
}

struct GpuBuffers {
  uint32_t* d_orderkey_comp = nullptr;
  uint32_t* d_quantity_comp = nullptr;
  uint32_t* d_extendedprice_comp = nullptr;

  int* d_orderkey = nullptr;
  int* d_quantity = nullptr;
  int* d_extendedprice = nullptr;

  unsigned long long* d_block_sums = nullptr;

  size_t orderkey_comp_bytes = 0;
  size_t quantity_comp_bytes = 0;
  size_t extendedprice_comp_bytes = 0;

  size_t decoded_rows_capacity = 0;
  size_t num_blocks = 0;
};

void free_gpu_buffers(GpuBuffers& b) {
  if (b.d_orderkey_comp) CUDA_CHECK(cudaFree(b.d_orderkey_comp));
  if (b.d_quantity_comp) CUDA_CHECK(cudaFree(b.d_quantity_comp));
  if (b.d_extendedprice_comp) CUDA_CHECK(cudaFree(b.d_extendedprice_comp));

  if (b.d_orderkey) CUDA_CHECK(cudaFree(b.d_orderkey));
  if (b.d_quantity) CUDA_CHECK(cudaFree(b.d_quantity));
  if (b.d_extendedprice) CUDA_CHECK(cudaFree(b.d_extendedprice));

  if (b.d_block_sums) CUDA_CHECK(cudaFree(b.d_block_sums));

  b = GpuBuffers{};
}

GpuBuffers allocate_gpu_buffers(
    const ColumnData& orderkey_col,
    const ColumnData& quantity_col,
    const ColumnData& extendedprice_col,
    size_t gpu_vecs,
    size_t gpu_rows_capacity) {
  GpuBuffers b;

  b.orderkey_comp_bytes =
      column_words_for_vecs(orderkey_col, gpu_vecs) * sizeof(uint32_t);
  b.quantity_comp_bytes =
      column_words_for_vecs(quantity_col, gpu_vecs) * sizeof(uint32_t);
  b.extendedprice_comp_bytes =
      column_words_for_vecs(extendedprice_col, gpu_vecs) * sizeof(uint32_t);

  b.decoded_rows_capacity = gpu_rows_capacity;
  b.num_blocks =
      (gpu_rows_capacity + kGpuQueryThreads - 1) / kGpuQueryThreads;

  cuda_malloc_bytes(&b.d_orderkey_comp, b.orderkey_comp_bytes);
  cuda_malloc_bytes(&b.d_quantity_comp, b.quantity_comp_bytes);
  cuda_malloc_bytes(&b.d_extendedprice_comp, b.extendedprice_comp_bytes);

  cuda_malloc_bytes(&b.d_orderkey, gpu_rows_capacity * sizeof(int));
  cuda_malloc_bytes(&b.d_quantity, gpu_rows_capacity * sizeof(int));
  cuda_malloc_bytes(&b.d_extendedprice, gpu_rows_capacity * sizeof(int));

  cuda_malloc_bytes(&b.d_block_sums,
                    b.num_blocks * sizeof(unsigned long long));

  return b;
}

struct GpuRunResult {
  double gpu_ms = 0.0;
  unsigned long long gpu_sum = 0ULL;
};

GpuRunResult run_gpu_once(
    const ColumnData& orderkey_col,
    const ColumnData& quantity_col,
    const ColumnData& extendedprice_col,
    size_t gpu_start_vec,
    size_t gpu_vecs,
    size_t gpu_rows,
    const int* d_order_custkey,
    const int* d_customer_nation,
    int order_count,
    int customer_count,
    int target_nation,
    GpuBuffers& b,
    std::vector<unsigned long long>& host_block_sums,
    cudaStream_t stream) {
  GpuRunResult result;

  if (gpu_vecs == 0 || gpu_rows == 0) {
    return result;
  }

  const size_t orderkey_offset =
      gpu_start_vec * static_cast<size_t>(orderkey_col.meta.bitwidth) * 32ULL;
  const size_t quantity_offset =
      gpu_start_vec * static_cast<size_t>(quantity_col.meta.bitwidth) * 32ULL;
  const size_t extendedprice_offset =
      gpu_start_vec * static_cast<size_t>(extendedprice_col.meta.bitwidth) * 32ULL;

  cudaEvent_t start_event;
  cudaEvent_t stop_event;

  CUDA_CHECK(cudaEventCreate(&start_event));
  CUDA_CHECK(cudaEventCreate(&stop_event));

  CUDA_CHECK(cudaEventRecord(start_event, stream));

  CUDA_CHECK(cudaMemcpyAsync(
      b.d_orderkey_comp,
      orderkey_col.compressed.data() + orderkey_offset,
      b.orderkey_comp_bytes,
      cudaMemcpyHostToDevice,
      stream));

  CUDA_CHECK(cudaMemcpyAsync(
      b.d_quantity_comp,
      quantity_col.compressed.data() + quantity_offset,
      b.quantity_comp_bytes,
      cudaMemcpyHostToDevice,
      stream));

  CUDA_CHECK(cudaMemcpyAsync(
      b.d_extendedprice_comp,
      extendedprice_col.compressed.data() + extendedprice_offset,
      b.extendedprice_comp_bytes,
      cudaMemcpyHostToDevice,
      stream));

  decode_fastlanes_unpack_kernel<kDecodeThreads, kDecodeItemsPerThread>
      <<<static_cast<unsigned int>(gpu_vecs), kDecodeThreads, 0, stream>>>(
          b.d_orderkey_comp,
          gpu_vecs,
          orderkey_col.meta.bitwidth,
          b.d_orderkey);

  decode_fastlanes_unpack_kernel<kDecodeThreads, kDecodeItemsPerThread>
      <<<static_cast<unsigned int>(gpu_vecs), kDecodeThreads, 0, stream>>>(
          b.d_quantity_comp,
          gpu_vecs,
          quantity_col.meta.bitwidth,
          b.d_quantity);

  decode_fastlanes_unpack_kernel<kDecodeThreads, kDecodeItemsPerThread>
      <<<static_cast<unsigned int>(gpu_vecs), kDecodeThreads, 0, stream>>>(
          b.d_extendedprice_comp,
          gpu_vecs,
          extendedprice_col.meta.bitwidth,
          b.d_extendedprice);

  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemsetAsync(
      b.d_block_sums,
      0,
      b.num_blocks * sizeof(unsigned long long),
      stream));

  spja_gpu_kernel<<<static_cast<unsigned int>(b.num_blocks),
                    kGpuQueryThreads,
                    kGpuQueryThreads * sizeof(unsigned long long),
                    stream>>>(
      b.d_orderkey,
      b.d_quantity,
      b.d_extendedprice,
      d_order_custkey,
      d_customer_nation,
      gpu_rows,
      order_count,
      customer_count,
      target_nation,
      b.d_block_sums);

  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpyAsync(
      host_block_sums.data(),
      b.d_block_sums,
      b.num_blocks * sizeof(unsigned long long),
      cudaMemcpyDeviceToHost,
      stream));

  CUDA_CHECK(cudaEventRecord(stop_event, stream));
  CUDA_CHECK(cudaEventSynchronize(stop_event));

  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));

  CUDA_CHECK(cudaEventDestroy(start_event));
  CUDA_CHECK(cudaEventDestroy(stop_event));

  result.gpu_ms = static_cast<double>(elapsed_ms);
  result.gpu_sum =
      std::accumulate(host_block_sums.begin(), host_block_sums.end(), 0ULL);

  return result;
}

SplitResult run_split(
    int gpu_percent,
    const ColumnData& orderkey_col,
    const ColumnData& quantity_col,
    const ColumnData& extendedprice_col,
    const int* order_custkey,
    const int* customer_nation,
    const int* d_order_custkey,
    const int* d_customer_nation,
    size_t n_rows,
    size_t n_vecs,
    int order_count,
    int customer_count,
    int target_nation,
    unsigned long long reference_sum,
    int warmup,
    int runs,
    int cpu_worker_threads,
    size_t total_query_bytes,
    cudaStream_t stream) {
  SplitResult final_result;
  final_result.gpu_percent = gpu_percent;
  final_result.cpu_percent = 100 - gpu_percent;

  final_result.gpu_vecs =
      (n_vecs * static_cast<size_t>(gpu_percent)) / 100ULL;
  final_result.cpu_vecs = n_vecs - final_result.gpu_vecs;

  const size_t cpu_start_vec = 0;
  const size_t gpu_start_vec = final_result.cpu_vecs;

  final_result.cpu_rows =
      std::min(n_rows,
               final_result.cpu_vecs * static_cast<size_t>(kVecSize));

  if (final_result.gpu_vecs > 0) {
    const size_t gpu_start_row =
        gpu_start_vec * static_cast<size_t>(kVecSize);

    final_result.gpu_rows =
        std::min(n_rows - gpu_start_row,
                 final_result.gpu_vecs * static_cast<size_t>(kVecSize));
  }

  GpuBuffers gpu_buffers;
  std::vector<unsigned long long> host_block_sums;

  if (final_result.gpu_vecs > 0 && final_result.gpu_rows > 0) {
    const size_t gpu_rows_capacity =
        final_result.gpu_vecs * static_cast<size_t>(kVecSize);

    gpu_buffers =
        allocate_gpu_buffers(
            orderkey_col,
            quantity_col,
            extendedprice_col,
            final_result.gpu_vecs,
            gpu_rows_capacity);

    host_block_sums.resize(gpu_buffers.num_blocks);
  }

  auto run_once = [&]() -> SplitResult {
    SplitResult r = final_result;

    unsigned long long cpu_sum = 0ULL;
    unsigned long long gpu_sum = 0ULL;

    double cpu_ms = 0.0;
    double gpu_ms = 0.0;

    const auto total_start = std::chrono::high_resolution_clock::now();

    std::thread cpu_thread;

    if (r.cpu_vecs > 0) {
      cpu_thread = std::thread([&]() {
        const auto cpu_start = std::chrono::high_resolution_clock::now();

        cpu_sum =
            spja_cpu_fastlanes_range_parallel(
                orderkey_col,
                quantity_col,
                extendedprice_col,
                order_custkey,
                customer_nation,
                cpu_start_vec,
                r.cpu_vecs,
                n_rows,
                order_count,
                customer_count,
                target_nation,
                cpu_worker_threads);

        const auto cpu_end = std::chrono::high_resolution_clock::now();

        cpu_ms =
            std::chrono::duration<double, std::milli>(
                cpu_end - cpu_start).count();
      });
    }

    if (r.gpu_vecs > 0 && r.gpu_rows > 0) {
      GpuRunResult gpu_result =
          run_gpu_once(
              orderkey_col,
              quantity_col,
              extendedprice_col,
              gpu_start_vec,
              r.gpu_vecs,
              r.gpu_rows,
              d_order_custkey,
              d_customer_nation,
              order_count,
              customer_count,
              target_nation,
              gpu_buffers,
              host_block_sums,
              stream);

      gpu_sum = gpu_result.gpu_sum;
      gpu_ms = gpu_result.gpu_ms;
    }

    if (cpu_thread.joinable()) {
      cpu_thread.join();
    }

    const auto total_end = std::chrono::high_resolution_clock::now();

    r.cpu_ms = cpu_ms;
    r.gpu_ms = gpu_ms;
    r.total_ms =
        std::chrono::duration<double, std::milli>(
            total_end - total_start).count();

    r.cpu_sum = cpu_sum;
    r.gpu_sum = gpu_sum;
    r.total_sum = cpu_sum + gpu_sum;
    r.reference_sum = reference_sum;
    r.valid = (r.total_sum == reference_sum);

    r.eff_gib_s =
        bytes_to_gib(total_query_bytes) / (r.total_ms / 1000.0);

    return r;
  };

  for (int i = 0; i < warmup; ++i) {
    (void)run_once();
  }

  double cpu_ms_sum = 0.0;
  double gpu_ms_sum = 0.0;
  double total_ms_sum = 0.0;
  double gib_s_sum = 0.0;

  SplitResult last;

  for (int i = 0; i < runs; ++i) {
    SplitResult r = run_once();
    last = r;

    cpu_ms_sum += r.cpu_ms;
    gpu_ms_sum += r.gpu_ms;
    total_ms_sum += r.total_ms;
    gib_s_sum += r.eff_gib_s;
  }

  last.cpu_ms = cpu_ms_sum / static_cast<double>(runs);
  last.gpu_ms = gpu_ms_sum / static_cast<double>(runs);
  last.total_ms = total_ms_sum / static_cast<double>(runs);
  last.eff_gib_s = gib_s_sum / static_cast<double>(runs);

  free_gpu_buffers(gpu_buffers);

  return last;
}

void print_compression_table(
    const ColumnData& orderkey_col,
    const ColumnData& quantity_col,
    const ColumnData& extendedprice_col) {
  const std::vector<const ColumnData*> cols = {
      &orderkey_col,
      &quantity_col,
      &extendedprice_col
  };

  size_t total_raw = 0;
  size_t total_comp = 0;

  for (const ColumnData* c : cols) {
    total_raw += c->raw_bytes;
    total_comp += c->compressed_bytes;
  }

  std::cout << "\nPreprocessing compression:\n";
  std::cout << "  Original MiB:   " << bytes_to_mib(total_raw) << "\n";
  std::cout << "  Compressed MiB: " << bytes_to_mib(total_comp) << "\n";
  std::cout << "  Compression %:  "
            << 100.0 * (1.0 - static_cast<double>(total_comp) /
                                  static_cast<double>(total_raw))
            << "\n";

  std::cout << "\nPer-column FastLanes compression statistics:\n";
  std::cout << std::left << std::setw(16) << "Column"
            << std::right << std::setw(16) << "Original MiB"
            << std::setw(18) << "Compressed MiB"
            << std::setw(16) << "Reduction %"
            << std::setw(10) << "Bitwidth"
            << "\n";

  for (const ColumnData* c : cols) {
    const double reduction =
        100.0 * (1.0 - static_cast<double>(c->compressed_bytes) /
                          static_cast<double>(c->raw_bytes));

    std::cout << std::left << std::setw(16) << c->name
              << std::right << std::setw(16) << bytes_to_mib(c->raw_bytes)
              << std::setw(18) << bytes_to_mib(c->compressed_bytes)
              << std::setw(16) << reduction
              << std::setw(10) << c->meta.bitwidth
              << "\n";
  }

  const double total_reduction =
      100.0 * (1.0 - static_cast<double>(total_comp) /
                        static_cast<double>(total_raw));

  std::cout << std::left << std::setw(16) << "total"
            << std::right << std::setw(16) << bytes_to_mib(total_raw)
            << std::setw(18) << bytes_to_mib(total_comp)
            << std::setw(16) << total_reduction
            << std::setw(10) << "-"
            << "\n";
}

void print_results_table(const std::vector<SplitResult>& results,
                         double input_mib) {
  std::cout << "\nFastLanes SPJA CPU/GPU co-processing results\n";
  std::cout << "Measured time excludes compression and disk I/O.\n";
  std::cout << "GPU time includes compressed H2D, FastLanes decode, SPJA kernel, and D2H partial sums.\n";
  std::cout << "Lookup columns are resident on GPU during timed runs.\n\n";

  std::cout << std::left << std::setw(12) << "MODE"
            << std::right << std::setw(10) << "MiB"
            << std::setw(8) << "CPU%"
            << std::setw(8) << "GPU%"
            << std::setw(12) << "CPU_ms"
            << std::setw(12) << "GPU_ms"
            << std::setw(12) << "E2E_ms"
            << std::setw(13) << "EFF_GiB/s"
            << std::setw(10) << "MATCH"
            << "\n";

  std::cout << std::string(97, '-') << "\n";

  for (const auto& r : results) {
    std::cout << std::left << std::setw(12) << "TPCH_X40"
              << std::right << std::setw(10) << input_mib
              << std::setw(8) << r.cpu_percent
              << std::setw(8) << r.gpu_percent
              << std::setw(12) << r.cpu_ms
              << std::setw(12) << r.gpu_ms
              << std::setw(12) << r.total_ms
              << std::setw(13) << r.eff_gib_s
              << std::setw(10) << (r.valid ? "YES" : "NO")
              << "\n";
  }
}

void write_csv(const std::filesystem::path& csv_path,
               const std::vector<SplitResult>& results,
               size_t n_rows,
               size_t total_query_bytes) {
  std::filesystem::create_directories(csv_path.parent_path());

  std::ofstream out(csv_path);

  if (!out) {
    throw std::runtime_error("Could not write CSV: " + csv_path.string());
  }

  out << "mode,n_rows,input_mib,input_gib,cpu_percent,gpu_percent,"
      << "cpu_vecs,gpu_vecs,cpu_rows,gpu_rows,cpu_ms,gpu_ms,total_ms,"
      << "eff_gib_s,cpu_sum,gpu_sum,total_sum,reference_sum,match\n";

  for (const auto& r : results) {
    out << "TPCH_X40,"
        << n_rows << ","
        << bytes_to_mib(total_query_bytes) << ","
        << bytes_to_gib(total_query_bytes) << ","
        << r.cpu_percent << ","
        << r.gpu_percent << ","
        << r.cpu_vecs << ","
        << r.gpu_vecs << ","
        << r.cpu_rows << ","
        << r.gpu_rows << ","
        << r.cpu_ms << ","
        << r.gpu_ms << ","
        << r.total_ms << ","
        << r.eff_gib_s << ","
        << r.cpu_sum << ","
        << r.gpu_sum << ","
        << r.total_sum << ","
        << r.reference_sum << ","
        << (r.valid ? "YES" : "NO")
        << "\n";
  }
}

} // namespace

int main() {
  try {
    CUDA_CHECK(cudaSetDevice(0));

    std::cout << std::fixed << std::setprecision(3);

    const std::filesystem::path root =
        "results/fastlanes_lz4nvcomp_dowda/spja_x40_fastlanes";

    const std::filesystem::path csv_path =
        "results/fastlanes_lz4nvcomp_dowda/csv/fastlanes_spja_coproc_x40_results.csv";

    const int target_nation = 3;
    const int warmup = 1;
    const int runs = 5;

    const unsigned int detected_cpu_threads = std::thread::hardware_concurrency();
    const int cpu_worker_threads =
        (detected_cpu_threads > 0)
            ? static_cast<int>(std::min(36u, detected_cpu_threads))
            : 36;

    ColumnData orderkey_col =
        load_column("orderkey", root / "orderkey");

    ColumnData quantity_col =
        load_column("quantity", root / "quantity");

    ColumnData extendedprice_col =
        load_column("extendedprice", root / "extendedprice");

    if (orderkey_col.meta.nTup != quantity_col.meta.nTup ||
        orderkey_col.meta.nTup != extendedprice_col.meta.nTup) {
      throw std::runtime_error("FastLanes SPJA column row counts do not match.");
    }

    const size_t n_rows = orderkey_col.meta.nTup;
    const size_t n_vecs = (n_rows + kVecSize - 1) / kVecSize;
    const size_t total_query_bytes = n_rows * sizeof(int) * 3ULL;
    const double input_mib = bytes_to_mib(total_query_bytes);

    std::cout << "Loaded FastLanes SPJA x40 compressed data:\n";
    std::cout << "  Rows: " << n_rows << "\n";
    std::cout << "  FastLanes vectors: " << n_vecs << "\n";
    std::cout << "  Query input size MiB: " << input_mib << "\n";
    std::cout << "  Target nation: " << target_nation << "\n";
    std::cout << "  Warmup runs per split: " << warmup << "\n";
    std::cout << "  Timed runs per split: " << runs << "\n";
    std::cout << "  CPU worker threads: " << cpu_worker_threads << "\n";
    std::cout << "  Execution mode: CPU/GPU overlap using std::thread + CUDA stream\n";
    std::cout << "  Split unit: FastLanes vectors of " << kVecSize << " rows\n";

    print_compression_table(orderkey_col, quantity_col, extendedprice_col);

    std::vector<int> raw_orderkey =
        read_int_file(root / "orderkey" / "raw.dat");

    std::vector<int> raw_quantity =
        read_int_file(root / "quantity" / "raw.dat");

    std::vector<int> raw_extendedprice =
        read_int_file(root / "extendedprice" / "raw.dat");

    std::vector<int> order_custkey =
        read_int_file(root / "lookup" / "order_custkey.bin");

    std::vector<int> customer_nation =
        read_int_file(root / "lookup" / "customer_nation.bin");

    const int order_count = static_cast<int>(order_custkey.size());
    const int customer_count = static_cast<int>(customer_nation.size());

    std::cout << "\nComputing raw CPU reference result...\n";

    const unsigned long long reference_sum =
        spja_cpu_raw_reference(
            raw_orderkey.data(),
            raw_quantity.data(),
            raw_extendedprice.data(),
            order_custkey.data(),
            customer_nation.data(),
            n_rows,
            order_count,
            customer_count,
            target_nation);

    std::cout << "Reference result: " << reference_sum << "\n";

    raw_orderkey.clear();
    raw_orderkey.shrink_to_fit();
    raw_quantity.clear();
    raw_quantity.shrink_to_fit();
    raw_extendedprice.clear();
    raw_extendedprice.shrink_to_fit();

    int* d_order_custkey = nullptr;
    int* d_customer_nation = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_order_custkey),
                          order_custkey.size() * sizeof(int)));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_customer_nation),
                          customer_nation.size() * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_order_custkey,
                          order_custkey.data(),
                          order_custkey.size() * sizeof(int),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_customer_nation,
                          customer_nation.data(),
                          customer_nation.size() * sizeof(int),
                          cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    const std::vector<int> gpu_percents = {0, 25, 50, 75, 100};
    std::vector<SplitResult> results;

    for (int gpu_percent : gpu_percents) {
      SplitResult r =
          run_split(gpu_percent,
                    orderkey_col,
                    quantity_col,
                    extendedprice_col,
                    order_custkey.data(),
                    customer_nation.data(),
                    d_order_custkey,
                    d_customer_nation,
                    n_rows,
                    n_vecs,
                    order_count,
                    customer_count,
                    target_nation,
                    reference_sum,
                    warmup,
                    runs,
                    cpu_worker_threads,
                    total_query_bytes,
                    stream);

      results.push_back(r);
    }

    print_results_table(results, input_mib);

    bool all_valid = true;
    for (const auto& r : results) {
      all_valid = all_valid && r.valid;
    }

    std::cout << "\nOverall correctness: "
              << (all_valid ? "MATCH YES" : "MATCH NO")
              << "\n";

    write_csv(csv_path, results, n_rows, total_query_bytes);

    std::cout << "\nCSV written to:\n" << csv_path << "\n";

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_order_custkey));
    CUDA_CHECK(cudaFree(d_customer_nation));

    return all_valid ? 0 : 2;

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
// cd ~/gpu_benchmark_clean
// source ~/nvcomp_env/bin/activate
// bash scripts/run_fastlanes_spja_x40.sh