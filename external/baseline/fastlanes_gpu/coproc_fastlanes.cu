#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <boost/program_options.hpp>
#include <cub/warp/warp_reduce.cuh>

#include "fls_gen/rsum/rsum.cuh"
#include "fls_gen/unpack/unpack_fused.cuh"

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

constexpr int kBlockThreads = 32;
constexpr int kItemsPerThread = 32;
constexpr int kTileSize = kBlockThreads * kItemsPerThread;

static_assert(kTileSize == kVecSize,
              "This benchmark assumes FastLanes vector size is 1024 values.");

struct TimingStats {
  double total_ms = 0.0;
  double cpu_ms = 0.0;
  double gpu_ms = 0.0;
};

struct SplitResult {
  int cpu_percent = 0;
  int gpu_percent = 0;

  size_t cpu_vecs = 0;
  size_t gpu_vecs = 0;

  unsigned long long cpu_sum = 0;
  unsigned long long gpu_sum = 0;
  unsigned long long total_sum = 0;
  unsigned long long reference_sum = 0;

  double total_ms = 0.0;
  double cpu_ms = 0.0;
  double gpu_ms = 0.0;
  double throughput_gbs = 0.0;
  double throughput_gibs = 0.0;

  bool valid = false;
};

std::vector<unsigned char> read_binary_file(const std::filesystem::path &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Could not open file: " + path.string());
  }

  file.seekg(0, std::ios::end);
  const auto size = static_cast<size_t>(file.tellg());
  file.seekg(0, std::ios::beg);

  std::vector<unsigned char> buffer(size);
  if (size > 0) {
    file.read(reinterpret_cast<char *>(buffer.data()), size);
  }

  return buffer;
}

std::vector<uint32_t> read_u32_file(const std::filesystem::path &path) {
  auto bytes = read_binary_file(path);

  if (bytes.size() % sizeof(uint32_t) != 0) {
    throw std::runtime_error("File size is not divisible by uint32_t: " +
                             path.string());
  }

  std::vector<uint32_t> out(bytes.size() / sizeof(uint32_t));
  if (!out.empty()) {
    std::memcpy(out.data(), bytes.data(), bytes.size());
  }

  return out;
}

uint32_t extract_packed_value_cpu(const uint32_t *packed, int bitwidth,
                                  size_t index) {
  if (bitwidth == 0) {
    return 0;
  }

  if (bitwidth == 32) {
    return packed[index];
  }

  const size_t bit_pos = index * static_cast<size_t>(bitwidth);
  const size_t word_idx = bit_pos / 32;
  const int bit_offset = static_cast<int>(bit_pos % 32);

  uint64_t value = static_cast<uint64_t>(packed[word_idx]);

  if (bit_offset + bitwidth > 32) {
    value |= static_cast<uint64_t>(packed[word_idx + 1]) << 32;
  }

  const uint64_t mask = (1ULL << bitwidth) - 1ULL;
  return static_cast<uint32_t>((value >> bit_offset) & mask);
}

unsigned long long reference_sum_from_raw(const std::vector<uint32_t> &raw,
                                          size_t n_tup) {
  unsigned long long sum = 0;
  const size_t n = std::min(n_tup, raw.size());

  for (size_t i = 0; i < n; ++i) {
    sum += raw[i];
  }

  return sum;
}

uint32_t fastlanes_unpack_lane_value_cpu(const uint32_t *encoded_vec,
                                         int bitwidth,
                                         int lane,
                                         int item) {
  if (bitwidth == 0) {
    return 0;
  }

  if (bitwidth == 32) {
    return encoded_vec[static_cast<size_t>(item) * 32ULL + lane];
  }

  const size_t bit_pos = static_cast<size_t>(item) * static_cast<size_t>(bitwidth);
  const size_t word = bit_pos / 32ULL;
  const int shift = static_cast<int>(bit_pos % 32ULL);

  uint64_t packed =
      static_cast<uint64_t>(encoded_vec[word * 32ULL + static_cast<size_t>(lane)]);

  if (shift + bitwidth > 32) {
    packed |= static_cast<uint64_t>(
                  encoded_vec[(word + 1ULL) * 32ULL + static_cast<size_t>(lane)])
              << 32ULL;
  }

  const uint64_t mask = (1ULL << static_cast<unsigned>(bitwidth)) - 1ULL;
  return static_cast<uint32_t>((packed >> shift) & mask);
}

unsigned long long cpu_sum_unpack(const uint32_t *compressed, size_t start_vec,
                                  size_t num_vecs, size_t n_tup,
                                  int bitwidth) {
  unsigned long long sum = 0;

  for (size_t local_vec = 0; local_vec < num_vecs; ++local_vec) {
    const size_t vec = start_vec + local_vec;
    const uint32_t *encoded_vec =
        compressed + vec * static_cast<size_t>(bitwidth) * 32ULL;

    for (int item = 0; item < 32; ++item) {
      for (int lane = 0; lane < 32; ++lane) {
        const size_t local_index =
            static_cast<size_t>(item) * 32ULL + static_cast<size_t>(lane);
        const size_t row = vec * static_cast<size_t>(kVecSize) + local_index;

        if (row >= n_tup) {
          break;
        }

        sum += fastlanes_unpack_lane_value_cpu(encoded_vec, bitwidth, lane, item);
      }
    }
  }

  return sum;
}

unsigned long long cpu_sum_rsum(const uint32_t *compressed, size_t n_vec_total,
                                size_t start_vec, size_t num_vecs,
                                size_t n_tup, int bitwidth) {
  const uint32_t *base = compressed;
  const uint32_t *encoded = compressed + 32ULL * n_vec_total;

  unsigned long long sum = 0;

  for (size_t local_vec = 0; local_vec < num_vecs; ++local_vec) {
    const size_t vec = start_vec + local_vec;

    const uint32_t *base_vec = base + vec * 32ULL;
    const uint32_t *encoded_vec =
        encoded + vec * static_cast<size_t>(bitwidth) * 32ULL;

    for (int lane = 0; lane < 32; ++lane) {
      uint32_t running = base_vec[lane];

      for (int i = 0; i < 32; ++i) {
        const size_t local_index = static_cast<size_t>(i) * 32ULL + lane;
        const uint32_t delta =
            extract_packed_value_cpu(encoded_vec, bitwidth, local_index);

        running += delta;

        const size_t row =
            vec * static_cast<size_t>(kVecSize) + local_index;

        if (row < n_tup) {
          sum += running;
        }
      }
    }
  }

  return sum;
}

template <int BlockThreads, int ItemsPerThread>
__global__ void gpu_sum_unpack_range(const uint32_t *__restrict__ encoded,
                                     size_t start_vec, size_t num_vecs,
                                     size_t n_tup, int bitwidth,
                                     unsigned long long *__restrict__ out) {
  const size_t local_vec = blockIdx.x;

  if (local_vec >= num_vecs) {
    return;
  }

  const size_t vec = start_vec + local_vec;
  const size_t encoded_offset = vec * static_cast<size_t>(bitwidth) * 32ULL;

  uint32_t items[ItemsPerThread];

  unpack_device(encoded + encoded_offset, items, bitwidth);

  unsigned long long thread_sum = 0;

#pragma unroll
  for (int i = 0; i < ItemsPerThread; ++i) {
    const size_t local_index =
        static_cast<size_t>(i) * static_cast<size_t>(BlockThreads) +
        threadIdx.x;

    const size_t row = vec * static_cast<size_t>(kVecSize) + local_index;

    if (row < n_tup) {
      thread_sum += items[i];
    }
  }

  using WarpReduce = cub::WarpReduce<unsigned long long>;
  __shared__ typename WarpReduce::TempStorage temp_storage;

  const auto warp_sum = WarpReduce(temp_storage).Sum(thread_sum);
  __syncthreads();

  if (threadIdx.x == 0) {
    atomicAdd(out, warp_sum);
  }
}

template <int BlockThreads, int ItemsPerThread>
__global__ void gpu_sum_rsum_range(const uint32_t *__restrict__ base,
                                   const uint32_t *__restrict__ encoded,
                                   size_t start_vec, size_t num_vecs,
                                   size_t n_tup, int bitwidth,
                                   unsigned long long *__restrict__ out) {
  const size_t local_vec = blockIdx.x;

  if (local_vec >= num_vecs) {
    return;
  }

  const size_t vec = start_vec + local_vec;

  constexpr int TileSize = BlockThreads * ItemsPerThread;

  uint32_t items[ItemsPerThread];
  __shared__ uint32_t unpacked[TileSize];
  __shared__ uint32_t rsumed[TileSize];

  const size_t base_offset = vec * 32ULL;
  const size_t encoded_offset = vec * static_cast<size_t>(bitwidth) * 32ULL;

  unpack_device(encoded + encoded_offset, items, bitwidth);

#pragma unroll
  for (int i = 0; i < ItemsPerThread; ++i) {
    unpacked[i * ItemsPerThread + threadIdx.x] = items[i];
  }

  __syncthreads();

  d_rsum_32(unpacked, rsumed, base + base_offset);

  __syncthreads();

  unsigned long long thread_sum = 0;

#pragma unroll
  for (int i = 0; i < ItemsPerThread; ++i) {
    const size_t local_index =
        static_cast<size_t>(i) * static_cast<size_t>(BlockThreads) +
        threadIdx.x;

    const size_t row = vec * static_cast<size_t>(kVecSize) + local_index;

    if (row < n_tup) {
      thread_sum += rsumed[i * ItemsPerThread + threadIdx.x];
    }
  }

  using WarpReduce = cub::WarpReduce<unsigned long long>;
  __shared__ typename WarpReduce::TempStorage temp_storage;

  const auto warp_sum = WarpReduce(temp_storage).Sum(thread_sum);
  __syncthreads();

  if (threadIdx.x == 0) {
    atomicAdd(out, warp_sum);
  }
}

unsigned long long run_cpu_sum(const Metadata &metadata,
                               const uint32_t *compressed_host,
                               size_t n_vec_total, size_t start_vec,
                               size_t num_vecs) {
  if (num_vecs == 0) {
    return 0ULL;
  }

  switch (metadata.scheme) {
  case Metadata::unpack:
    return cpu_sum_unpack(compressed_host, start_vec, num_vecs, metadata.nTup,
                          metadata.bitwidth);

  case Metadata::rsum:
    return cpu_sum_rsum(compressed_host, n_vec_total, start_vec, num_vecs,
                        metadata.nTup, metadata.bitwidth);

  default:
    throw std::runtime_error("Unsupported FastLanes metadata scheme.");
  }
}

float run_gpu_sum(const Metadata &metadata, const uint32_t *compressed_device,
                  size_t n_vec_total, size_t start_vec, size_t num_vecs,
                  unsigned long long *gpu_out_device,
                  unsigned long long *gpu_out_host, cudaStream_t stream) {
  if (num_vecs == 0) {
    *gpu_out_host = 0ULL;
    return 0.0f;
  }

  CUDA_CHECK(cudaMemsetAsync(gpu_out_device, 0, sizeof(unsigned long long),
                             stream));

  cudaEvent_t start_event;
  cudaEvent_t stop_event;

  CUDA_CHECK(cudaEventCreate(&start_event));
  CUDA_CHECK(cudaEventCreate(&stop_event));

  CUDA_CHECK(cudaEventRecord(start_event, stream));

  switch (metadata.scheme) {
  case Metadata::unpack: {
    gpu_sum_unpack_range<kBlockThreads, kItemsPerThread>
        <<<static_cast<unsigned int>(num_vecs), kBlockThreads, 0, stream>>>(
            compressed_device, start_vec, num_vecs, metadata.nTup,
            metadata.bitwidth, gpu_out_device);
    break;
  }

  case Metadata::rsum: {
    const uint32_t *base = compressed_device;
    const uint32_t *encoded = compressed_device + 32ULL * n_vec_total;

    gpu_sum_rsum_range<kBlockThreads, kItemsPerThread>
        <<<static_cast<unsigned int>(num_vecs), kBlockThreads, 0, stream>>>(
            base, encoded, start_vec, num_vecs, metadata.nTup,
            metadata.bitwidth, gpu_out_device);
    break;
  }

  default:
    throw std::runtime_error("Unsupported FastLanes metadata scheme.");
  }

  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaEventRecord(stop_event, stream));

  CUDA_CHECK(cudaMemcpyAsync(gpu_out_host, gpu_out_device,
                             sizeof(unsigned long long),
                             cudaMemcpyDeviceToHost, stream));

  CUDA_CHECK(cudaStreamSynchronize(stream));

  float gpu_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start_event, stop_event));

  CUDA_CHECK(cudaEventDestroy(start_event));
  CUDA_CHECK(cudaEventDestroy(stop_event));

  return gpu_ms;
}

SplitResult run_one_split(const Metadata &metadata,
                          const std::vector<uint32_t> &compressed_host,
                          const uint32_t *compressed_device,
                          size_t n_vec_total, int cpu_percent,
                          unsigned long long reference_sum, int warmup_runs,
                          int timed_runs, cudaStream_t stream,
                          unsigned long long *gpu_out_device,
                          unsigned long long *gpu_out_host) {
  SplitResult result;
  result.cpu_percent = cpu_percent;
  result.gpu_percent = 100 - cpu_percent;

  result.cpu_vecs = (n_vec_total * static_cast<size_t>(cpu_percent)) / 100ULL;
  result.gpu_vecs = n_vec_total - result.cpu_vecs;

  const size_t cpu_start_vec = 0;
  const size_t gpu_start_vec = result.cpu_vecs;

  auto run_once = [&]() -> TimingStats {
    TimingStats t;

    unsigned long long cpu_sum = 0ULL;
    unsigned long long gpu_sum = 0ULL;

    const auto total_start = std::chrono::high_resolution_clock::now();

    std::thread cpu_thread;

    const auto cpu_start = std::chrono::high_resolution_clock::now();

    if (result.cpu_vecs > 0) {
      cpu_thread = std::thread([&]() {
        cpu_sum = run_cpu_sum(metadata, compressed_host.data(), n_vec_total,
                              cpu_start_vec, result.cpu_vecs);
      });
    }

    float gpu_ms = run_gpu_sum(metadata, compressed_device, n_vec_total,
                               gpu_start_vec, result.gpu_vecs, gpu_out_device,
                               gpu_out_host, stream);

    if (cpu_thread.joinable()) {
      cpu_thread.join();
    }

    const auto cpu_end = std::chrono::high_resolution_clock::now();
    const auto total_end = std::chrono::high_resolution_clock::now();

    gpu_sum = *gpu_out_host;

    t.cpu_ms =
        std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    t.gpu_ms = static_cast<double>(gpu_ms);
    t.total_ms =
        std::chrono::duration<double, std::milli>(total_end - total_start)
            .count();

    result.cpu_sum = cpu_sum;
    result.gpu_sum = gpu_sum;
    result.total_sum = cpu_sum + gpu_sum;

    return t;
  };

  for (int i = 0; i < warmup_runs; ++i) {
    (void)run_once();
  }

  double total_ms_sum = 0.0;
  double cpu_ms_sum = 0.0;
  double gpu_ms_sum = 0.0;

  for (int i = 0; i < timed_runs; ++i) {
    TimingStats t = run_once();

    total_ms_sum += t.total_ms;
    cpu_ms_sum += t.cpu_ms;
    gpu_ms_sum += t.gpu_ms;
  }

  result.total_ms = total_ms_sum / static_cast<double>(timed_runs);
  result.cpu_ms = cpu_ms_sum / static_cast<double>(timed_runs);
  result.gpu_ms = gpu_ms_sum / static_cast<double>(timed_runs);

  result.reference_sum = reference_sum;
  result.valid = (result.total_sum == reference_sum);

  const double input_gb =
      static_cast<double>(metadata.nTup) * sizeof(uint32_t) / 1e9;
  const double input_gib =
      static_cast<double>(metadata.nTup) * sizeof(uint32_t) /
      (1024.0 * 1024.0 * 1024.0);

  result.throughput_gbs = input_gb / (result.total_ms / 1000.0);
  result.throughput_gibs = input_gib / (result.total_ms / 1000.0);

  return result;
}

void print_result_table_header() {
  std::cout << "\n";
  std::cout << "FastLanes one-column CPU-GPU co-processing benchmark\n";
  std::cout << "Resident compressed data on GPU. H2D transfer is NOT included "
               "in this first version.\n";
  std::cout << "\n";

  std::cout << std::left << std::setw(14) << "Split" << std::right
            << std::setw(12) << "CPU vecs" << std::setw(12) << "GPU vecs"
            << std::setw(14) << "Total ms" << std::setw(14) << "CPU ms"
            << std::setw(14) << "GPU ms" << std::setw(16) << "GiB/s"
            << std::setw(18) << "Sum" << std::setw(10) << "Valid"
            << "\n";

  std::cout << std::string(126, '-') << "\n";
}

void print_result_row(const SplitResult &r) {
  const std::string split =
      std::to_string(r.cpu_percent) + "CPU/" + std::to_string(r.gpu_percent) +
      "GPU";

  std::cout << std::left << std::setw(14) << split << std::right
            << std::setw(12) << r.cpu_vecs << std::setw(12) << r.gpu_vecs
            << std::setw(14) << std::fixed << std::setprecision(3)
            << r.total_ms << std::setw(14) << r.cpu_ms << std::setw(14)
            << r.gpu_ms << std::setw(16) << r.throughput_gibs
            << std::setw(18) << r.total_sum << std::setw(10)
            << (r.valid ? "YES" : "NO") << "\n";
}

void write_csv(const std::filesystem::path &path,
               const std::vector<SplitResult> &results,
               size_t input_bytes, size_t compressed_bytes,
               const Metadata &metadata) {
  std::filesystem::create_directories(path.parent_path());

  std::ofstream file(path);
  if (!file) {
    throw std::runtime_error("Could not write CSV: " + path.string());
  }

  file << "split_label,cpu_percent,gpu_percent,cpu_vecs,gpu_vecs,"
       << "n_tuples,input_bytes,compressed_bytes,total_ms,cpu_ms,gpu_ms,"
       << "throughput_gbs,throughput_gibs,cpu_sum,gpu_sum,total_sum,"
       << "reference_sum,valid,scheme,bitwidth\n";

  for (const auto &r : results) {
    const std::string split =
        std::to_string(r.cpu_percent) + "CPU_" +
        std::to_string(r.gpu_percent) + "GPU";

    std::string scheme_name = "unknown";
    if (metadata.scheme == Metadata::unpack) {
      scheme_name = "unpack";
    } else if (metadata.scheme == Metadata::rsum) {
      scheme_name = "rsum";
    }

    file << split << "," << r.cpu_percent << "," << r.gpu_percent << ","
         << r.cpu_vecs << "," << r.gpu_vecs << "," << metadata.nTup << ","
         << input_bytes << "," << compressed_bytes << "," << r.total_ms << ","
         << r.cpu_ms << "," << r.gpu_ms << "," << r.throughput_gbs << ","
         << r.throughput_gibs << "," << r.cpu_sum << "," << r.gpu_sum << ","
         << r.total_sum << "," << r.reference_sum << ","
         << (r.valid ? "YES" : "NO") << "," << scheme_name << ","
         << metadata.bitwidth << "\n";
  }
}

} // namespace

int main(int argc, char **argv) {
  namespace po = boost::program_options;

  po::options_description desc("Options");
  desc.add_options()("help,h", "Print help message");
  desc.add_options()("input,i", po::value<std::string>()->required(),
                     "Input FastLanes-GPU directory containing metadata.txt "
                     "and data.dat");
  desc.add_options()("raw", po::value<std::string>(),
                     "Optional raw uint32 column file for reference sum. If not "
                     "given, the program tries ../raw.dat");
  desc.add_options()("output-benchmark", po::value<std::string>(),
                     "Optional output CSV file");
  desc.add_options()("warmup", po::value<int>()->default_value(1),
                     "Warmup runs per split");
  desc.add_options()("runs", po::value<int>()->default_value(5),
                     "Timed runs per split");

  po::variables_map vm;

  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help")) {
      std::cout << desc << "\n";
      return 0;
    }

    po::notify(vm);
  } catch (const std::exception &e) {
    std::cerr << "Argument error: " << e.what() << "\n";
    std::cerr << desc << "\n";
    return 1;
  }

  try {
    const std::filesystem::path input_dir = vm["input"].as<std::string>();
    const std::filesystem::path metadata_path = input_dir / "metadata.txt";
    const std::filesystem::path data_path = input_dir / "data.dat";

    const Metadata metadata = Metadata::read(metadata_path);

    const size_t n_vec = (metadata.nTup + kVecSize - 1) / kVecSize;
    const size_t input_bytes = metadata.nTup * sizeof(uint32_t);

    std::vector<uint32_t> compressed_host = read_u32_file(data_path);
    const size_t compressed_bytes = compressed_host.size() * sizeof(uint32_t);

    std::filesystem::path raw_path;

    if (vm.count("raw")) {
      raw_path = vm["raw"].as<std::string>();
    } else {
      raw_path = input_dir.parent_path() / "raw.dat";
    }

    unsigned long long reference_sum = 0ULL;
    bool have_raw_reference = false;

    if (std::filesystem::exists(raw_path)) {
      auto raw = read_u32_file(raw_path);
      reference_sum = reference_sum_from_raw(raw, metadata.nTup);
      have_raw_reference = true;
    } else {
      std::cerr << "Warning: raw reference file not found: " << raw_path
                << "\n";
      std::cerr << "The program will use 100% CPU FastLanes decode as "
                   "reference.\n";
    }

    std::cout << "\nInput directory: " << input_dir << "\n";
    std::cout << "Metadata tuples: " << metadata.nTup << "\n";
    std::cout << "FastLanes vectors: " << n_vec << "\n";
    std::cout << "Bitwidth: " << metadata.bitwidth << "\n";

    if (metadata.scheme == Metadata::unpack) {
      std::cout << "Scheme: unpack\n";
    } else if (metadata.scheme == Metadata::rsum) {
      std::cout << "Scheme: rsum\n";
    } else {
      std::cout << "Scheme: unknown\n";
    }

    std::cout << "Input MiB: "
              << static_cast<double>(input_bytes) / (1024.0 * 1024.0)
              << "\n";
    std::cout << "Compressed MiB: "
              << static_cast<double>(compressed_bytes) / (1024.0 * 1024.0)
              << "\n";

    if (input_bytes > 0) {
      const double reduction =
          100.0 *
          (1.0 - static_cast<double>(compressed_bytes) /
                     static_cast<double>(input_bytes));
      std::cout << "Compression reduction: " << reduction << "%\n";
    }

    uint32_t *compressed_device = nullptr;
    unsigned long long *gpu_out_device = nullptr;
    unsigned long long *gpu_out_host = nullptr;

    CUDA_CHECK(cudaMalloc(&compressed_device, compressed_bytes));
    CUDA_CHECK(cudaMemcpy(compressed_device, compressed_host.data(),
                          compressed_bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&gpu_out_device, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMallocHost(&gpu_out_host, sizeof(unsigned long long)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    const int warmup_runs = vm["warmup"].as<int>();
    const int timed_runs = vm["runs"].as<int>();

    if (!have_raw_reference) {
      reference_sum =
          run_cpu_sum(metadata, compressed_host.data(), n_vec, 0, n_vec);
    }

    std::cout << "Reference sum: " << reference_sum << "\n";

    std::vector<int> cpu_splits = {100, 75, 50, 25, 0};
    std::vector<SplitResult> results;

    print_result_table_header();

    for (int cpu_percent : cpu_splits) {
      SplitResult r =
          run_one_split(metadata, compressed_host, compressed_device, n_vec,
                        cpu_percent, reference_sum, warmup_runs, timed_runs,
                        stream, gpu_out_device, gpu_out_host);

      results.push_back(r);
      print_result_row(r);
    }

    bool all_valid = true;
    for (const auto &r : results) {
      all_valid = all_valid && r.valid;
    }

    std::cout << std::string(126, '-') << "\n";
    std::cout << "Overall correctness: " << (all_valid ? "MATCH YES" : "MATCH NO")
              << "\n";

    if (vm.count("output-benchmark")) {
      const std::filesystem::path csv_path =
          vm["output-benchmark"].as<std::string>();
      write_csv(csv_path, results, input_bytes, compressed_bytes, metadata);
      std::cout << "CSV written to: " << csv_path << "\n";
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFreeHost(gpu_out_host));
    CUDA_CHECK(cudaFree(gpu_out_device));
    CUDA_CHECK(cudaFree(compressed_device));

    return all_valid ? 0 : 2;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
