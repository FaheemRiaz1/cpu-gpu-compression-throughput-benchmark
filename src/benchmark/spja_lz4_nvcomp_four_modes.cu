#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>
#include <lz4.h>
#include <lz4hc.h>
#include <nvcomp/lz4.h>

#define CUDA_CHECK(call) do { \
    const cudaError_t e__ = (call); \
    if (e__ != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(e__) \
                  << " at " << __FILE__ << ':' << __LINE__ << '\n'; \
        std::exit(EXIT_FAILURE); \
    } \
} while (false)

#define NVCOMP_CHECK(call) do { \
    const nvcompStatus_t s__ = (call); \
    if (s__ != nvcompSuccess) { \
        std::cerr << "nvCOMP error: " << static_cast<int>(s__) \
                  << " at " << __FILE__ << ':' << __LINE__ << '\n'; \
        std::exit(EXIT_FAILURE); \
    } \
} while (false)

namespace {

const std::string OUTPUT_CSV =
    "results/spja_workload/csv/spja_lz4_nvcomp_four_mode_results.csv";

constexpr int BLOCK_SIZE = 256;
constexpr size_t MAX_GPU_BATCHES = 2;

enum class Mode {
    UncompressedWithH2D,
    CompressedWithH2D,
    CompressedWithoutH2D
};

const char* mode_name(Mode mode) {
    switch (mode) {
        case Mode::UncompressedWithH2D:
            return "UNCOMPRESSED_WITH_H2D";
        case Mode::CompressedWithH2D:
            return "COMPRESSED_WITH_H2D";
        case Mode::CompressedWithoutH2D:
            return "COMPRESSED_WITHOUT_H2D";
    }
    return "UNKNOWN";
}

template <typename A, typename B>
double ms_between(const A& start, const B& end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

double mean(const std::vector<double>& values) {
    if (values.empty()) return 0.0;
    return std::accumulate(values.begin(), values.end(), 0.0) /
           static_cast<double>(values.size());
}

double sample_stddev(const std::vector<double>& values) {
    if (values.size() < 2) return 0.0;
    const double avg = mean(values);
    double sum = 0.0;
    for (double value : values) {
        const double difference = value - avg;
        sum += difference * difference;
    }
    return std::sqrt(sum / static_cast<double>(values.size() - 1));
}

double bytes_to_mib(size_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

double bytes_to_gib(size_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
}

size_t file_size_bytes(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Could not open file: " + path);
    return static_cast<size_t>(file.tellg());
}

std::vector<int> read_int_column(const std::string& path) {
    const size_t bytes = file_size_bytes(path);
    if (bytes % sizeof(int) != 0) {
        throw std::runtime_error("Invalid int column size: " + path);
    }

    std::vector<int> values(bytes / sizeof(int));
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Could not open file: " + path);

    file.read(reinterpret_cast<char*>(values.data()),
              static_cast<std::streamsize>(bytes));
    if (static_cast<size_t>(file.gcount()) != bytes) {
        throw std::runtime_error("Could not read complete file: " + path);
    }
    return values;
}

unsigned long long spja_cpu_rows(
    const int* orderkey,
    const int* quantity,
    const int* extendedprice,
    const int* order_custkey,
    const int* customer_nation,
    size_t rows,
    int order_count,
    int customer_count,
    int target_nation
) {
    unsigned long long sum = 0ULL;
    for (size_t i = 0; i < rows; ++i) {
        const int ok = orderkey[i];
        if (ok <= 0 || ok >= order_count) continue;
        const int custkey = order_custkey[ok];
        if (custkey <= 0 || custkey >= customer_count) continue;
        if (quantity[i] > 25 && customer_nation[custkey] == target_nation) {
            sum += static_cast<unsigned long long>(extendedprice[i]);
        }
    }
    return sum;
}

__global__ void spja_gpu_kernel(
    const int* orderkey,
    const int* quantity,
    const int* extendedprice,
    const int* order_custkey,
    const int* customer_nation,
    size_t rows,
    int order_count,
    int customer_count,
    int target_nation,
    unsigned long long* block_sums
) {
    extern __shared__ unsigned long long shared_sum[];
    const unsigned int tid = threadIdx.x;
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned long long value = 0ULL;
    if (i < rows) {
        const int ok = orderkey[i];
        if (ok > 0 && ok < order_count) {
            const int custkey = order_custkey[ok];
            if (custkey > 0 && custkey < customer_count &&
                quantity[i] > 25 &&
                customer_nation[custkey] == target_nation) {
                value = static_cast<unsigned long long>(extendedprice[i]);
            }
        }
    }

    shared_sum[tid] = value;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared_sum[tid] += shared_sum[tid + stride];
        __syncthreads();
    }
    if (tid == 0) block_sums[blockIdx.x] = shared_sum[0];
}

__global__ void reduce_block_sums_kernel(
    const unsigned long long* block_sums,
    size_t n_blocks,
    unsigned long long* result
) {
    extern __shared__ unsigned long long shared_sum[];
    const unsigned int tid = threadIdx.x;
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    shared_sum[tid] = (i < n_blocks) ? block_sums[i] : 0ULL;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared_sum[tid] += shared_sum[tid + stride];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(result, shared_sum[0]);
}

struct CompressedColumn {
    std::vector<size_t> uncomp_sizes;
    std::vector<size_t> comp_sizes;
    std::vector<std::vector<char>> comp_chunks;
    size_t total_comp_bytes = 0;
};

CompressedColumn compress_column_lz4_hc(
    const int* data,
    size_t rows,
    size_t chunk_rows,
    int level
) {
    CompressedColumn output;
    const size_t chunks = (rows + chunk_rows - 1) / chunk_rows;
    output.uncomp_sizes.resize(chunks);
    output.comp_sizes.resize(chunks);
    output.comp_chunks.resize(chunks);

    const char* bytes = reinterpret_cast<const char*>(data);
    for (size_t chunk = 0; chunk < chunks; ++chunk) {
        const size_t first_row = chunk * chunk_rows;
        const size_t rows_here = std::min(chunk_rows, rows - first_row);
        const size_t uncompressed_bytes = rows_here * sizeof(int);
        const int bound = LZ4_compressBound(static_cast<int>(uncompressed_bytes));

        output.comp_chunks[chunk].resize(static_cast<size_t>(bound));
        const int compressed = LZ4_compress_HC(
            bytes + first_row * sizeof(int),
            output.comp_chunks[chunk].data(),
            static_cast<int>(uncompressed_bytes),
            bound,
            level
        );
        if (compressed <= 0) throw std::runtime_error("LZ4_HC failed.");

        output.comp_chunks[chunk].resize(static_cast<size_t>(compressed));
        output.uncomp_sizes[chunk] = uncompressed_bytes;
        output.comp_sizes[chunk] = static_cast<size_t>(compressed);
        output.total_comp_bytes += static_cast<size_t>(compressed);
    }
    return output;
}

size_t gcd_size_t(size_t a, size_t b) {
    while (b != 0) {
        const size_t remainder = a % b;
        a = b;
        b = remainder;
    }
    return a;
}

size_t choose_coprime_stride(size_t chunks, int trial) {
    const size_t candidates[] = {1, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
    const size_t count = sizeof(candidates) / sizeof(candidates[0]);
    for (size_t offset = 0; offset < count; ++offset) {
        const size_t candidate = candidates[(static_cast<size_t>(trial) + offset) % count];
        if (candidate < chunks && gcd_size_t(candidate, chunks) == 1) {
            return candidate;
        }
    }
    return 1;
}

void build_fair_assignment(
    size_t total_chunks,
    size_t gpu_chunks,
    int trial,
    std::vector<size_t>& cpu_ids,
    std::vector<size_t>& gpu_ids
) {
    cpu_ids.clear();
    gpu_ids.clear();
    if (total_chunks == 0) return;

    const size_t stride = choose_coprime_stride(total_chunks, trial);
    const size_t offset = (static_cast<size_t>(trial) * 17ULL) % total_chunks;

    for (size_t i = 0; i < total_chunks; ++i) {
        const size_t chunk = (offset + i * stride) % total_chunks;
        if (i < gpu_chunks) gpu_ids.push_back(chunk);
        else cpu_ids.push_back(chunk);
    }

    std::sort(cpu_ids.begin(), cpu_ids.end());
    std::sort(gpu_ids.begin(), gpu_ids.end());
}

struct GpuBatch {
    size_t rows = 0;
    size_t comp_bytes = 0;

    std::vector<int> raw_orderkey;
    std::vector<int> raw_quantity;
    std::vector<int> raw_price;

    std::vector<char> comp_flat;
    std::vector<size_t> comp_sizes;
    std::vector<size_t> uncomp_sizes;
    std::vector<size_t> comp_offsets;
    std::vector<size_t> row_offsets;
    std::vector<int> column_ids;
};

GpuBatch build_gpu_batch(
    const std::vector<size_t>& gpu_ids,
    size_t begin,
    size_t count,
    size_t total_rows,
    size_t chunk_rows,
    const std::vector<int>& raw_orderkey,
    const std::vector<int>& raw_quantity,
    const std::vector<int>& raw_price,
    const CompressedColumn& comp_orderkey,
    const CompressedColumn& comp_quantity,
    const CompressedColumn& comp_price
) {
    GpuBatch batch;
    size_t rows_total = 0;
    size_t comp_total = 0;

    const CompressedColumn* compressed_columns[3] = {
        &comp_orderkey, &comp_quantity, &comp_price
    };

    for (size_t local = 0; local < count; ++local) {
        const size_t chunk = gpu_ids[begin + local];
        const size_t first_row = chunk * chunk_rows;
        const size_t rows_here = std::min(chunk_rows, total_rows - first_row);

        for (int column = 0; column < 3; ++column) {
            const CompressedColumn& source = *compressed_columns[column];
            batch.comp_offsets.push_back(comp_total);
            batch.comp_sizes.push_back(source.comp_sizes[chunk]);
            batch.uncomp_sizes.push_back(rows_here * sizeof(int));
            batch.row_offsets.push_back(rows_total);
            batch.column_ids.push_back(column);
            comp_total += source.comp_sizes[chunk];
        }
        rows_total += rows_here;
    }

    batch.rows = rows_total;
    batch.comp_bytes = comp_total;
    batch.raw_orderkey.resize(rows_total);
    batch.raw_quantity.resize(rows_total);
    batch.raw_price.resize(rows_total);
    batch.comp_flat.resize(comp_total);

    size_t row_destination = 0;
    size_t comp_destination = 0;

    for (size_t local = 0; local < count; ++local) {
        const size_t chunk = gpu_ids[begin + local];
        const size_t first_row = chunk * chunk_rows;
        const size_t rows_here = std::min(chunk_rows, total_rows - first_row);

        std::memcpy(batch.raw_orderkey.data() + row_destination,
                    raw_orderkey.data() + first_row,
                    rows_here * sizeof(int));
        std::memcpy(batch.raw_quantity.data() + row_destination,
                    raw_quantity.data() + first_row,
                    rows_here * sizeof(int));
        std::memcpy(batch.raw_price.data() + row_destination,
                    raw_price.data() + first_row,
                    rows_here * sizeof(int));

        const CompressedColumn* columns[3] = {
            &comp_orderkey, &comp_quantity, &comp_price
        };
        for (int column = 0; column < 3; ++column) {
            const CompressedColumn& source = *columns[column];
            const size_t bytes = source.comp_sizes[chunk];
            std::memcpy(batch.comp_flat.data() + comp_destination,
                        source.comp_chunks[chunk].data(), bytes);
            comp_destination += bytes;
        }
        row_destination += rows_here;
    }

    return batch;
}

struct CpuResult {
    unsigned long long sum = 0ULL;
    double total_ms = 0.0;
};

CpuResult process_cpu_raw(
    const std::vector<size_t>& chunk_ids,
    const std::vector<int>& orderkey,
    const std::vector<int>& quantity,
    const std::vector<int>& price,
    const std::vector<int>& order_custkey,
    const std::vector<int>& customer_nation,
    size_t total_rows,
    size_t chunk_rows,
    int order_count,
    int customer_count,
    int target_nation,
    int requested_threads
) {
    const auto start = std::chrono::high_resolution_clock::now();
    if (chunk_ids.empty()) return CpuResult{};

    const int threads = std::max(1, std::min(
        requested_threads, static_cast<int>(chunk_ids.size())));
    const size_t chunks_per_thread =
        (chunk_ids.size() + static_cast<size_t>(threads) - 1) /
        static_cast<size_t>(threads);

    std::vector<unsigned long long> sums(static_cast<size_t>(threads), 0ULL);
    std::vector<std::exception_ptr> exceptions(static_cast<size_t>(threads));
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(threads));

    for (int thread = 0; thread < threads; ++thread) {
        workers.emplace_back([&, thread]() {
            try {
                const size_t begin = static_cast<size_t>(thread) * chunks_per_thread;
                const size_t end = std::min(begin + chunks_per_thread, chunk_ids.size());
                unsigned long long local_sum = 0ULL;

                for (size_t i = begin; i < end; ++i) {
                    const size_t chunk = chunk_ids[i];
                    const size_t first_row = chunk * chunk_rows;
                    const size_t rows_here = std::min(chunk_rows, total_rows - first_row);
                    local_sum += spja_cpu_rows(
                        orderkey.data() + first_row,
                        quantity.data() + first_row,
                        price.data() + first_row,
                        order_custkey.data(),
                        customer_nation.data(),
                        rows_here,
                        order_count,
                        customer_count,
                        target_nation
                    );
                }
                sums[static_cast<size_t>(thread)] = local_sum;
            } catch (...) {
                exceptions[static_cast<size_t>(thread)] = std::current_exception();
            }
        });
    }

    for (auto& worker : workers) worker.join();
    for (const auto& exception : exceptions) {
        if (exception) std::rethrow_exception(exception);
    }

    CpuResult result;
    result.sum = std::accumulate(sums.begin(), sums.end(), 0ULL);
    result.total_ms = ms_between(start, std::chrono::high_resolution_clock::now());
    return result;
}

CpuResult process_cpu_compressed(
    const std::vector<size_t>& chunk_ids,
    const CompressedColumn& orderkey,
    const CompressedColumn& quantity,
    const CompressedColumn& price,
    const std::vector<int>& order_custkey,
    const std::vector<int>& customer_nation,
    size_t total_rows,
    size_t chunk_rows,
    int order_count,
    int customer_count,
    int target_nation,
    int requested_threads
) {
    const auto start = std::chrono::high_resolution_clock::now();
    if (chunk_ids.empty()) return CpuResult{};

    const int threads = std::max(1, std::min(
        requested_threads, static_cast<int>(chunk_ids.size())));
    const size_t chunks_per_thread =
        (chunk_ids.size() + static_cast<size_t>(threads) - 1) /
        static_cast<size_t>(threads);

    std::vector<unsigned long long> sums(static_cast<size_t>(threads), 0ULL);
    std::vector<std::exception_ptr> exceptions(static_cast<size_t>(threads));
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(threads));

    for (int thread = 0; thread < threads; ++thread) {
        workers.emplace_back([&, thread]() {
            try {
                const size_t begin = static_cast<size_t>(thread) * chunks_per_thread;
                const size_t end = std::min(begin + chunks_per_thread, chunk_ids.size());
                std::vector<int> local_orderkey;
                std::vector<int> local_quantity;
                std::vector<int> local_price;
                unsigned long long local_sum = 0ULL;

                for (size_t i = begin; i < end; ++i) {
                    const size_t chunk = chunk_ids[i];
                    const size_t first_row = chunk * chunk_rows;
                    const size_t rows_here = std::min(chunk_rows, total_rows - first_row);
                    const int expected = static_cast<int>(rows_here * sizeof(int));

                    local_orderkey.resize(rows_here);
                    local_quantity.resize(rows_here);
                    local_price.resize(rows_here);

                    const int a = LZ4_decompress_safe(
                        orderkey.comp_chunks[chunk].data(),
                        reinterpret_cast<char*>(local_orderkey.data()),
                        static_cast<int>(orderkey.comp_sizes[chunk]), expected);
                    const int b = LZ4_decompress_safe(
                        quantity.comp_chunks[chunk].data(),
                        reinterpret_cast<char*>(local_quantity.data()),
                        static_cast<int>(quantity.comp_sizes[chunk]), expected);
                    const int c = LZ4_decompress_safe(
                        price.comp_chunks[chunk].data(),
                        reinterpret_cast<char*>(local_price.data()),
                        static_cast<int>(price.comp_sizes[chunk]), expected);

                    if (a != expected || b != expected || c != expected) {
                        throw std::runtime_error("CPU LZ4 decompression failed.");
                    }

                    local_sum += spja_cpu_rows(
                        local_orderkey.data(),
                        local_quantity.data(),
                        local_price.data(),
                        order_custkey.data(),
                        customer_nation.data(),
                        rows_here,
                        order_count,
                        customer_count,
                        target_nation
                    );
                }
                sums[static_cast<size_t>(thread)] = local_sum;
            } catch (...) {
                exceptions[static_cast<size_t>(thread)] = std::current_exception();
            }
        });
    }

    for (auto& worker : workers) worker.join();
    for (const auto& exception : exceptions) {
        if (exception) std::rethrow_exception(exception);
    }

    CpuResult result;
    result.sum = std::accumulate(sums.begin(), sums.end(), 0ULL);
    result.total_ms = ms_between(start, std::chrono::high_resolution_clock::now());
    return result;
}

struct DeviceBatchState {
    size_t count = 0;
    void** d_comp_ptrs = nullptr;
    void** d_decomp_ptrs = nullptr;
    size_t* d_comp_sizes = nullptr;
    size_t* d_uncomp_sizes = nullptr;
    size_t* d_actual_sizes = nullptr;
    nvcompStatus_t* d_statuses = nullptr;
};

struct DeviceBatch {
    cudaStream_t stream = nullptr;
    char* d_comp = nullptr;
    int* d_orderkey = nullptr;
    int* d_quantity = nullptr;
    int* d_price = nullptr;
    void* d_temp = nullptr;
    size_t temp_bytes = 0;
    unsigned long long* d_block_sums = nullptr;
    DeviceBatchState state;
};

struct GpuResources {
    std::vector<DeviceBatch> batches;
    int* d_order_custkey = nullptr;
    int* d_customer_nation = nullptr;
    unsigned long long* d_result = nullptr;
};

void destroy_device_state(DeviceBatchState& state) {
    if (state.d_comp_ptrs) CUDA_CHECK(cudaFree(state.d_comp_ptrs));
    if (state.d_decomp_ptrs) CUDA_CHECK(cudaFree(state.d_decomp_ptrs));
    if (state.d_comp_sizes) CUDA_CHECK(cudaFree(state.d_comp_sizes));
    if (state.d_uncomp_sizes) CUDA_CHECK(cudaFree(state.d_uncomp_sizes));
    if (state.d_actual_sizes) CUDA_CHECK(cudaFree(state.d_actual_sizes));
    if (state.d_statuses) CUDA_CHECK(cudaFree(state.d_statuses));
    state = DeviceBatchState{};
}

void destroy_gpu_resources(GpuResources& resources) {
    for (DeviceBatch& batch : resources.batches) {
        destroy_device_state(batch.state);
        if (batch.d_comp) CUDA_CHECK(cudaFree(batch.d_comp));
        if (batch.d_orderkey) CUDA_CHECK(cudaFree(batch.d_orderkey));
        if (batch.d_quantity) CUDA_CHECK(cudaFree(batch.d_quantity));
        if (batch.d_price) CUDA_CHECK(cudaFree(batch.d_price));
        if (batch.d_temp) CUDA_CHECK(cudaFree(batch.d_temp));
        if (batch.d_block_sums) CUDA_CHECK(cudaFree(batch.d_block_sums));
        if (batch.stream) CUDA_CHECK(cudaStreamDestroy(batch.stream));
    }
    if (resources.d_order_custkey) CUDA_CHECK(cudaFree(resources.d_order_custkey));
    if (resources.d_customer_nation) CUDA_CHECK(cudaFree(resources.d_customer_nation));
    if (resources.d_result) CUDA_CHECK(cudaFree(resources.d_result));
    resources = GpuResources{};
}

GpuResources create_gpu_resources(
    const std::vector<GpuBatch>& host_batches,
    const std::vector<int>& order_custkey,
    const std::vector<int>& customer_nation,
    size_t chunk_bytes
) {
    GpuResources resources;
    if (host_batches.empty()) return resources;
    if (host_batches.size() > MAX_GPU_BATCHES) {
        throw std::runtime_error(
            "More than two GPU batches were created. Increase gpu_batch_chunks.");
    }

    resources.batches.resize(host_batches.size());
    CUDA_CHECK(cudaMalloc(&resources.d_order_custkey,
                          order_custkey.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&resources.d_customer_nation,
                          customer_nation.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&resources.d_result, sizeof(unsigned long long)));

    CUDA_CHECK(cudaMemcpy(resources.d_order_custkey, order_custkey.data(),
                          order_custkey.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(resources.d_customer_nation, customer_nation.data(),
                          customer_nation.size() * sizeof(int), cudaMemcpyHostToDevice));

    const nvcompBatchedLZ4DecompressOpts_t opts =
        nvcompBatchedLZ4DecompressDefaultOpts;

    for (size_t i = 0; i < host_batches.size(); ++i) {
        const GpuBatch& host = host_batches[i];
        DeviceBatch& device = resources.batches[i];
        CUDA_CHECK(cudaStreamCreate(&device.stream));
        CUDA_CHECK(cudaMalloc(&device.d_comp, host.comp_bytes));
        CUDA_CHECK(cudaMalloc(&device.d_orderkey, host.rows * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&device.d_quantity, host.rows * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&device.d_price, host.rows * sizeof(int)));

        const size_t max_blocks = (host.rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
        CUDA_CHECK(cudaMalloc(&device.d_block_sums,
                              max_blocks * sizeof(unsigned long long)));

        DeviceBatchState& state = device.state;
        state.count = host.comp_sizes.size();
        CUDA_CHECK(cudaMalloc(&state.d_comp_ptrs, state.count * sizeof(void*)));
        CUDA_CHECK(cudaMalloc(&state.d_decomp_ptrs, state.count * sizeof(void*)));
        CUDA_CHECK(cudaMalloc(&state.d_comp_sizes, state.count * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&state.d_uncomp_sizes, state.count * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&state.d_actual_sizes, state.count * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&state.d_statuses, state.count * sizeof(nvcompStatus_t)));

        std::vector<void*> comp_ptrs(state.count);
        std::vector<void*> decomp_ptrs(state.count);
        for (size_t k = 0; k < state.count; ++k) {
            comp_ptrs[k] = device.d_comp + host.comp_offsets[k];
            const size_t row_offset = host.row_offsets[k];
            if (host.column_ids[k] == 0) {
                decomp_ptrs[k] = device.d_orderkey + row_offset;
            } else if (host.column_ids[k] == 1) {
                decomp_ptrs[k] = device.d_quantity + row_offset;
            } else {
                decomp_ptrs[k] = device.d_price + row_offset;
            }
        }

        CUDA_CHECK(cudaMemcpy(state.d_comp_ptrs, comp_ptrs.data(),
                              state.count * sizeof(void*), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(state.d_decomp_ptrs, decomp_ptrs.data(),
                              state.count * sizeof(void*), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(state.d_comp_sizes, host.comp_sizes.data(),
                              state.count * sizeof(size_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(state.d_uncomp_sizes, host.uncomp_sizes.data(),
                              state.count * sizeof(size_t), cudaMemcpyHostToDevice));

        size_t temp_bytes = 0;
        NVCOMP_CHECK(nvcompBatchedLZ4DecompressGetTempSizeSync(
            (const void* const* const)state.d_comp_ptrs,
            state.d_comp_sizes,
            state.count,
            chunk_bytes,
            &temp_bytes,
            host.rows * sizeof(int) * 3,
            opts,
            state.d_statuses,
            device.stream
        ));
        device.temp_bytes = std::max<size_t>(1, temp_bytes);
        CUDA_CHECK(cudaMalloc(&device.d_temp, device.temp_bytes));
    }

    return resources;
}

void preload_compressed(
    const std::vector<GpuBatch>& host_batches,
    GpuResources& resources
) {
    for (size_t i = 0; i < host_batches.size(); ++i) {
        CUDA_CHECK(cudaMemcpyAsync(
            resources.batches[i].d_comp,
            host_batches[i].comp_flat.data(),
            host_batches[i].comp_bytes,
            cudaMemcpyHostToDevice,
            resources.batches[i].stream
        ));
    }
    for (DeviceBatch& batch : resources.batches) {
        CUDA_CHECK(cudaStreamSynchronize(batch.stream));
    }
}

struct GpuResult {
    unsigned long long sum = 0ULL;
    double wall_ms = 0.0;
};

GpuResult run_gpu(
    Mode mode,
    const std::vector<GpuBatch>& host_batches,
    GpuResources& resources,
    int order_count,
    int customer_count,
    int target_nation
) {
    GpuResult result;
    if (host_batches.empty()) return result;

    cudaEvent_t start_event = nullptr;
    cudaEvent_t end_event = nullptr;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&end_event));
    CUDA_CHECK(cudaEventRecord(start_event, resources.batches[0].stream));

    for (size_t i = 1; i < resources.batches.size(); ++i) {
        CUDA_CHECK(cudaStreamWaitEvent(resources.batches[i].stream,
                                       start_event, 0));
    }

    const nvcompBatchedLZ4DecompressOpts_t opts =
        nvcompBatchedLZ4DecompressDefaultOpts;

    std::vector<cudaEvent_t> done(resources.batches.size(), nullptr);

    for (size_t i = 0; i < host_batches.size(); ++i) {
        const GpuBatch& host = host_batches[i];
        DeviceBatch& device = resources.batches[i];
        cudaStream_t stream = device.stream;

        if (mode == Mode::UncompressedWithH2D) {
            CUDA_CHECK(cudaMemcpyAsync(device.d_orderkey,
                                       host.raw_orderkey.data(),
                                       host.rows * sizeof(int),
                                       cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(device.d_quantity,
                                       host.raw_quantity.data(),
                                       host.rows * sizeof(int),
                                       cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(device.d_price,
                                       host.raw_price.data(),
                                       host.rows * sizeof(int),
                                       cudaMemcpyHostToDevice, stream));
        } else {
            if (mode == Mode::CompressedWithH2D) {
                CUDA_CHECK(cudaMemcpyAsync(device.d_comp,
                                           host.comp_flat.data(),
                                           host.comp_bytes,
                                           cudaMemcpyHostToDevice, stream));
            }

            DeviceBatchState& state = device.state;
            NVCOMP_CHECK(nvcompBatchedLZ4DecompressAsync(
                (const void* const*)state.d_comp_ptrs,
                state.d_comp_sizes,
                state.d_uncomp_sizes,
                state.d_actual_sizes,
                state.count,
                device.d_temp,
                device.temp_bytes,
                state.d_decomp_ptrs,
                opts,
                state.d_statuses,
                stream
            ));
        }

        const int blocks = static_cast<int>(
            (host.rows + BLOCK_SIZE - 1) / BLOCK_SIZE);
        spja_gpu_kernel<<<blocks, BLOCK_SIZE,
                          BLOCK_SIZE * sizeof(unsigned long long), stream>>>(
            device.d_orderkey,
            device.d_quantity,
            device.d_price,
            resources.d_order_custkey,
            resources.d_customer_nation,
            host.rows,
            order_count,
            customer_count,
            target_nation,
            device.d_block_sums
        );
        CUDA_CHECK(cudaGetLastError());

        const int reduce_blocks =
            (blocks + BLOCK_SIZE - 1) / BLOCK_SIZE;
        reduce_block_sums_kernel<<<reduce_blocks, BLOCK_SIZE,
                                   BLOCK_SIZE * sizeof(unsigned long long), stream>>>(
            device.d_block_sums,
            static_cast<size_t>(blocks),
            resources.d_result
        );
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaEventCreate(&done[i]));
        CUDA_CHECK(cudaEventRecord(done[i], stream));
    }

    for (size_t i = 0; i < done.size(); ++i) {
        CUDA_CHECK(cudaStreamWaitEvent(resources.batches[0].stream,
                                       done[i], 0));
    }

    CUDA_CHECK(cudaMemcpyAsync(
        &result.sum,
        resources.d_result,
        sizeof(unsigned long long),
        cudaMemcpyDeviceToHost,
        resources.batches[0].stream
    ));
    CUDA_CHECK(cudaEventRecord(end_event, resources.batches[0].stream));
    CUDA_CHECK(cudaStreamSynchronize(resources.batches[0].stream));

    float elapsed = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, start_event, end_event));
    result.wall_ms = static_cast<double>(elapsed);

    for (cudaEvent_t event : done) CUDA_CHECK(cudaEventDestroy(event));
    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(end_event));
    return result;
}

struct RunResult {
    unsigned long long final_sum = 0ULL;
    double cpu_ms = 0.0;
    double gpu_ms = 0.0;
    double e2e_ms = 0.0;
};

RunResult run_mode(
    Mode mode,
    const std::vector<size_t>& cpu_ids,
    const std::vector<GpuBatch>& gpu_batches,
    GpuResources& gpu_resources,
    const std::vector<int>& raw_orderkey,
    const std::vector<int>& raw_quantity,
    const std::vector<int>& raw_price,
    const CompressedColumn& comp_orderkey,
    const CompressedColumn& comp_quantity,
    const CompressedColumn& comp_price,
    const std::vector<int>& order_custkey,
    const std::vector<int>& customer_nation,
    size_t total_rows,
    size_t chunk_rows,
    int order_count,
    int customer_count,
    int target_nation,
    int cpu_threads
) {
    if (mode == Mode::CompressedWithoutH2D && !gpu_batches.empty()) {
        preload_compressed(gpu_batches, gpu_resources);
    }

    if (!gpu_batches.empty()) {
        CUDA_CHECK(cudaMemset(
            gpu_resources.d_result, 0, sizeof(unsigned long long)));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    RunResult output;
    CpuResult cpu_result;
    GpuResult gpu_result;
    std::exception_ptr cpu_exception = nullptr;

    const auto e2e_start = std::chrono::high_resolution_clock::now();

    std::thread cpu_thread([&]() {
        try {
            if (mode == Mode::UncompressedWithH2D) {
                cpu_result = process_cpu_raw(
                    cpu_ids,
                    raw_orderkey,
                    raw_quantity,
                    raw_price,
                    order_custkey,
                    customer_nation,
                    total_rows,
                    chunk_rows,
                    order_count,
                    customer_count,
                    target_nation,
                    cpu_threads
                );
            } else {
                cpu_result = process_cpu_compressed(
                    cpu_ids,
                    comp_orderkey,
                    comp_quantity,
                    comp_price,
                    order_custkey,
                    customer_nation,
                    total_rows,
                    chunk_rows,
                    order_count,
                    customer_count,
                    target_nation,
                    cpu_threads
                );
            }
        } catch (...) {
            cpu_exception = std::current_exception();
        }
    });

    gpu_result = run_gpu(
        mode,
        gpu_batches,
        gpu_resources,
        order_count,
        customer_count,
        target_nation
    );

    cpu_thread.join();
    if (cpu_exception) std::rethrow_exception(cpu_exception);

    output.cpu_ms = cpu_result.total_ms;
    output.gpu_ms = gpu_result.wall_ms;
    output.final_sum = cpu_result.sum + gpu_result.sum;
    output.e2e_ms = ms_between(
        e2e_start, std::chrono::high_resolution_clock::now());
    return output;
}

struct ModeStats {
    std::vector<double> e2e_ms;
    std::vector<double> throughput_gibps;
    std::vector<double> cpu_ms;
    std::vector<double> gpu_ms;
    unsigned long long last_result = 0ULL;
    bool valid = true;
};

struct SplitStats {
    int cpu_percent = 0;
    int gpu_percent = 0;
    double cpu_rows_avg = 0.0;
    double gpu_rows_avg = 0.0;
    double compressed_gpu_bytes_avg = 0.0;

    // Experiment 1: uncompressed vs compressed; H2D is timed in both.
    ModeStats compression_uncompressed;
    ModeStats compression_compressed;

    // Experiment 2: H2D vs no H2D; LZ4/nvCOMP is used in both.
    ModeStats h2d_with;
    ModeStats h2d_without;
};

static void record_run(
    ModeStats& stats,
    const RunResult& run,
    double logical_input_gib,
    unsigned long long reference
) {
    if (run.e2e_ms <= 0.0) {
        throw std::runtime_error("Measured E2E runtime is not positive.");
    }

    stats.e2e_ms.push_back(run.e2e_ms);
    stats.throughput_gibps.push_back(
        logical_input_gib / (run.e2e_ms / 1000.0)
    );
    stats.cpu_ms.push_back(run.cpu_ms);
    stats.gpu_ms.push_back(run.gpu_ms);
    stats.last_result = run.final_sum;
    stats.valid = stats.valid && (run.final_sum == reference);
}

static void write_result_row(
    std::ofstream& csv,
    const char* experiment,
    const char* mode,
    double input_mib,
    const SplitStats& split,
    int assignment_trials,
    int timed_runs,
    double timed_h2d_bytes,
    double offline_compression_ms,
    const ModeStats& stats,
    unsigned long long reference
) {
    csv << experiment << ','
        << mode << ','
        << input_mib << ','
        << split.cpu_percent << ','
        << split.gpu_percent << ','
        << assignment_trials << ','
        << timed_runs << ','
        << split.cpu_rows_avg << ','
        << split.gpu_rows_avg << ','
        << timed_h2d_bytes << ','
        << offline_compression_ms << ','
        << mean(stats.e2e_ms) << ','
        << sample_stddev(stats.e2e_ms) << ','
        << mean(stats.throughput_gibps) << ','
        << sample_stddev(stats.throughput_gibps) << ','
        << mean(stats.cpu_ms) << ','
        << sample_stddev(stats.cpu_ms) << ','
        << mean(stats.gpu_ms) << ','
        << sample_stddev(stats.gpu_ms) << ','
        << stats.last_result << ','
        << reference << ','
        << (stats.valid ? "YES" : "NO") << '\n';
}

}  // namespace

int main() {
    try {
        CUDA_CHECK(cudaSetDevice(0));
        std::system("mkdir -p results/spja_workload/csv");

        const std::string orderkey_path =
            "data/tpch_columnar/orderkey_sfx40.bin";
        const std::string quantity_path =
            "data/tpch_columnar/quantity_sfx40.bin";
        const std::string price_path =
            "data/tpch_columnar/extendedprice_sfx40.bin";
        const std::string order_custkey_path =
            "data/tpch_columnar/order_custkey_sfx40.bin";
        const std::string customer_nation_path =
            "data/tpch_columnar/customer_nation_sfx40.bin";

        const std::string compression_csv_path =
            "results/spja_workload/csv/"
            "spja_compressed_vs_uncompressed_results.csv";

        const std::string h2d_csv_path =
            "results/spja_workload/csv/"
            "spja_h2d_vs_no_h2d_results.csv";

        const size_t chunk_bytes = 1ULL << 20;  // 1 MiB
        const size_t chunk_rows = chunk_bytes / sizeof(int);
        const size_t gpu_batch_chunks = 768;
        const int lz4_hc_level = 8;
        const int warmups = 5;
        const int timed_runs = 5;
        const int assignment_trials = 5;
        const int target_nation = 3;

        const unsigned int detected_threads =
            std::thread::hardware_concurrency();
        const int cpu_threads = detected_threads > 0
            ? static_cast<int>(std::min(36u, detected_threads))
            : 36;

        std::vector<int> orderkey = read_int_column(orderkey_path);
        std::vector<int> quantity = read_int_column(quantity_path);
        std::vector<int> price = read_int_column(price_path);
        std::vector<int> order_custkey = read_int_column(order_custkey_path);
        std::vector<int> customer_nation = read_int_column(customer_nation_path);

        if (orderkey.size() != quantity.size() ||
            orderkey.size() != price.size()) {
            throw std::runtime_error("Fact-column sizes do not match.");
        }

        const size_t rows = orderkey.size();
        const int order_count = static_cast<int>(order_custkey.size());
        const int customer_count = static_cast<int>(customer_nation.size());
        const size_t total_chunks =
            (rows + chunk_rows - 1) / chunk_rows;
        const size_t logical_bytes = rows * sizeof(int) * 3;
        const double logical_gib = bytes_to_gib(logical_bytes);
        const double input_mib = bytes_to_mib(logical_bytes);

        const unsigned long long reference = spja_cpu_rows(
            orderkey.data(),
            quantity.data(),
            price.data(),
            order_custkey.data(),
            customer_nation.data(),
            rows,
            order_count,
            customer_count,
            target_nation
        );

        const auto compression_start =
            std::chrono::high_resolution_clock::now();

        CompressedColumn comp_orderkey = compress_column_lz4_hc(
            orderkey.data(), rows, chunk_rows, lz4_hc_level
        );
        CompressedColumn comp_quantity = compress_column_lz4_hc(
            quantity.data(), rows, chunk_rows, lz4_hc_level
        );
        CompressedColumn comp_price = compress_column_lz4_hc(
            price.data(), rows, chunk_rows, lz4_hc_level
        );

        const double offline_compression_ms = ms_between(
            compression_start,
            std::chrono::high_resolution_clock::now()
        );

        const size_t compressed_bytes =
            comp_orderkey.total_comp_bytes +
            comp_quantity.total_comp_bytes +
            comp_price.total_comp_bytes;

        std::cout << std::fixed << std::setprecision(3)
                  << "Rows: " << rows << '\n'
                  << "Logical input MiB: " << input_mib << '\n'
                  << "Compressed input MiB: "
                  << bytes_to_mib(compressed_bytes) << '\n'
                  << "Offline compression ms (excluded from all four graphs): "
                  << offline_compression_ms << '\n'
                  << "Reference result: " << reference << "\n\n";

        const std::vector<int> gpu_percents = {0, 25, 50, 75, 100};
        std::vector<SplitStats> all_splits;

        for (int gpu_percent : gpu_percents) {
            SplitStats split;
            split.gpu_percent = gpu_percent;
            split.cpu_percent = 100 - gpu_percent;

            const size_t gpu_chunk_count =
                (total_chunks * static_cast<size_t>(gpu_percent)) / 100;

            for (int trial = 0; trial < assignment_trials; ++trial) {
                std::vector<size_t> cpu_ids;
                std::vector<size_t> gpu_ids;

                build_fair_assignment(
                    total_chunks,
                    gpu_chunk_count,
                    trial,
                    cpu_ids,
                    gpu_ids
                );

                size_t cpu_rows = 0;
                for (size_t chunk : cpu_ids) {
                    const size_t first = chunk * chunk_rows;
                    cpu_rows += std::min(chunk_rows, rows - first);
                }
                const size_t gpu_rows = rows - cpu_rows;

                split.cpu_rows_avg += static_cast<double>(cpu_rows);
                split.gpu_rows_avg += static_cast<double>(gpu_rows);

                std::vector<GpuBatch> gpu_batches;
                for (size_t begin = 0;
                     begin < gpu_ids.size();
                     begin += gpu_batch_chunks) {
                    const size_t count = std::min(
                        gpu_batch_chunks,
                        gpu_ids.size() - begin
                    );

                    gpu_batches.push_back(build_gpu_batch(
                        gpu_ids,
                        begin,
                        count,
                        rows,
                        chunk_rows,
                        orderkey,
                        quantity,
                        price,
                        comp_orderkey,
                        comp_quantity,
                        comp_price
                    ));
                }

                if (gpu_batches.size() > MAX_GPU_BATCHES) {
                    throw std::runtime_error(
                        "More than two GPU batches were created. "
                        "Increase gpu_batch_chunks."
                    );
                }

                size_t split_compressed_gpu_bytes = 0;
                for (const GpuBatch& batch : gpu_batches) {
                    split_compressed_gpu_bytes += batch.comp_bytes;
                }
                split.compressed_gpu_bytes_avg +=
                    static_cast<double>(split_compressed_gpu_bytes);

                std::vector<void*> registered_buffers;
                bool all_registered = true;

                for (GpuBatch& batch : gpu_batches) {
                    void* pointers[] = {
                        batch.comp_flat.data(),
                        batch.raw_orderkey.data(),
                        batch.raw_quantity.data(),
                        batch.raw_price.data()
                    };
                    size_t sizes[] = {
                        batch.comp_bytes,
                        batch.rows * sizeof(int),
                        batch.rows * sizeof(int),
                        batch.rows * sizeof(int)
                    };

                    for (int index = 0; index < 4; ++index) {
                        if (sizes[index] == 0) {
                            continue;
                        }

                        const cudaError_t status = cudaHostRegister(
                            pointers[index],
                            sizes[index],
                            cudaHostRegisterDefault
                        );

                        if (status != cudaSuccess) {
                            cudaGetLastError();
                            all_registered = false;
                            break;
                        }

                        registered_buffers.push_back(pointers[index]);
                    }

                    if (!all_registered) {
                        break;
                    }
                }

                if (!all_registered) {
                    for (void* pointer : registered_buffers) {
                        CUDA_CHECK(cudaHostUnregister(pointer));
                    }
                    registered_buffers.clear();
                    std::cerr
                        << "Warning: host registration failed; "
                        << "both comparisons use pageable host memory.\n";
                }

                GpuResources resources = create_gpu_resources(
                    gpu_batches,
                    order_custkey,
                    customer_nation,
                    chunk_bytes
                );

                auto execute_pair = [&]
                (
                    Mode first_mode,
                    Mode second_mode,
                    ModeStats& first_stats,
                    ModeStats& second_stats
                ) {
                    const int total_iterations = warmups + timed_runs;

                    for (int iteration = 0;
                         iteration < total_iterations;
                         ++iteration) {
                        const bool reverse_order = (iteration % 2) != 0;
                        const Mode order[2] = {
                            reverse_order ? second_mode : first_mode,
                            reverse_order ? first_mode : second_mode
                        };

                        for (Mode mode : order) {
                            RunResult run = run_mode(
                                mode,
                                cpu_ids,
                                gpu_batches,
                                resources,
                                orderkey,
                                quantity,
                                price,
                                comp_orderkey,
                                comp_quantity,
                                comp_price,
                                order_custkey,
                                customer_nation,
                                rows,
                                chunk_rows,
                                order_count,
                                customer_count,
                                target_nation,
                                cpu_threads
                            );

                            if (run.final_sum != reference) {
                                throw std::runtime_error(
                                    std::string("Correctness mismatch in mode ") +
                                    mode_name(mode)
                                );
                            }

                            if (iteration >= warmups) {
                                ModeStats& destination =
                                    (mode == first_mode)
                                        ? first_stats
                                        : second_stats;

                                record_run(
                                    destination,
                                    run,
                                    logical_gib,
                                    reference
                                );
                            }
                        }
                    }
                };

                // Experiment 1 is measured independently.
                // Both modes include H2D; only representation changes.
                execute_pair(
                    Mode::UncompressedWithH2D,
                    Mode::CompressedWithH2D,
                    split.compression_uncompressed,
                    split.compression_compressed
                );

                // Experiment 2 is measured independently.
                // Both modes are compressed; only timed H2D changes.
                execute_pair(
                    Mode::CompressedWithH2D,
                    Mode::CompressedWithoutH2D,
                    split.h2d_with,
                    split.h2d_without
                );

                destroy_gpu_resources(resources);

                for (void* pointer : registered_buffers) {
                    CUDA_CHECK(cudaHostUnregister(pointer));
                }
            }

            split.cpu_rows_avg /= static_cast<double>(assignment_trials);
            split.gpu_rows_avg /= static_cast<double>(assignment_trials);
            split.compressed_gpu_bytes_avg /=
                static_cast<double>(assignment_trials);

            std::cout << split.cpu_percent << '/' << split.gpu_percent << '\n'
                      << "  Compression experiment:\n"
                      << "    UNCOMPRESSED: E2E="
                      << mean(split.compression_uncompressed.e2e_ms)
                      << " ms, TP="
                      << mean(split.compression_uncompressed.throughput_gibps)
                      << " GiB/s\n"
                      << "    COMPRESSED:   E2E="
                      << mean(split.compression_compressed.e2e_ms)
                      << " ms, TP="
                      << mean(split.compression_compressed.throughput_gibps)
                      << " GiB/s\n"
                      << "  H2D experiment (compressed pipeline only):\n"
                      << "    WITH_H2D:     E2E="
                      << mean(split.h2d_with.e2e_ms)
                      << " ms, TP="
                      << mean(split.h2d_with.throughput_gibps)
                      << " GiB/s\n"
                      << "    WITHOUT_H2D:  E2E="
                      << mean(split.h2d_without.e2e_ms)
                      << " ms, TP="
                      << mean(split.h2d_without.throughput_gibps)
                      << " GiB/s\n";

            all_splits.push_back(std::move(split));
        }

        const char* csv_header =
            "Experiment,Mode,Input_MiB,CPU_Percent,GPU_Percent,"
            "Assignment_Trials,Timed_Runs_Per_Assignment,"
            "CPU_Rows_Avg,GPU_Rows_Avg,Timed_H2D_Bytes_Avg,"
            "Offline_Compression_ms_Excluded,"
            "E2E_ms_Avg,E2E_ms_StdDev,"
            "Effective_GiBps_Avg,Effective_GiBps_StdDev,"
            "CPU_Path_ms_Avg,CPU_Path_ms_StdDev,"
            "GPU_Path_ms_Avg,GPU_Path_ms_StdDev,"
            "Final_Result,Reference_Result,Valid\n";

        std::ofstream compression_csv(compression_csv_path);
        if (!compression_csv) {
            throw std::runtime_error(
                "Could not create compression comparison CSV."
            );
        }
        compression_csv << csv_header;

        std::ofstream h2d_csv(h2d_csv_path);
        if (!h2d_csv) {
            throw std::runtime_error(
                "Could not create H2D comparison CSV."
            );
        }
        h2d_csv << csv_header;

        for (const SplitStats& split : all_splits) {
            const double raw_h2d_bytes =
                split.gpu_rows_avg * 3.0 * sizeof(int);
            const double compressed_h2d_bytes =
                split.compressed_gpu_bytes_avg;

            write_result_row(
                compression_csv,
                "COMPRESSION",
                "UNCOMPRESSED",
                input_mib,
                split,
                assignment_trials,
                timed_runs,
                raw_h2d_bytes,
                offline_compression_ms,
                split.compression_uncompressed,
                reference
            );

            write_result_row(
                compression_csv,
                "COMPRESSION",
                "COMPRESSED_LZ4_NVCOMP",
                input_mib,
                split,
                assignment_trials,
                timed_runs,
                compressed_h2d_bytes,
                offline_compression_ms,
                split.compression_compressed,
                reference
            );

            write_result_row(
                h2d_csv,
                "H2D",
                "WITH_H2D",
                input_mib,
                split,
                assignment_trials,
                timed_runs,
                compressed_h2d_bytes,
                offline_compression_ms,
                split.h2d_with,
                reference
            );

            write_result_row(
                h2d_csv,
                "H2D",
                "WITHOUT_H2D",
                input_mib,
                split,
                assignment_trials,
                timed_runs,
                0.0,
                offline_compression_ms,
                split.h2d_without,
                reference
            );
        }

        compression_csv.close();
        h2d_csv.close();

        std::cout << "\nWrote:\n"
                  << "  " << compression_csv_path << '\n'
                  << "  " << h2d_csv_path << '\n';

        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "Exception: " << error.what() << '\n';
        return 1;
    }
}

// nvcc -std=c++17 -O3 \
//   -I ~/nvcomp_env/lib/python3.12/site-packages/nvidia/libnvcomp/include \
//   src/benchmark/spja_lz4_nvcomp_four_modes.cu \
//   -L ~/nvcomp_env/lib/python3.12/site-packages/nvidia/libnvcomp/lib64 \
//   -lnvcomp \
//   -llz4 \
//   -o bin/spja_lz4_nvcomp_four_modes


// LD_LIBRARY_PATH=~/nvcomp_env/lib/python3.12/site-packages/nvidia/libnvcomp/lib64:$LD_LIBRARY_PATH \
// ./bin/spja_lz4_nvcomp_four_modes