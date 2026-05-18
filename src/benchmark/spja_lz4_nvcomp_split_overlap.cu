#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <thread>
#include <exception>

#include <cuda_runtime.h>
#include <lz4.h>
#include <lz4hc.h>
#include <nvcomp/lz4.h>

// ------------------------------------------------------------
// Output colors
// ------------------------------------------------------------
const std::string RED   = "\033[31m";
const std::string GREEN = "\033[32m";
const std::string BLUE  = "\033[34m";
const std::string RESET = "\033[0m";

// ------------------------------------------------------------
// Output files
// Main CSV filename is intentionally kept unchanged.
// ------------------------------------------------------------
const std::string MAIN_CSV_PATH =
    "results/spja_workload/csv/spja_lz4_nvcomp_hybrid_split_overlap_results.csv";

const std::string DETAILED_CSV_PATH =
    "results/spja_workload/csv/spja_lz4_nvcomp_hybrid_split_overlap_detailed_trials.csv";

const std::string COMPRESSION_CSV_PATH =
    "results/spja_workload/csv/spja_lz4_nvcomp_compression_stats.csv";

const std::string METADATA_PATH =
    "results/spja_workload/csv/spja_lz4_nvcomp_benchmark_metadata.txt";

const std::string SUMMARY_PATH =
    "results/spja_workload/csv/spja_lz4_nvcomp_summary.txt";

// ------------------------------------------------------------
// Basic error checking helpers
// ------------------------------------------------------------
inline void cuda_check(cudaError_t e, const char* file, int line) {
    if (e != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(e)
                  << " at " << file << ":" << line << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

inline void nvcomp_check(nvcompStatus_t s, const char* file, int line) {
    if (s != nvcompSuccess) {
        std::cerr << "nvCOMP error: status=" << static_cast<int>(s)
                  << " at " << file << ":" << line << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(x) cuda_check((x), __FILE__, __LINE__)
#define NVCOMP_CHECK(x) nvcomp_check((x), __FILE__, __LINE__)

// ------------------------------------------------------------
// Small utility functions
// ------------------------------------------------------------
template <typename T1, typename T2>
double ms_between(const T1& a, const T2& b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

static double mean(const std::vector<double>& values) {
    if (values.empty()) return 0.0;

    double sum = 0.0;
    for (double v : values) {
        sum += v;
    }

    return sum / static_cast<double>(values.size());
}

static double stddev_sample(const std::vector<double>& values) {
    if (values.size() < 2) return 0.0;

    const double avg = mean(values);
    double var = 0.0;

    for (double v : values) {
        const double diff = v - avg;
        var += diff * diff;
    }

    var /= static_cast<double>(values.size() - 1);
    return std::sqrt(var);
}

static double bytes_to_mib(size_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

static double bytes_to_gib(size_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
}

static double compression_reduction(size_t original_bytes, size_t compressed_bytes) {
    if (original_bytes == 0) return 0.0;

    return std::max(
        0.0,
        (1.0 -
         static_cast<double>(compressed_bytes) /
         static_cast<double>(original_bytes)) * 100.0
    );
}

static size_t file_size_bytes(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);

    if (!file) {
        throw std::runtime_error("Could not open file: " + path);
    }

    return static_cast<size_t>(file.tellg());
}

static std::vector<int> read_int_column(const std::string& path) {
    const size_t bytes = file_size_bytes(path);

    if (bytes % sizeof(int) != 0) {
        throw std::runtime_error("Invalid int column size: " + path);
    }

    std::vector<int> values(bytes / sizeof(int));

    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Could not open file: " + path);
    }

    file.read(
        reinterpret_cast<char*>(values.data()),
        static_cast<std::streamsize>(bytes)
    );

    if (static_cast<size_t>(file.gcount()) != bytes) {
        throw std::runtime_error("Could not read full file: " + path);
    }

    return values;
}

// ------------------------------------------------------------
// CPU SPJA query. Used for CPU-owned chunks and correctness check.
// ------------------------------------------------------------
static unsigned long long spja_cpu_columnar(
    const int* orderkey,
    const int* quantity,
    const int* extendedprice,
    const int* order_custkey,
    const int* customer_nation,
    size_t n_rows,
    int order_count,
    int customer_count,
    int target_nation
) {
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
// ------------------------------------------------------------
// GPU query kernel. Each block produces one partial aggregate.
// ------------------------------------------------------------
__global__ void spja_gpu_columnar_kernel(
    const int* orderkey,
    const int* quantity,
    const int* extendedprice,
    const int* order_custkey,
    const int* customer_nation,
    size_t n_rows,
    int order_count,
    int customer_count,
    int target_nation,
    unsigned long long* block_sums
) {
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
                    local_sum =
                        static_cast<unsigned long long>(extendedprice[i]);
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
// ------------------------------------------------------------
// Reduce block sums into one device-side result.
// ------------------------------------------------------------
__global__ void reduce_block_sums_kernel(
    const unsigned long long* block_sums,
    size_t n_blocks,
    unsigned long long* result
) {
    extern __shared__ unsigned long long shared_sum[];

    const unsigned int tid = threadIdx.x;
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned long long local_sum = 0ULL;

    if (i < n_blocks) {
        local_sum = block_sums[i];
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
        atomicAdd(result, shared_sum[0]);
    }
}

// ------------------------------------------------------------
// Chunked compressed representation of one column.
// ------------------------------------------------------------
struct CompressedColumn {
    std::vector<size_t> uncomp_sizes;
    std::vector<size_t> comp_sizes;
    std::vector<size_t> comp_offsets;
    std::vector<std::vector<char>> comp_chunks;

    size_t total_comp_bytes = 0;
};

// ------------------------------------------------------------
// Compress one integer column chunk-wise using LZ4_HC.
// ------------------------------------------------------------
static CompressedColumn compress_column_lz4_hc(
    const int* data,
    size_t n_rows,
    size_t chunk_rows,
    int lz4_hc_level
) {
    CompressedColumn out;

    const size_t total_chunks =
        (n_rows + chunk_rows - 1) / chunk_rows;

    out.uncomp_sizes.resize(total_chunks);
    out.comp_sizes.resize(total_chunks);
    out.comp_offsets.resize(total_chunks);
    out.comp_chunks.resize(total_chunks);

    out.total_comp_bytes = 0;

    const char* bytes =
        reinterpret_cast<const char*>(data);

    for (size_t c = 0; c < total_chunks; ++c) {
        const size_t start_row = c * chunk_rows;

        const size_t rows_this =
            std::min(chunk_rows, n_rows - start_row);

        const size_t uncomp_bytes =
            rows_this * sizeof(int);

        const size_t byte_offset =
            start_row * sizeof(int);

        out.uncomp_sizes[c] = uncomp_bytes;

        const int max_comp_size =
            LZ4_compressBound(static_cast<int>(uncomp_bytes));

        out.comp_chunks[c].resize(
            static_cast<size_t>(max_comp_size)
        );

        const int comp_size =
            LZ4_compress_HC(
                bytes + byte_offset,
                out.comp_chunks[c].data(),
                static_cast<int>(uncomp_bytes),
                max_comp_size,
                lz4_hc_level
            );

        if (comp_size <= 0) {
            throw std::runtime_error("LZ4_HC compression failed.");
        }

        out.comp_chunks[c].resize(static_cast<size_t>(comp_size));
        out.comp_sizes[c] = static_cast<size_t>(comp_size);
        out.comp_offsets[c] = out.total_comp_bytes;
        out.total_comp_bytes += static_cast<size_t>(comp_size);
    }

    return out;
}

// ------------------------------------------------------------
// GPU batch representation.
// ------------------------------------------------------------
struct GpuBatch {
    size_t first_chunk = 0;
    size_t num_chunks = 0;
    size_t rows = 0;
    size_t comp_bytes = 0;

    std::vector<char> comp_flat;

    std::vector<size_t> comp_sizes;
    std::vector<size_t> uncomp_sizes;
    std::vector<size_t> comp_offsets;

    std::vector<int> column_ids;
    std::vector<size_t> row_offsets;
};

// ------------------------------------------------------------
// CUDA events used for per-batch timing.
// ------------------------------------------------------------
struct BatchTimingEvents {
    cudaEvent_t batch_start = nullptr;
    cudaEvent_t h2d_end = nullptr;
    cudaEvent_t decomp_end = nullptr;
    cudaEvent_t kernel_end = nullptr;
};

static void create_batch_events(BatchTimingEvents& ev) {
    CUDA_CHECK(cudaEventCreate(&ev.batch_start));
    CUDA_CHECK(cudaEventCreate(&ev.h2d_end));
    CUDA_CHECK(cudaEventCreate(&ev.decomp_end));
    CUDA_CHECK(cudaEventCreate(&ev.kernel_end));
}

static void destroy_batch_events(BatchTimingEvents& ev) {
    CUDA_CHECK(cudaEventDestroy(ev.batch_start));
    CUDA_CHECK(cudaEventDestroy(ev.h2d_end));
    CUDA_CHECK(cudaEventDestroy(ev.decomp_end));
    CUDA_CHECK(cudaEventDestroy(ev.kernel_end));
}

// ------------------------------------------------------------
// Build a GPU batch from an explicit list of chunk IDs.
// This supports fair deterministic chunk assignments.
// ------------------------------------------------------------
static GpuBatch build_gpu_batch_from_chunk_ids(
    const std::vector<size_t>& gpu_chunk_ids,
    size_t first_index,
    size_t num_chunks,
    size_t n_rows,
    size_t chunk_rows,
    const CompressedColumn& comp_partkey,
    const CompressedColumn& comp_quantity,
    const CompressedColumn& comp_extendedprice
) {
    GpuBatch batch;

    batch.first_chunk = first_index;
    batch.num_chunks = num_chunks;

    const CompressedColumn* cols[3] = {
        &comp_partkey,
        &comp_quantity,
        &comp_extendedprice
    };

    size_t local_row_offset = 0;
    size_t flat_offset = 0;

    for (size_t j = 0; j < num_chunks; ++j) {
        const size_t c = gpu_chunk_ids[first_index + j];
        const size_t global_start_row = c * chunk_rows;

        const size_t rows_this =
            std::min(chunk_rows, n_rows - global_start_row);

        const size_t uncomp_bytes = rows_this * sizeof(int);

        for (int col = 0; col < 3; ++col) {
            const CompressedColumn& cc = *cols[col];

            batch.comp_offsets.push_back(flat_offset);
            batch.comp_sizes.push_back(cc.comp_sizes[c]);
            batch.uncomp_sizes.push_back(uncomp_bytes);
            batch.column_ids.push_back(col);
            batch.row_offsets.push_back(local_row_offset);

            flat_offset += cc.comp_sizes[c];
        }

        local_row_offset += rows_this;
    }

    batch.rows = local_row_offset;
    batch.comp_bytes = flat_offset;
    batch.comp_flat.resize(batch.comp_bytes);

    flat_offset = 0;

    const CompressedColumn* batch_cols[3] = {
        &comp_partkey,
        &comp_quantity,
        &comp_extendedprice
    };

    for (size_t j = 0; j < num_chunks; ++j) {
        const size_t c = gpu_chunk_ids[first_index + j];

        for (int col = 0; col < 3; ++col) {
            const CompressedColumn& cc = *batch_cols[col];

            std::memcpy(
                batch.comp_flat.data() + flat_offset,
                cc.comp_chunks[c].data(),
                cc.comp_sizes[c]
            );

            flat_offset += cc.comp_sizes[c];
        }
    }

    return batch;
}

// ------------------------------------------------------------
// Greatest common divisor for deterministic fair chunk strides.
// ------------------------------------------------------------
static size_t gcd_size_t(size_t a, size_t b) {
    while (b != 0) {
        const size_t r = a % b;
        a = b;
        b = r;
    }

    return a;
}

// ------------------------------------------------------------
// Pick a stride that is coprime with total_chunks.
// This creates a full deterministic permutation of chunk IDs.
// ------------------------------------------------------------
static size_t choose_coprime_stride(size_t total_chunks, int trial) {
    const size_t candidates[] = {
        1, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37
    };

    const size_t n_candidates =
        sizeof(candidates) / sizeof(candidates[0]);

    for (size_t i = 0; i < n_candidates; ++i) {
        const size_t s =
            candidates[(static_cast<size_t>(trial) + i) % n_candidates];

        if (s < total_chunks && gcd_size_t(s, total_chunks) == 1) {
            return s;
        }
    }

    return 1;
}

// ------------------------------------------------------------
// Build deterministic fair CPU/GPU chunk assignment.
// Each trial uses a different full permutation. This avoids
// relying on one fixed sorted chunk layout.
// ------------------------------------------------------------
static void build_fair_chunk_assignment(
    size_t total_chunks,
    size_t gpu_chunks,
    int trial,
    std::vector<size_t>& cpu_chunk_ids,
    std::vector<size_t>& gpu_chunk_ids
) {
    cpu_chunk_ids.clear();
    gpu_chunk_ids.clear();

    cpu_chunk_ids.reserve(total_chunks - gpu_chunks);
    gpu_chunk_ids.reserve(gpu_chunks);

    if (total_chunks == 0) {
        return;
    }

    const size_t stride =
        choose_coprime_stride(total_chunks, trial);

    const size_t offset =
        (static_cast<size_t>(trial) * 17ULL) % total_chunks;

    for (size_t i = 0; i < total_chunks; ++i) {
        const size_t c =
            (offset + i * stride) % total_chunks;

        if (i < gpu_chunks) {
            gpu_chunk_ids.push_back(c);
        } else {
            cpu_chunk_ids.push_back(c);
        }
    }

    std::sort(cpu_chunk_ids.begin(), cpu_chunk_ids.end());
    std::sort(gpu_chunk_ids.begin(), gpu_chunk_ids.end());
}

// ------------------------------------------------------------
// Main benchmark
// ------------------------------------------------------------
int main() {
    try {
        CUDA_CHECK(cudaSetDevice(0));

        std::system("mkdir -p bin");
        std::system("mkdir -p results/spja_workload/csv");

        const std::string partkey_path =
            "data/tpch_columnar/partkey_sf1.bin";

        const std::string quantity_path =
            "data/tpch_columnar/quantity_sf1.bin";

        const std::string extendedprice_path =
            "data/tpch_columnar/extendedprice_sf1.bin";

        const std::string orderkey_path =
            "data/tpch_columnar/orderkey_sf1.bin";

        const std::string order_custkey_path =
            "data/tpch_columnar/order_custkey_sf1.bin";

        const std::string customer_nation_path =
            "data/tpch_columnar/customer_nation_sf1.bin";

        const std::string part_category_path =
            "data/tpch_columnar/part_category_sf1.bin";

        const std::string part_factor_path =
            "data/tpch_columnar/part_factor_sf1.bin";

        const size_t chunk_bytes = 1ULL << 19;   // 512 KiB chunks for official SF=1
        const size_t chunk_rows = chunk_bytes / sizeof(int);
        const size_t gpu_batch_chunks = 86;      // one GPU batch for SF=1-sized input

        const int lz4_hc_level = 8;
        const int warmup = 5;
        const int iterations = 20;
        const int assignment_trials = 5;

        const bool detailed_gpu_stage_timing = true;

        const std::vector<int> gpu_percents = {
            0, 25, 50, 75, 100
        };

        std::vector<int> host_partkey =
            read_int_column(orderkey_path);

        std::vector<int> host_quantity =
            read_int_column(quantity_path);

        std::vector<int> host_extendedprice =
            read_int_column(extendedprice_path);

        
        if (host_partkey.size() != host_quantity.size() ||
            host_partkey.size() != host_extendedprice.size()) {
            throw std::runtime_error("LINEITEM column sizes do not match.");
        }

        std::vector<int> host_order_custkey =
            read_int_column(order_custkey_path);

        std::vector<int> host_customer_nation =
            read_int_column(customer_nation_path);

        const size_t n_rows = host_partkey.size();

        const int order_count =
            static_cast<int>(host_order_custkey.size());

        const int customer_count =
            static_cast<int>(host_customer_nation.size());

        const int target_nation = 3;

        const size_t total_chunks =
            (n_rows + chunk_rows - 1) / chunk_rows;

        const size_t total_query_bytes =
            n_rows * sizeof(int) * 3;

        const double input_mib =
            bytes_to_mib(total_query_bytes);

        const double input_gib =
            bytes_to_gib(total_query_bytes);

        std::cout << std::fixed << std::setprecision(3);

        std::cout << "Loaded official tpch-dbgen SF=1 columnar data:\n";
        std::cout << "  Rows: " << n_rows << "\n";
        std::cout << "  Query input size MiB: " << input_mib << "\n";
        std::cout << "  Chunks per column: " << total_chunks << "\n";
        std::cout << "  GPU batch chunks: " << gpu_batch_chunks << "\n";
        std::cout << "  LZ4_HC level: " << lz4_hc_level << "\n";
        std::cout << "  Warmup runs per assignment: " << warmup << "\n";
        std::cout << "  Timed runs per assignment: " << iterations << "\n";
        std::cout << "  Assignment trials per split: " << assignment_trials << "\n";
        std::cout << "  Execution mode: CPU/GPU overlap using std::thread + CUDA stream\n";
        std::cout << "  Chunk assignment: multiple deterministic fair assignments\n\n";

        const unsigned long long reference_result =
            spja_cpu_columnar(
                host_partkey.data(),
                host_quantity.data(),
                host_extendedprice.data(),
                host_order_custkey.data(),
                host_customer_nation.data(),
                n_rows,
                order_count,
                customer_count,
                target_nation
            );

        auto comp_start =
            std::chrono::high_resolution_clock::now();

        CompressedColumn comp_partkey =
            compress_column_lz4_hc(
                host_partkey.data(),
                n_rows,
                chunk_rows,
                lz4_hc_level
            );
        CompressedColumn comp_quantity =
            compress_column_lz4_hc(
                host_quantity.data(),
                n_rows,
                chunk_rows,
                lz4_hc_level
            );

        CompressedColumn comp_extendedprice =
            compress_column_lz4_hc(
                host_extendedprice.data(),
                n_rows,
                chunk_rows,
                lz4_hc_level
            );

        auto comp_end =
            std::chrono::high_resolution_clock::now();

        const double preprocess_compress_ms =
            ms_between(comp_start, comp_end);

        const size_t partkey_original_bytes =
            n_rows * sizeof(int);

        const size_t quantity_original_bytes =
            n_rows * sizeof(int);

        const size_t extendedprice_original_bytes =
            n_rows * sizeof(int);

        const size_t total_comp_bytes =
            comp_partkey.total_comp_bytes +
            comp_quantity.total_comp_bytes +
            comp_extendedprice.total_comp_bytes;

        const double partkey_reduction_percent =
            compression_reduction(
                partkey_original_bytes,
                comp_partkey.total_comp_bytes
            );

        const double quantity_reduction_percent =
            compression_reduction(
                quantity_original_bytes,
                comp_quantity.total_comp_bytes
            );

        const double extendedprice_reduction_percent =
            compression_reduction(
                extendedprice_original_bytes,
                comp_extendedprice.total_comp_bytes
            );

        const double compression_reduction_percent =
            compression_reduction(total_query_bytes, total_comp_bytes);

        std::cout << "Preprocessing compression:\n";
        std::cout << "  Original MiB: " << input_mib << "\n";
        std::cout << "  Compressed MiB: "
                  << bytes_to_mib(total_comp_bytes) << "\n";
        std::cout << "  Compression reduction %: "
                  << compression_reduction_percent << "%" << "\n";
        std::cout << "  Compression %: "
                  << (100 - compression_reduction_percent)<< "%" << "\n";
        std::cout << "  Compression time not counted ms: "
                  << preprocess_compress_ms << "\n\n";

        std::cout << "Per-column compression statistics:\n";
        std::cout << std::left
                  << std::setw(18) << "Column"
                  << std::setw(18) << "Original MiB"
                  << std::setw(18) << "Compressed MiB"
                  << std::setw(18) << "Reduction %"
                  << "\n";

        std::cout << std::left
                  << std::setw(18) << "orderkey"

                  << std::setw(18) << bytes_to_mib(partkey_original_bytes)
                  << std::setw(18) << bytes_to_mib(comp_partkey.total_comp_bytes)
                  << std::setw(18) << partkey_reduction_percent
                  << "\n";

        std::cout << std::left
                  << std::setw(18) << "quantity"
                  << std::setw(18) << bytes_to_mib(quantity_original_bytes)
                  << std::setw(18) << bytes_to_mib(comp_quantity.total_comp_bytes)
                  << std::setw(18) << quantity_reduction_percent
                  << "\n";

        std::cout << std::left
                  << std::setw(18) << "extendedprice"
                  << std::setw(18) << bytes_to_mib(extendedprice_original_bytes)
                  << std::setw(18) << bytes_to_mib(comp_extendedprice.total_comp_bytes)
                  << std::setw(18) << extendedprice_reduction_percent
                  << "\n";

        std::cout << std::left
                  << std::setw(18) << "total"
                  << std::setw(18) << bytes_to_mib(total_query_bytes)
                  << std::setw(18) << bytes_to_mib(total_comp_bytes)
                  << std::setw(18) << compression_reduction_percent
                  << "\n\n";

        // ------------------------------------------------------------
        // Write compression statistics and reproducibility metadata.
        // ------------------------------------------------------------
        std::ofstream compression_csv(COMPRESSION_CSV_PATH);

        compression_csv
            << "Column,Original_MiB,Compressed_MiB,Reduction_Percent\n";

        compression_csv
            << "orderkey,"

            << bytes_to_mib(partkey_original_bytes) << ","
            << bytes_to_mib(comp_partkey.total_comp_bytes) << ","
            << partkey_reduction_percent << "\n";

        compression_csv
            << "quantity,"
            << bytes_to_mib(quantity_original_bytes) << ","
            << bytes_to_mib(comp_quantity.total_comp_bytes) << ","
            << quantity_reduction_percent << "\n";

        compression_csv
            << "extendedprice,"
            << bytes_to_mib(extendedprice_original_bytes) << ","
            << bytes_to_mib(comp_extendedprice.total_comp_bytes) << ","
            << extendedprice_reduction_percent << "\n";

        compression_csv
            << "total,"
            << bytes_to_mib(total_query_bytes) << ","
            << bytes_to_mib(total_comp_bytes) << ","
            << compression_reduction_percent << "\n";

        compression_csv.close();

        std::ofstream metadata(METADATA_PATH);

        metadata << "Benchmark metadata\n";
        metadata << "==================\n";
        metadata << "Dataset: official tpch-dbgen SF=1\n";
        metadata << "Input format: converted .tbl files to binary int32 columns\n";
        metadata << "Columns: orderkey, quantity, extendedprice\n";
        metadata << "Query: LINEITEM + ORDERS + CUSTOMER using orderkey -> custkey -> nation\n";
        metadata << "Rows: " << n_rows << "\n";
        metadata << "Input MiB: " << input_mib << "\n";
        metadata << "Total chunks per column: " << total_chunks << "\n";
        metadata << "Chunk size bytes: " << chunk_bytes << "\n";
        metadata << "Chunk size KiB: " << static_cast<double>(chunk_bytes) / 1024.0 << "\n";
        metadata << "GPU batch chunks: " << gpu_batch_chunks << "\n";
        metadata << "Compression: LZ4_HC level " << lz4_hc_level << "\n";
        metadata << "GPU decompression: nvCOMP LZ4\n";
        metadata << "Execution: CPU/GPU overlap using std::thread + CUDA stream\n";
        metadata << "Assignment trials per split: " << assignment_trials << "\n";
        metadata << "Warmup runs per assignment: " << warmup << "\n";
        metadata << "Timed runs per assignment: " << iterations << "\n";
        metadata << "Total compression reduction percent: "
                 << compression_reduction_percent << "\n";
        metadata << "Preprocess compression ms not counted: "
                 << preprocess_compress_ms << "\n";

        metadata.close();

        // ------------------------------------------------------------
        // Main aggregate CSV. Filename is kept unchanged.
        // ------------------------------------------------------------
        std::ofstream csv(MAIN_CSV_PATH);

        csv << "Mode,Input_MiB,CPU_Percent,GPU_Percent,"
            << "Assignment_Trials,Timed_Runs_Per_Assignment,"
            << "Total_Rows,CPU_Rows_Avg,GPU_Rows_Avg,"
            << "Total_Chunks,CPU_Chunks_Avg,GPU_Chunks_Avg,"
            << "Compression_Reduction_Percent,"
            << "Preprocess_Compress_ms_Not_Counted,"
            << "CPU_Decomp_ms_Avg,CPU_Decomp_ms_StdDev,"
            << "CPU_SPJA_ms_Avg,CPU_SPJA_ms_StdDev,"
            << "CPU_Total_ms_Avg,CPU_Total_ms_StdDev,"
            << "GPU_H2D_Comp_ms_Avg,GPU_H2D_Comp_ms_StdDev,"
            << "GPU_Decomp_ms_Avg,GPU_Decomp_ms_StdDev,"
            << "GPU_SPJA_ms_Avg,GPU_SPJA_ms_StdDev,"
            << "GPU_D2H_Result_ms_Avg,GPU_D2H_Result_ms_StdDev,"
            << "GPU_Total_ms_Avg,GPU_Total_ms_StdDev,"
            << "CPU_Effective_GiBps_Avg,CPU_Effective_GiBps_StdDev,"
            << "GPU_Effective_GiBps_Avg,GPU_Effective_GiBps_StdDev,"
            << "Balance_Difference_ms,Total_ms_Avg,Total_ms_StdDev,"
            << "Effective_GiBps_Avg,Effective_GiBps_StdDev,"
            << "Physical_GiBps_Avg,Physical_GiBps_StdDev,"
            << "Overlap_Efficiency_Avg,Overlap_Efficiency_StdDev,"
            << "Final_Result,Reference_Result,Valid\n";

        // ------------------------------------------------------------
        // Detailed CSV. One row per CPU/GPU split and assignment trial.
        // ------------------------------------------------------------
        std::ofstream detailed_csv(DETAILED_CSV_PATH);

        detailed_csv
            << "Mode,CPU_Percent,GPU_Percent,"
            << "Assignment_Trial,"
            << "CPU_Rows,GPU_Rows,"
            << "CPU_Chunks,GPU_Chunks,"
            << "CPU_Decomp_ms_Avg,"
            << "CPU_SPJA_ms_Avg,"
            << "CPU_Total_ms_Avg,"
            << "GPU_H2D_Comp_ms_Avg,"
            << "GPU_Decomp_ms_Avg,"
            << "GPU_SPJA_ms_Avg,"
            << "GPU_D2H_Result_ms_Avg,"
            << "GPU_Total_ms_Avg,"
            << "Balance_Difference_ms_Avg,"
            << "Total_ms_Avg,"
            << "Effective_GiBps_Avg,"
            << "Physical_GiBps_Avg,"
            << "Overlap_Efficiency_Avg,"
            << "CPU_Effective_GiBps_Avg,"
            << "GPU_Effective_GiBps_Avg,"
            << "Final_Result,"
            << "Reference_Result,"
            << "Valid\n";

        std::cout << std::left
                  << std::setw(18) << "MODE"
                  << std::setw(12) << "MiB"
                  << std::setw(10) << "CPU%"
                  << std::setw(10) << "GPU%"
                  << std::setw(12) << "Trials"
                  << std::setw(14) << "CPU ms"
                  << std::setw(14) << "GPU ms"
                  << std::setw(14) << "DIFF ms"
                  << std::setw(14) << "TOTAL ms"
                  << std::setw(14) << "EFF GiB/s"
                  << std::setw(12) << "MATCH?"
                  << "\n";

        std::cout
            << "------------------------------------------------------------------------------------------------------------------------------------------------\n";

        std::vector<int> summary_cpu_percent;
        std::vector<int> summary_gpu_percent;
        std::vector<double> summary_total_ms;
        std::vector<double> summary_eff_gibps;
        std::vector<bool> summary_valid;

        for (int gpu_percent : gpu_percents) {
            const int cpu_percent = 100 - gpu_percent;

            const size_t gpu_chunks =
                (total_chunks * gpu_percent) / 100;

            std::vector<double> trial_cpu_rows;
            std::vector<double> trial_gpu_rows;
            std::vector<double> trial_cpu_chunks;
            std::vector<double> trial_gpu_chunks;

            std::vector<double> trial_cpu_decomp_ms;
            std::vector<double> trial_cpu_spja_ms;
            std::vector<double> trial_cpu_total_ms;

            std::vector<double> trial_gpu_h2d_ms;
            std::vector<double> trial_gpu_decomp_ms;
            std::vector<double> trial_gpu_spja_ms;
            std::vector<double> trial_gpu_d2h_ms;
            std::vector<double> trial_gpu_total_ms;

            std::vector<double> trial_cpu_eff_gibps;
            std::vector<double> trial_gpu_eff_gibps;
            std::vector<double> trial_balance_diff_ms;
            std::vector<double> trial_total_ms;
            std::vector<double> trial_eff_gibps;
            std::vector<double> trial_phys_gibps;
            std::vector<double> trial_overlap_efficiency;

            bool all_valid = true;
            unsigned long long last_final_result_for_split = 0ULL;

            for (int assignment_trial = 0;
                 assignment_trial < assignment_trials;
                 ++assignment_trial) {

                std::vector<size_t> cpu_chunk_ids;
                std::vector<size_t> gpu_chunk_ids;

                build_fair_chunk_assignment(
                    total_chunks,
                    gpu_chunks,
                    assignment_trial,
                    cpu_chunk_ids,
                    gpu_chunk_ids
                );

                size_t cpu_rows = 0;

                for (size_t idx = 0; idx < cpu_chunk_ids.size(); ++idx) {
                    const size_t c = cpu_chunk_ids[idx];
                    const size_t start = c * chunk_rows;

                    cpu_rows += std::min(chunk_rows, n_rows - start);
                }

                size_t gpu_rows = 0;

                for (size_t idx = 0; idx < gpu_chunk_ids.size(); ++idx) {
                    const size_t c = gpu_chunk_ids[idx];
                    const size_t start = c * chunk_rows;

                    gpu_rows += std::min(chunk_rows, n_rows - start);
                }

                std::vector<GpuBatch> gpu_batches;

                if (!gpu_chunk_ids.empty()) {
                    for (size_t processed = 0;
                         processed < gpu_chunk_ids.size();
                         processed += gpu_batch_chunks) {

                        const size_t current_batch_chunks =
                            std::min(
                                gpu_batch_chunks,
                                gpu_chunk_ids.size() - processed
                            );

                        gpu_batches.push_back(
                            build_gpu_batch_from_chunk_ids(
                                gpu_chunk_ids,
                                processed,
                                current_batch_chunks,
                                n_rows,
                                chunk_rows,
                                comp_partkey,
                                comp_quantity,
                                comp_extendedprice
                            )
                        );
                    }
                }

                size_t max_batch_rows = 0;
                size_t max_batch_comp_bytes = 0;
                size_t max_batch_count = 0;

                for (const auto& batch : gpu_batches) {
                    max_batch_rows =
                        std::max(max_batch_rows, batch.rows);

                    max_batch_comp_bytes =
                        std::max(max_batch_comp_bytes, batch.comp_bytes);

                    max_batch_count =
                        std::max(max_batch_count, batch.comp_sizes.size());
                }

                std::vector<int> cpu_partkey(cpu_rows);
                std::vector<int> cpu_quantity(cpu_rows);
                std::vector<int> cpu_extendedprice(cpu_rows);

                cudaStream_t stream;
                CUDA_CHECK(cudaStreamCreate(&stream));

                char* h_pinned_comp = nullptr;

                char* d_comp_flat = nullptr;
                int* d_partkey = nullptr;
                int* d_quantity = nullptr;
                int* d_extendedprice = nullptr;

                void** d_comp_ptrs = nullptr;
                void** d_decomp_ptrs = nullptr;

                size_t* d_comp_sizes = nullptr;
                size_t* d_uncomp_sizes = nullptr;
                size_t* d_actual_uncomp_sizes = nullptr;

                nvcompStatus_t* d_statuses = nullptr;
                void* d_temp = nullptr;

                int* d_order_custkey = nullptr;
                int* d_customer_nation = nullptr;

                unsigned long long* d_gpu_result = nullptr;
                unsigned long long* d_block_sums = nullptr;

                size_t temp_bytes = 0;

                if (!gpu_chunk_ids.empty()) {
                    CUDA_CHECK(cudaMallocHost(
                        &h_pinned_comp,
                        max_batch_comp_bytes
                    ));

                    CUDA_CHECK(cudaMalloc(
                        &d_comp_flat,
                        max_batch_comp_bytes
                    ));

                    CUDA_CHECK(cudaMalloc(
                        &d_partkey,
                        max_batch_rows * sizeof(int)
                    ));

                    CUDA_CHECK(cudaMalloc(
                        &d_quantity,
                        max_batch_rows * sizeof(int)
                    ));

                    CUDA_CHECK(cudaMalloc(
                        &d_extendedprice,
                        max_batch_rows * sizeof(int)
                    ));

                    CUDA_CHECK(cudaMalloc(
                        &d_comp_ptrs,
                        max_batch_count * sizeof(void*)
                    ));

                    CUDA_CHECK(cudaMalloc(
                        &d_decomp_ptrs,
                        max_batch_count * sizeof(void*)
                    ));

                    CUDA_CHECK(cudaMalloc(
                        &d_comp_sizes,
                        max_batch_count * sizeof(size_t)
                    ));

                    CUDA_CHECK(cudaMalloc(
                        &d_uncomp_sizes,
                        max_batch_count * sizeof(size_t)
                    ));

                    CUDA_CHECK(cudaMalloc(
                        &d_actual_uncomp_sizes,
                        max_batch_count * sizeof(size_t)
                    ));

                    CUDA_CHECK(cudaMalloc(
                        &d_statuses,
                        max_batch_count * sizeof(nvcompStatus_t)
                    ));

                    CUDA_CHECK(cudaMalloc(
                        &d_order_custkey,
                        order_count * sizeof(int)
                    ));

                    CUDA_CHECK(cudaMalloc(
                        &d_customer_nation,
                        customer_count * sizeof(int)
                    ));


                    CUDA_CHECK(cudaMalloc(
                        &d_gpu_result,
                        sizeof(unsigned long long)
                    ));

                    const int block_size_for_alloc = 256;

                    const size_t max_spja_blocks =
                        (max_batch_rows + block_size_for_alloc - 1) /
                        block_size_for_alloc;

                    CUDA_CHECK(cudaMalloc(
                        &d_block_sums,
                        max_spja_blocks * sizeof(unsigned long long)
                    ));

                    CUDA_CHECK(cudaMemcpyAsync(
                        d_order_custkey,
                        host_order_custkey.data(),
                        order_count * sizeof(int),
                        cudaMemcpyHostToDevice,
                        stream
                    ));

                    CUDA_CHECK(cudaMemcpyAsync(
                        d_customer_nation,
                        host_customer_nation.data(),
                        customer_count * sizeof(int),
                        cudaMemcpyHostToDevice,
                        stream
                    ));

                    CUDA_CHECK(cudaStreamSynchronize(stream));

                    const nvcompBatchedLZ4DecompressOpts_t opts =
                        nvcompBatchedLZ4DecompressDefaultOpts;

                    for (const auto& batch : gpu_batches) {
                        const size_t batch_count =
                            batch.comp_sizes.size();

                        std::vector<void*> host_d_comp_ptrs(batch_count);
                        std::vector<void*> host_d_decomp_ptrs(batch_count);

                        for (size_t k = 0; k < batch_count; ++k) {
                            host_d_comp_ptrs[k] =
                                d_comp_flat + batch.comp_offsets[k];

                            const size_t row_offset =
                                batch.row_offsets[k];

                            const int col =
                                batch.column_ids[k];

                            if (col == 0) {
                                host_d_decomp_ptrs[k] =
                                    d_partkey + row_offset;
                            } else if (col == 1) {
                                host_d_decomp_ptrs[k] =
                                    d_quantity + row_offset;
                            } else {
                                host_d_decomp_ptrs[k] =
                                    d_extendedprice + row_offset;
                            }
                        }

                        CUDA_CHECK(cudaMemcpy(
                            d_comp_ptrs,
                            host_d_comp_ptrs.data(),
                            batch_count * sizeof(void*),
                            cudaMemcpyHostToDevice
                        ));

                        CUDA_CHECK(cudaMemcpy(
                            d_decomp_ptrs,
                            host_d_decomp_ptrs.data(),
                            batch_count * sizeof(void*),
                            cudaMemcpyHostToDevice
                        ));

                        CUDA_CHECK(cudaMemcpy(
                            d_comp_sizes,
                            batch.comp_sizes.data(),
                            batch_count * sizeof(size_t),
                            cudaMemcpyHostToDevice
                        ));

                        CUDA_CHECK(cudaMemcpy(
                            d_uncomp_sizes,
                            batch.uncomp_sizes.data(),
                            batch_count * sizeof(size_t),
                            cudaMemcpyHostToDevice
                        ));

                        size_t temp_candidate = 0;

                        NVCOMP_CHECK(
                            nvcompBatchedLZ4DecompressGetTempSizeSync(
                                (const void* const* const)d_comp_ptrs,
                                d_comp_sizes,
                                batch_count,
                                chunk_bytes,
                                &temp_candidate,
                                batch.rows * sizeof(int) * 3,
                                opts,
                                d_statuses,
                                stream
                            )
                        );

                        temp_bytes =
                            std::max(temp_bytes, temp_candidate);
                    }

                    CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));
                }

                std::vector<double> cpu_decomp_ms_runs;
                std::vector<double> cpu_spja_ms_runs;
                std::vector<double> cpu_total_ms_runs;

                std::vector<double> gpu_h2d_ms_runs;
                std::vector<double> gpu_decomp_ms_runs;
                std::vector<double> gpu_spja_ms_runs;
                std::vector<double> gpu_d2h_ms_runs;
                std::vector<double> gpu_total_ms_runs;

                std::vector<double> balance_diff_ms_runs;
                std::vector<double> total_ms_runs;
                std::vector<double> eff_gibps_runs;
                std::vector<double> phys_gibps_runs;
                std::vector<double> overlap_efficiency_runs;

                unsigned long long last_cpu_result = 0ULL;
                unsigned long long last_gpu_result = 0ULL;
                unsigned long long last_final_result = 0ULL;

                for (int it = 0; it < warmup + iterations; ++it) {
                    unsigned long long cpu_result = 0ULL;
                    unsigned long long gpu_result = 0ULL;

                    cudaEvent_t ev_gpu_start;
                    cudaEvent_t ev_gpu_end;

                    CUDA_CHECK(cudaEventCreate(&ev_gpu_start));
                    CUDA_CHECK(cudaEventCreate(&ev_gpu_end));

                    double cpu_decomp_ms = 0.0;
                    double cpu_spja_ms = 0.0;
                    double cpu_total_ms = 0.0;

                    double gpu_h2d_sum_ms = 0.0;
                    double gpu_decomp_sum_ms = 0.0;
                    double gpu_spja_sum_ms = 0.0;
                    double gpu_total_ms = 0.0;
                    double gpu_d2h_ms = 0.0;

                    std::exception_ptr cpu_exception = nullptr;

                    std::vector<BatchTimingEvents> batch_events;

                    if (!gpu_chunk_ids.empty() && detailed_gpu_stage_timing) {
                        batch_events.resize(gpu_batches.size());

                        for (auto& ev : batch_events) {
                            create_batch_events(ev);
                        }
                    }

                    auto total_start =
                        std::chrono::high_resolution_clock::now();

                    std::thread cpu_thread([&]() {
                        try {
                            auto cpu_start =
                                std::chrono::high_resolution_clock::now();

                            auto cpu_decomp_start =
                                std::chrono::high_resolution_clock::now();

                            if (!cpu_chunk_ids.empty()) {
                                size_t local_row_offset = 0;

                                for (size_t idx = 0;
                                     idx < cpu_chunk_ids.size();
                                     ++idx) {

                                    const size_t c = cpu_chunk_ids[idx];

                                    const size_t global_start =
                                        c * chunk_rows;

                                    const size_t rows_this =
                                        std::min(
                                            chunk_rows,
                                            n_rows - global_start
                                        );

                                    const int expected_bytes =
                                        static_cast<int>(
                                            rows_this * sizeof(int)
                                        );

                                    const CompressedColumn* cols[3] = {
                                        &comp_partkey,
                                        &comp_quantity,
                                        &comp_extendedprice
                                    };

                                    int* dst_cols[3] = {
                                        cpu_partkey.data(),
                                        cpu_quantity.data(),
                                        cpu_extendedprice.data()
                                    };

                                    for (int col = 0; col < 3; ++col) {
                                        const CompressedColumn& cc =
                                            *cols[col];

                                        const int compressed_size =
                                            static_cast<int>(
                                                cc.comp_sizes[c]
                                            );

                                        const int decompressed_size =
                                            LZ4_decompress_safe(
                                                cc.comp_chunks[c].data(),
                                                reinterpret_cast<char*>(
                                                    dst_cols[col] +
                                                    local_row_offset
                                                ),
                                                compressed_size,
                                                expected_bytes
                                            );

                                        if (decompressed_size != expected_bytes) {
                                            throw std::runtime_error(
                                                "CPU LZ4 column decompression failed."
                                            );
                                        }
                                    }

                                    local_row_offset += rows_this;
                                }
                            }

                            auto cpu_decomp_end =
                                std::chrono::high_resolution_clock::now();

                            auto cpu_spja_start =
                                std::chrono::high_resolution_clock::now();

                            if (cpu_rows > 0) {
                                    cpu_result =
                                        spja_cpu_columnar(
                                            cpu_partkey.data(),          // actually orderkey now
                                            cpu_quantity.data(),
                                            cpu_extendedprice.data(),
                                            host_order_custkey.data(),
                                            host_customer_nation.data(),
                                            cpu_rows,
                                            order_count,
                                            customer_count,
                                            target_nation
                                        );
                            }

                            auto cpu_spja_end =
                                std::chrono::high_resolution_clock::now();

                            auto cpu_end =
                                std::chrono::high_resolution_clock::now();

                            cpu_decomp_ms =
                                ms_between(cpu_decomp_start, cpu_decomp_end);

                            cpu_spja_ms =
                                ms_between(cpu_spja_start, cpu_spja_end);

                            cpu_total_ms =
                                ms_between(cpu_start, cpu_end);
                        }
                        catch (...) {
                            cpu_exception = std::current_exception();
                        }
                    });

                    if (!gpu_chunk_ids.empty()) {
                        const nvcompBatchedLZ4DecompressOpts_t opts =
                            nvcompBatchedLZ4DecompressDefaultOpts;

                        CUDA_CHECK(cudaMemsetAsync(
                            d_gpu_result,
                            0,
                            sizeof(unsigned long long),
                            stream
                        ));

                        CUDA_CHECK(cudaEventRecord(
                            ev_gpu_start,
                            stream
                        ));
                        //rewrtitng
                        for (size_t b = 0;
                             b < gpu_batches.size();
                             ++b) {

                            const auto& batch = gpu_batches[b];

                            const size_t batch_count =
                                batch.comp_sizes.size();

                            std::vector<void*> host_d_comp_ptrs(batch_count);
                            std::vector<void*> host_d_decomp_ptrs(batch_count);

                            for (size_t k = 0; k < batch_count; ++k) {
                                host_d_comp_ptrs[k] =
                                    d_comp_flat + batch.comp_offsets[k];

                                const size_t row_offset =
                                    batch.row_offsets[k];

                                const int col =
                                    batch.column_ids[k];

                                if (col == 0) {
                                    host_d_decomp_ptrs[k] =
                                        d_partkey + row_offset;
                                } else if (col == 1) {
                                    host_d_decomp_ptrs[k] =
                                        d_quantity + row_offset;
                                } else {
                                    host_d_decomp_ptrs[k] =
                                        d_extendedprice + row_offset;
                                }
                            }

                            if (detailed_gpu_stage_timing) {
                                CUDA_CHECK(cudaEventRecord(
                                    batch_events[b].batch_start,
                                    stream
                                ));
                            }

                            CUDA_CHECK(cudaMemcpyAsync(
                                d_comp_ptrs,
                                host_d_comp_ptrs.data(),
                                batch_count * sizeof(void*),
                                cudaMemcpyHostToDevice,
                                stream
                            ));

                            CUDA_CHECK(cudaMemcpyAsync(
                                d_decomp_ptrs,
                                host_d_decomp_ptrs.data(),
                                batch_count * sizeof(void*),
                                cudaMemcpyHostToDevice,
                                stream
                            ));

                            CUDA_CHECK(cudaMemcpyAsync(
                                d_comp_sizes,
                                batch.comp_sizes.data(),
                                batch_count * sizeof(size_t),
                                cudaMemcpyHostToDevice,
                                stream
                            ));

                            CUDA_CHECK(cudaMemcpyAsync(
                                d_uncomp_sizes,
                                batch.uncomp_sizes.data(),
                                batch_count * sizeof(size_t),
                                cudaMemcpyHostToDevice,
                                stream
                            ));

                            std::memcpy(
                                h_pinned_comp,
                                batch.comp_flat.data(),
                                batch.comp_bytes
                            );

                            CUDA_CHECK(cudaMemcpyAsync(
                                d_comp_flat,
                                h_pinned_comp,
                                batch.comp_bytes,
                                cudaMemcpyHostToDevice,
                                stream
                            ));

                            CUDA_CHECK(cudaStreamSynchronize(stream));

                            if (detailed_gpu_stage_timing) {
                                CUDA_CHECK(cudaEventRecord(
                                    batch_events[b].h2d_end,
                                    stream
                                ));
                            }

                            NVCOMP_CHECK(
                                nvcompBatchedLZ4GetDecompressSizeAsync(
                                    (const void* const*)d_comp_ptrs,
                                    d_comp_sizes,
                                    d_uncomp_sizes,
                                    batch_count,
                                    stream
                                )
                            );

                            NVCOMP_CHECK(
                                nvcompBatchedLZ4DecompressAsync(
                                    (const void* const*)d_comp_ptrs,
                                    d_comp_sizes,
                                    d_uncomp_sizes,
                                    d_actual_uncomp_sizes,
                                    batch_count,
                                    d_temp,
                                    temp_bytes,
                                    d_decomp_ptrs,
                                    opts,
                                    d_statuses,
                                    stream
                                )
                            );

                            if (detailed_gpu_stage_timing) {
                                CUDA_CHECK(cudaEventRecord(
                                    batch_events[b].decomp_end,
                                    stream
                                ));
                            }

                            const int block_size = 256;

                            const int grid_size =
                                static_cast<int>(
                                    (batch.rows + block_size - 1) /
                                    block_size
                                );

                                spja_gpu_columnar_kernel<<<
                                    grid_size,
                                    block_size,
                                    block_size * sizeof(unsigned long long),
                                    stream
                                >>>(
                                    d_partkey,              // actually orderkey now
                                    d_quantity,
                                    d_extendedprice,
                                    d_order_custkey,
                                    d_customer_nation,
                                    batch.rows,
                                    order_count,
                                    customer_count,
                                    target_nation,
                                    d_block_sums
                                );

                            CUDA_CHECK(cudaGetLastError());

                            const int reduce_grid_size =
                                static_cast<int>(
                                    (static_cast<size_t>(grid_size) +
                                     block_size - 1) /
                                    block_size
                                );

                            reduce_block_sums_kernel<<<
                                reduce_grid_size,
                                block_size,
                                block_size * sizeof(unsigned long long),
                                stream
                            >>>(
                                d_block_sums,
                                static_cast<size_t>(grid_size),
                                d_gpu_result
                            );

                            CUDA_CHECK(cudaGetLastError());

                            if (detailed_gpu_stage_timing) {
                                CUDA_CHECK(cudaEventRecord(
                                    batch_events[b].kernel_end,
                                    stream
                                ));
                            }
                        }

                        CUDA_CHECK(cudaMemcpyAsync(
                            &gpu_result,
                            d_gpu_result,
                            sizeof(unsigned long long),
                            cudaMemcpyDeviceToHost,
                            stream
                        ));

                        CUDA_CHECK(cudaEventRecord(
                            ev_gpu_end,
                            stream
                        ));
                    }

                    if (!gpu_chunk_ids.empty()) {
                        CUDA_CHECK(cudaStreamSynchronize(stream));

                        float f_gpu_total = 0.0f;

                        CUDA_CHECK(cudaEventElapsedTime(
                            &f_gpu_total,
                            ev_gpu_start,
                            ev_gpu_end
                        ));

                        gpu_total_ms =
                            static_cast<double>(f_gpu_total);

                        if (detailed_gpu_stage_timing) {
                            for (auto& ev : batch_events) {
                                float f_h2d = 0.0f;
                                float f_decomp = 0.0f;
                                float f_spja = 0.0f;

                                CUDA_CHECK(cudaEventElapsedTime(
                                    &f_h2d,
                                    ev.batch_start,
                                    ev.h2d_end
                                ));

                                CUDA_CHECK(cudaEventElapsedTime(
                                    &f_decomp,
                                    ev.h2d_end,
                                    ev.decomp_end
                                ));

                                CUDA_CHECK(cudaEventElapsedTime(
                                    &f_spja,
                                    ev.decomp_end,
                                    ev.kernel_end
                                ));

                                gpu_h2d_sum_ms +=
                                    static_cast<double>(f_h2d);

                                gpu_decomp_sum_ms +=
                                    static_cast<double>(f_decomp);

                                gpu_spja_sum_ms +=
                                    static_cast<double>(f_spja);
                            }

                            gpu_d2h_ms =
                                std::max(
                                    0.0,
                                    gpu_total_ms -
                                    gpu_h2d_sum_ms -
                                    gpu_decomp_sum_ms -
                                    gpu_spja_sum_ms
                                );
                        }
                    }

                    cpu_thread.join();

                    if (cpu_exception) {
                        std::rethrow_exception(cpu_exception);
                    }

                    for (auto& ev : batch_events) {
                        destroy_batch_events(ev);
                    }

                    auto total_end =
                        std::chrono::high_resolution_clock::now();

                    const double total_ms =
                        ms_between(total_start, total_end);

                    const double balance_diff_ms =
                        std::fabs(cpu_total_ms - gpu_total_ms);

                    const unsigned long long final_result =
                        cpu_result + gpu_result;

                    const double eff_gibps =
                        input_gib / (total_ms / 1000.0);

                    const double phys_gib =
                        bytes_to_gib(total_comp_bytes);

                    const double phys_gibps =
                        phys_gib / (total_ms / 1000.0);

                    const double ideal_overlap_ms =
                        std::max(cpu_total_ms, gpu_total_ms);

                    const double overlap_efficiency =
                        (total_ms > 0.0)
                            ? ideal_overlap_ms / total_ms
                            : 0.0;

                    CUDA_CHECK(cudaEventDestroy(ev_gpu_start));
                    CUDA_CHECK(cudaEventDestroy(ev_gpu_end));

                    if (it >= warmup) {
                        cpu_decomp_ms_runs.push_back(cpu_decomp_ms);
                        cpu_spja_ms_runs.push_back(cpu_spja_ms);
                        cpu_total_ms_runs.push_back(cpu_total_ms);

                        gpu_h2d_ms_runs.push_back(gpu_h2d_sum_ms);
                        gpu_decomp_ms_runs.push_back(gpu_decomp_sum_ms);
                        gpu_spja_ms_runs.push_back(gpu_spja_sum_ms);
                        gpu_d2h_ms_runs.push_back(gpu_d2h_ms);
                        gpu_total_ms_runs.push_back(gpu_total_ms);

                        balance_diff_ms_runs.push_back(balance_diff_ms);
                        total_ms_runs.push_back(total_ms);
                        eff_gibps_runs.push_back(eff_gibps);
                        phys_gibps_runs.push_back(phys_gibps);
                        overlap_efficiency_runs.push_back(overlap_efficiency);

                        last_cpu_result = cpu_result;
                        last_gpu_result = gpu_result;
                        last_final_result = final_result;
                    }
                }

                const bool valid =
                    (last_final_result == reference_result);

                all_valid = all_valid && valid;
                last_final_result_for_split = last_final_result;

                const double avg_cpu_total_ms =
                    mean(cpu_total_ms_runs);

                const double avg_gpu_total_ms =
                    mean(gpu_total_ms_runs);

                const double avg_total_ms =
                    mean(total_ms_runs);

                const double cpu_input_gib =
                    input_gib *
                    (static_cast<double>(cpu_rows) /
                     static_cast<double>(n_rows));

                const double gpu_input_gib =
                    input_gib *
                    (static_cast<double>(gpu_rows) /
                     static_cast<double>(n_rows));

                const double avg_cpu_eff_gibps =
                    (avg_cpu_total_ms > 0.0)
                        ? cpu_input_gib / (avg_cpu_total_ms / 1000.0)
                        : 0.0;

                const double avg_gpu_eff_gibps =
                    (avg_gpu_total_ms > 0.0)
                        ? gpu_input_gib / (avg_gpu_total_ms / 1000.0)
                        : 0.0;

                trial_cpu_rows.push_back(static_cast<double>(cpu_rows));
                trial_gpu_rows.push_back(static_cast<double>(gpu_rows));
                trial_cpu_chunks.push_back(static_cast<double>(cpu_chunk_ids.size()));
                trial_gpu_chunks.push_back(static_cast<double>(gpu_chunk_ids.size()));

                trial_cpu_decomp_ms.push_back(mean(cpu_decomp_ms_runs));
                trial_cpu_spja_ms.push_back(mean(cpu_spja_ms_runs));
                trial_cpu_total_ms.push_back(avg_cpu_total_ms);

                trial_gpu_h2d_ms.push_back(mean(gpu_h2d_ms_runs));
                trial_gpu_decomp_ms.push_back(mean(gpu_decomp_ms_runs));
                trial_gpu_spja_ms.push_back(mean(gpu_spja_ms_runs));
                trial_gpu_d2h_ms.push_back(mean(gpu_d2h_ms_runs));
                trial_gpu_total_ms.push_back(avg_gpu_total_ms);

                trial_cpu_eff_gibps.push_back(avg_cpu_eff_gibps);
                trial_gpu_eff_gibps.push_back(avg_gpu_eff_gibps);
                trial_balance_diff_ms.push_back(mean(balance_diff_ms_runs));
                trial_total_ms.push_back(avg_total_ms);
                trial_eff_gibps.push_back(mean(eff_gibps_runs));
                trial_phys_gibps.push_back(mean(phys_gibps_runs));
                trial_overlap_efficiency.push_back(mean(overlap_efficiency_runs));

                detailed_csv
                    << "TPCH_FAIR_ASSIGN_DETAIL" << ","
                    << cpu_percent << ","
                    << gpu_percent << ","
                    << assignment_trial << ","
                    << cpu_rows << ","
                    << gpu_rows << ","
                    << cpu_chunk_ids.size() << ","
                    << gpu_chunk_ids.size() << ","
                    << mean(cpu_decomp_ms_runs) << ","
                    << mean(cpu_spja_ms_runs) << ","
                    << avg_cpu_total_ms << ","
                    << mean(gpu_h2d_ms_runs) << ","
                    << mean(gpu_decomp_ms_runs) << ","
                    << mean(gpu_spja_ms_runs) << ","
                    << mean(gpu_d2h_ms_runs) << ","
                    << avg_gpu_total_ms << ","
                    << mean(balance_diff_ms_runs) << ","
                    << avg_total_ms << ","
                    << mean(eff_gibps_runs) << ","
                    << mean(phys_gibps_runs) << ","
                    << mean(overlap_efficiency_runs) << ","
                    << avg_cpu_eff_gibps << ","
                    << avg_gpu_eff_gibps << ","
                    << last_final_result << ","
                    << reference_result << ","
                    << (valid ? "YES" : "NO")
                    << "\n";

                if (!gpu_chunk_ids.empty()) {
                    CUDA_CHECK(cudaFreeHost(h_pinned_comp));
                    CUDA_CHECK(cudaFree(d_comp_flat));
                    CUDA_CHECK(cudaFree(d_partkey));
                    CUDA_CHECK(cudaFree(d_quantity));
                    CUDA_CHECK(cudaFree(d_extendedprice));
                    CUDA_CHECK(cudaFree(d_comp_ptrs));
                    CUDA_CHECK(cudaFree(d_decomp_ptrs));
                    CUDA_CHECK(cudaFree(d_comp_sizes));
                    CUDA_CHECK(cudaFree(d_uncomp_sizes));
                    CUDA_CHECK(cudaFree(d_actual_uncomp_sizes));
                    CUDA_CHECK(cudaFree(d_statuses));
                    CUDA_CHECK(cudaFree(d_temp));
                    CUDA_CHECK(cudaFree(d_order_custkey));
                    CUDA_CHECK(cudaFree(d_customer_nation));
                    CUDA_CHECK(cudaFree(d_gpu_result));
                    CUDA_CHECK(cudaFree(d_block_sums));
                }

                CUDA_CHECK(cudaStreamDestroy(stream));
            }

            const double avg_cpu_total_ms =
                mean(trial_cpu_total_ms);

            const double avg_gpu_total_ms =
                mean(trial_gpu_total_ms);

            const double avg_balance_diff_ms =
                mean(trial_balance_diff_ms);

            const double avg_total_ms =
                mean(trial_total_ms);

            const double avg_eff_gibps =
                mean(trial_eff_gibps);

            std::string row_color = BLUE;

            if (avg_cpu_total_ms < avg_gpu_total_ms) {
                row_color = RED;
            } else if (avg_gpu_total_ms < avg_cpu_total_ms) {
                row_color = GREEN;
            }

            if (avg_balance_diff_ms <= 7.0) {
                row_color = BLUE;
            }

            std::cout << row_color
                      << std::left
                      << std::setw(18) << "TPCH_FAIR"
                      << std::setw(12) << input_mib
                      << std::setw(10) << cpu_percent
                      << std::setw(10) << gpu_percent
                      << std::setw(12) << assignment_trials
                      << std::setw(14) << avg_cpu_total_ms
                      << std::setw(14) << avg_gpu_total_ms
                      << std::setw(14) << avg_balance_diff_ms
                      << std::setw(14) << avg_total_ms
                      << std::setw(14) << avg_eff_gibps
                      << std::setw(12) << (all_valid ? "YES" : "NO")
                      << RESET
                      << "\n";

            csv << "TPCH_FAIR_ASSIGN" << ","
                << input_mib << ","
                << cpu_percent << ","
                << gpu_percent << ","
                << assignment_trials << ","
                << iterations << ","
                << n_rows << ","
                << mean(trial_cpu_rows) << ","
                << mean(trial_gpu_rows) << ","
                << total_chunks << ","
                << mean(trial_cpu_chunks) << ","
                << mean(trial_gpu_chunks) << ","
                << compression_reduction_percent << ","
                << preprocess_compress_ms << ","
                << mean(trial_cpu_decomp_ms) << ","
                << stddev_sample(trial_cpu_decomp_ms) << ","
                << mean(trial_cpu_spja_ms) << ","
                << stddev_sample(trial_cpu_spja_ms) << ","
                << mean(trial_cpu_total_ms) << ","
                << stddev_sample(trial_cpu_total_ms) << ","
                << mean(trial_gpu_h2d_ms) << ","
                << stddev_sample(trial_gpu_h2d_ms) << ","
                << mean(trial_gpu_decomp_ms) << ","
                << stddev_sample(trial_gpu_decomp_ms) << ","
                << mean(trial_gpu_spja_ms) << ","
                << stddev_sample(trial_gpu_spja_ms) << ","
                << mean(trial_gpu_d2h_ms) << ","
                << stddev_sample(trial_gpu_d2h_ms) << ","
                << mean(trial_gpu_total_ms) << ","
                << stddev_sample(trial_gpu_total_ms) << ","
                << mean(trial_cpu_eff_gibps) << ","
                << stddev_sample(trial_cpu_eff_gibps) << ","
                << mean(trial_gpu_eff_gibps) << ","
                << stddev_sample(trial_gpu_eff_gibps) << ","
                << mean(trial_balance_diff_ms) << ","
                << mean(trial_total_ms) << ","
                << stddev_sample(trial_total_ms) << ","
                << mean(trial_eff_gibps) << ","
                << stddev_sample(trial_eff_gibps) << ","
                << mean(trial_phys_gibps) << ","
                << stddev_sample(trial_phys_gibps) << ","
                << mean(trial_overlap_efficiency) << ","
                << stddev_sample(trial_overlap_efficiency) << ","
                << last_final_result_for_split << ","
                << reference_result << ","
                << (all_valid ? "YES" : "NO")
                << "\n";

            summary_cpu_percent.push_back(cpu_percent);
            summary_gpu_percent.push_back(gpu_percent);
            summary_total_ms.push_back(mean(trial_total_ms));
            summary_eff_gibps.push_back(mean(trial_eff_gibps));
            summary_valid.push_back(all_valid);
        }

        detailed_csv.close();

        // ------------------------------------------------------------
        // Final summary: CPU-only, GPU-only, and best valid split.
        // ------------------------------------------------------------
        int cpu_only_idx = -1;
        int gpu_only_idx = -1;
        int best_idx = -1;

        for (size_t i = 0; i < summary_gpu_percent.size(); ++i) {
            if (summary_gpu_percent[i] == 0) {
                cpu_only_idx = static_cast<int>(i);
            }

            if (summary_gpu_percent[i] == 100) {
                gpu_only_idx = static_cast<int>(i);
            }

            if (summary_valid[i]) {
                if (best_idx < 0 ||
                    summary_eff_gibps[i] > summary_eff_gibps[best_idx]) {
                    best_idx = static_cast<int>(i);
                }
            }
        }

        std::ofstream summary_file(SUMMARY_PATH);

        summary_file << "Benchmark summary\n";
        summary_file << "=================\n";

        if (cpu_only_idx >= 0) {
            summary_file << "CPU-only total ms: "
                         << summary_total_ms[cpu_only_idx] << "\n";
            summary_file << "CPU-only throughput GiB/s: "
                         << summary_eff_gibps[cpu_only_idx] << "\n";
        }

        if (gpu_only_idx >= 0) {
            summary_file << "GPU-only total ms: "
                         << summary_total_ms[gpu_only_idx] << "\n";
            summary_file << "GPU-only throughput GiB/s: "
                         << summary_eff_gibps[gpu_only_idx] << "\n";
        }

        if (best_idx >= 0) {
            summary_file << "Best split: "
                         << summary_cpu_percent[best_idx]
                         << "% CPU / "
                         << summary_gpu_percent[best_idx]
                         << "% GPU\n";

            summary_file << "Best total ms: "
                         << summary_total_ms[best_idx] << "\n";

            summary_file << "Best throughput GiB/s: "
                         << summary_eff_gibps[best_idx] << "\n";

            if (cpu_only_idx >= 0) {
                const double improvement_cpu =
                    (summary_total_ms[cpu_only_idx] -
                     summary_total_ms[best_idx]) /
                    summary_total_ms[cpu_only_idx] * 100.0;

                summary_file << "Improvement vs CPU-only %: "
                             << improvement_cpu << "\n";
            }

            if (gpu_only_idx >= 0) {
                const double improvement_gpu =
                    (summary_total_ms[gpu_only_idx] -
                     summary_total_ms[best_idx]) /
                    summary_total_ms[gpu_only_idx] * 100.0;

                summary_file << "Improvement vs GPU-only %: "
                             << improvement_gpu << "\n";
            }
        }

        summary_file.close();

        std::cout << "\nSummary:\n";

        if (best_idx >= 0) {
            std::cout << "  Best split: "
                      << summary_cpu_percent[best_idx]
                      << "% CPU / "
                      << summary_gpu_percent[best_idx]
                      << "% GPU\n";

            std::cout << "  Best total ms: "
                      << summary_total_ms[best_idx] << "\n";

            std::cout << "  Best throughput GiB/s: "
                      << summary_eff_gibps[best_idx] << "\n";
        }

        if (cpu_only_idx >= 0 && best_idx >= 0) {
            const double improvement_cpu =
                (summary_total_ms[cpu_only_idx] -
                 summary_total_ms[best_idx]) /
                summary_total_ms[cpu_only_idx] * 100.0;

            std::cout << "  Improvement vs CPU-only %: "
                      << improvement_cpu << "\n";
        }

        if (gpu_only_idx >= 0 && best_idx >= 0) {
            const double improvement_gpu =
                (summary_total_ms[gpu_only_idx] -
                 summary_total_ms[best_idx]) /
                summary_total_ms[gpu_only_idx] * 100.0;

            std::cout << "  Improvement vs GPU-only %: "
                      << improvement_gpu << "\n";
        }

        csv.close();

        std::cout << "\nDone. Main CSV written to:\n";
        std::cout << MAIN_CSV_PATH << "\n";

        std::cout << "\nAdditional files written to:\n";
        std::cout << DETAILED_CSV_PATH << "\n";
        std::cout << COMPRESSION_CSV_PATH << "\n";
        std::cout << METADATA_PATH << "\n";
        std::cout << SUMMARY_PATH << "\n";

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
}
