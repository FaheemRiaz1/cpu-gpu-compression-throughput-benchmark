#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t err__ = (call);                                    \
    if (err__ != cudaSuccess) {                                    \
        std::cerr << "CUDA error: " << cudaGetErrorString(err__)   \
                  << " at " << __FILE__ << ":" << __LINE__        \
                  << std::endl;                                    \
        std::exit(1);                                              \
    }                                                              \
} while (0)

static constexpr int BLOCK_THREADS = 128;
static constexpr int ITEMS_PER_THREAD = 4;
static constexpr int BLOCK_SIZE = 128;
static constexpr int MINIBLOCK_COUNT = 4;
static constexpr int TILE_ROWS = BLOCK_THREADS * ITEMS_PER_THREAD; // 512 rows
static constexpr int TARGET_NATION = 3;
static constexpr int WARMUP_RUNS = 1;
static constexpr int TIMED_RUNS = 5;

struct PackedColumn {
    std::vector<uint32_t> block_offsets; // nblocks + 1
    std::vector<uint32_t> data;          // header + packed blocks
    uint64_t original_rows = 0;
    uint64_t padded_rows = 0;
    uint64_t nblocks = 0;

    uint64_t bytes() const {
        return data.size() * sizeof(uint32_t)
             + block_offsets.size() * sizeof(uint32_t);
    }
};

struct PackedSlice {
    std::vector<uint32_t> block_offsets; // local offsets, nblocks_slice + 1
    std::vector<uint32_t> data;          // only packed block region, no global header
    uint64_t global_block_begin = 0;
    uint64_t nblocks = 0;

    uint64_t bytes() const {
        return data.size() * sizeof(uint32_t)
             + block_offsets.size() * sizeof(uint32_t);
    }
};

struct DevicePackedBuffer {
    uint32_t* d_offsets = nullptr;
    uint32_t* d_data = nullptr;
    uint64_t offsets_capacity = 0;
    uint64_t data_capacity = 0;
};

struct RunResult {
    int cpu_percent = 0;
    int gpu_percent = 0;
    double cpu_ms = 0.0;
    double gpu_ms = 0.0;   // COMMON TIMING: H2D encoded slice + fused kernel + D2H result
    double e2e_ms = 0.0;
    double eff_gib_s = 0.0;
    int64_t result = 0;
    bool match = false;
};

template <typename T>
std::vector<T> read_binary_vector(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        throw std::runtime_error("Cannot open file: " + path);
    }

    std::streamsize bytes = in.tellg();
    if (bytes < 0 || bytes % static_cast<std::streamsize>(sizeof(T)) != 0) {
        throw std::runtime_error("Bad binary size for file: " + path);
    }

    in.seekg(0, std::ios::beg);
    std::vector<T> data(static_cast<size_t>(bytes / sizeof(T)));
    if (!data.empty()) {
        in.read(reinterpret_cast<char*>(data.data()), bytes);
    }
    return data;
}

static inline uint32_t bit_width_u32(uint32_t x) {
    if (x == 0) return 0;
    return 32u - static_cast<uint32_t>(__builtin_clz(x));
}

// DPF-style CPU binpack/PFOR encoder.
// Same layout idea as DPFProto: block size 128, 4 miniblocks,
// one reference/min value per block, bitwidth metadata, then bit-packed values.
PackedColumn pack_dpf_u32_from_i32(const std::vector<int32_t>& input) {
    if (input.empty()) {
        throw std::runtime_error("Cannot pack empty column.");
    }

    PackedColumn out;
    out.original_rows = input.size();
    out.nblocks = (input.size() + BLOCK_SIZE - 1) / BLOCK_SIZE;
    out.padded_rows = out.nblocks * BLOCK_SIZE;

    std::vector<uint32_t> padded(static_cast<size_t>(out.padded_rows));
    for (size_t i = 0; i < input.size(); ++i) {
        padded[i] = static_cast<uint32_t>(input[i]);
    }

    // Padding rows are never processed, but values must be valid for packing.
    uint32_t last = static_cast<uint32_t>(input.back());
    for (size_t i = input.size(); i < padded.size(); ++i) {
        padded[i] = last;
    }

    out.block_offsets.resize(static_cast<size_t>(out.nblocks + 1));

    // Small header, similar to DPFProto pack.cu. Kernels do not use this header.
    out.data.push_back(BLOCK_SIZE);
    out.data.push_back(MINIBLOCK_COUNT);
    out.data.push_back(static_cast<uint32_t>(input.size()));
    out.data.push_back(static_cast<uint32_t>(input[0]));

    for (uint64_t b = 0; b < out.nblocks; ++b) {
        const uint64_t base = b * BLOCK_SIZE;
        out.block_offsets[static_cast<size_t>(b)] = static_cast<uint32_t>(out.data.size());

        uint32_t min_val = padded[static_cast<size_t>(base)];
        for (int i = 1; i < BLOCK_SIZE; ++i) {
            min_val = std::min(min_val, padded[static_cast<size_t>(base + i)]);
        }

        uint32_t diffs[BLOCK_SIZE];
        uint32_t max_diff = 0;
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            diffs[i] = padded[static_cast<size_t>(base + i)] - min_val;
            max_diff = std::max(max_diff, diffs[i]);
        }

        uint32_t bw = bit_width_u32(max_diff);

        // Block reference value
        out.data.push_back(min_val);

        // Same bitwidth for all four miniblocks, matching simple DPF-style binpack.
        uint32_t bitwidth_word = bw | (bw << 8) | (bw << 16) | (bw << 24);
        out.data.push_back(bitwidth_word);

        // Four miniblocks, each has 32 values.
        // Each miniblock uses exactly bw 32-bit words because 32 values * bw bits.
        for (int mb = 0; mb < MINIBLOCK_COUNT; ++mb) {
            if (bw == 0) {
                continue;
            }

            std::vector<uint32_t> words(bw, 0);
            for (int j = 0; j < 32; ++j) {
                uint32_t v = diffs[mb * 32 + j];
                uint32_t bitpos = static_cast<uint32_t>(j) * bw;
                uint32_t word = bitpos >> 5;
                uint32_t shift = bitpos & 31;

                words[word] |= (v << shift);
                if (shift + bw > 32) {
                    words[word + 1] |= (v >> (32 - shift));
                }
            }

            out.data.insert(out.data.end(), words.begin(), words.end());
        }
    }

    out.block_offsets[static_cast<size_t>(out.nblocks)] = static_cast<uint32_t>(out.data.size());

    // One padding word protects the decoder's 64-bit two-word read at the last block.
    out.data.push_back(0);

    return out;
}

// Create a host-side slice for only the GPU-owned block range.
// This lets common timing copy only the GPU-owned encoded representation.
PackedSlice make_slice(const PackedColumn& full, uint64_t block_begin, uint64_t block_end) {
    if (block_begin > block_end || block_end > full.nblocks) {
        throw std::runtime_error("Invalid packed slice range.");
    }

    PackedSlice s;
    s.global_block_begin = block_begin;
    s.nblocks = block_end - block_begin;

    if (s.nblocks == 0) {
        return s;
    }

    uint32_t data_begin = full.block_offsets[static_cast<size_t>(block_begin)];
    uint32_t data_end = full.block_offsets[static_cast<size_t>(block_end)];

    s.data.assign(full.data.begin() + data_begin, full.data.begin() + data_end);

    // Padding word for safe two-word decoder reads at final item.
    s.data.push_back(0);

    s.block_offsets.resize(static_cast<size_t>(s.nblocks + 1));
    for (uint64_t i = 0; i <= s.nblocks; ++i) {
        s.block_offsets[static_cast<size_t>(i)] =
            full.block_offsets[static_cast<size_t>(block_begin + i)] - data_begin;
    }

    return s;
}

void allocate_device_buffer(DevicePackedBuffer& d, uint64_t offsets_count, uint64_t data_count) {
    if (offsets_count > d.offsets_capacity) {
        if (d.d_offsets) CUDA_CHECK(cudaFree(d.d_offsets));
        CUDA_CHECK(cudaMalloc(&d.d_offsets, offsets_count * sizeof(uint32_t)));
        d.offsets_capacity = offsets_count;
    }

    if (data_count > d.data_capacity) {
        if (d.d_data) CUDA_CHECK(cudaFree(d.d_data));
        CUDA_CHECK(cudaMalloc(&d.d_data, data_count * sizeof(uint32_t)));
        d.data_capacity = data_count;
    }
}

void free_device_buffer(DevicePackedBuffer& d) {
    if (d.d_offsets) CUDA_CHECK(cudaFree(d.d_offsets));
    if (d.d_data) CUDA_CHECK(cudaFree(d.d_data));
    d = {};
}

void copy_slice_to_device(const PackedSlice& h, DevicePackedBuffer& d) {
    if (h.nblocks == 0) return;

    allocate_device_buffer(d, h.block_offsets.size(), h.data.size());

    CUDA_CHECK(cudaMemcpy(d.d_offsets, h.block_offsets.data(),
                          h.block_offsets.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d.d_data, h.data.data(),
                          h.data.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
}

__device__ __forceinline__
uint32_t dpf_decode_u32(
    const uint32_t* __restrict__ block_offsets,
    const uint32_t* __restrict__ data,
    uint32_t local_block_idx,
    uint32_t index_in_block)
{
    const uint32_t* block = data + block_offsets[local_block_idx];

    uint32_t reference = block[0];
    uint32_t bitwidths = block[1];

    uint32_t miniblock_index = index_in_block >> 5;       // / 32
    uint32_t index_into_miniblock = index_in_block & 31;  // % 32

    uint32_t bitwidth = (bitwidths >> (miniblock_index << 3)) & 255u;
    if (bitwidth == 0) {
        return reference;
    }

    // DPFProto-style prefix offset trick for 4 miniblocks:
    // offsets = [0, bw, 2bw, 3bw] when all miniblocks use same bw.
    uint32_t miniblock_offsets =
        (bitwidths << 8) + (bitwidths << 16) + (bitwidths << 24);
    uint32_t miniblock_offset =
        (miniblock_offsets >> (miniblock_index << 3)) & 255u;

    uint32_t start_bit = bitwidth * index_into_miniblock;
    uint32_t start_word = 2 + miniblock_offset + (start_bit >> 5);
    uint32_t shift = start_bit & 31;

    uint64_t two_words =
        (static_cast<uint64_t>(block[start_word + 1]) << 32)
        | static_cast<uint64_t>(block[start_word]);

    uint32_t mask = (bitwidth == 32) ? 0xffffffffu : ((1u << bitwidth) - 1u);
    uint32_t element = static_cast<uint32_t>((two_words >> shift) & mask);

    return reference + element;
}

__global__
void dpf_fused_spja_kernel_common(
    const uint32_t* __restrict__ orderkey_offsets,
    const uint32_t* __restrict__ orderkey_data,
    const uint32_t* __restrict__ quantity_offsets,
    const uint32_t* __restrict__ quantity_data,
    const uint32_t* __restrict__ extendedprice_offsets,
    const uint32_t* __restrict__ extendedprice_data,
    uint64_t global_block_begin,
    uint64_t nblocks_slice,
    const int32_t* __restrict__ order_custkey,
    uint64_t order_custkey_n,
    const int32_t* __restrict__ customer_nation,
    uint64_t customer_nation_n,
    uint64_t n_rows,
    unsigned long long* __restrict__ out_sum)
{
    __shared__ unsigned long long block_sums[BLOCK_THREADS];

    const uint32_t tid = threadIdx.x;
    const uint64_t n_local_tiles = (nblocks_slice + ITEMS_PER_THREAD - 1) / ITEMS_PER_THREAD;

    for (uint64_t tile = blockIdx.x; tile < n_local_tiles; tile += gridDim.x) {
        unsigned long long local = 0;

        uint64_t first_local_block = tile * ITEMS_PER_THREAD;

        #pragma unroll
        for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
            uint64_t local_block = first_local_block + item;

            if (local_block >= nblocks_slice) {
                continue;
            }

            uint64_t global_block = global_block_begin + local_block;
            uint64_t row = global_block * BLOCK_SIZE + tid;

            if (row >= n_rows) {
                continue;
            }

            uint32_t lb = static_cast<uint32_t>(local_block);

            int32_t q = static_cast<int32_t>(
                dpf_decode_u32(quantity_offsets, quantity_data, lb, tid));

            if (q <= 25) {
                continue;
            }

            int32_t ok = static_cast<int32_t>(
                dpf_decode_u32(orderkey_offsets, orderkey_data, lb, tid));

            if (ok < 0 || static_cast<uint64_t>(ok) >= order_custkey_n) {
                continue;
            }

            int32_t custkey = order_custkey[ok];
            if (custkey < 0 || static_cast<uint64_t>(custkey) >= customer_nation_n) {
                continue;
            }

            if (customer_nation[custkey] != TARGET_NATION) {
                continue;
            }

            int32_t ep = static_cast<int32_t>(
                dpf_decode_u32(extendedprice_offsets, extendedprice_data, lb, tid));

            local += static_cast<unsigned long long>(static_cast<int64_t>(ep));
        }

        block_sums[tid] = local;
        __syncthreads();

        for (uint32_t stride = BLOCK_THREADS / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                block_sums[tid] += block_sums[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0 && block_sums[0] != 0) {
            atomicAdd(out_sum, block_sums[0]);
        }

        __syncthreads();
    }
}

int64_t cpu_spja_range_raw(
    const std::vector<int32_t>& orderkey,
    const std::vector<int32_t>& quantity,
    const std::vector<int32_t>& extendedprice,
    const std::vector<int32_t>& order_custkey,
    const std::vector<int32_t>& customer_nation,
    uint64_t begin,
    uint64_t end)
{
    int64_t sum = 0;
    end = std::min<uint64_t>(end, orderkey.size());

    for (uint64_t i = begin; i < end; ++i) {
        if (quantity[i] <= 25) continue;

        int32_t ok = orderkey[i];
        if (ok < 0 || static_cast<uint64_t>(ok) >= order_custkey.size()) continue;

        int32_t custkey = order_custkey[ok];
        if (custkey < 0 || static_cast<uint64_t>(custkey) >= customer_nation.size()) continue;

        if (customer_nation[custkey] != TARGET_NATION) continue;

        sum += static_cast<int64_t>(extendedprice[i]);
    }

    return sum;
}

RunResult run_split_once_common(
    int cpu_percent,
    int gpu_percent,
    const std::vector<int32_t>& orderkey,
    const std::vector<int32_t>& quantity,
    const std::vector<int32_t>& extendedprice,
    const std::vector<int32_t>& order_custkey,
    const std::vector<int32_t>& customer_nation,
    const PackedSlice& s_orderkey,
    const PackedSlice& s_quantity,
    const PackedSlice& s_extendedprice,
    DevicePackedBuffer& d_orderkey,
    DevicePackedBuffer& d_quantity,
    DevicePackedBuffer& d_extendedprice,
    const int32_t* d_order_custkey,
    const int32_t* d_customer_nation,
    unsigned long long* d_sum,
    uint64_t n_tiles,
    int64_t reference)
{
    RunResult rr;
    rr.cpu_percent = cpu_percent;
    rr.gpu_percent = gpu_percent;

    uint64_t cpu_tiles = (n_tiles * static_cast<uint64_t>(cpu_percent)) / 100ULL;
    uint64_t gpu_start_tile = cpu_tiles;
    uint64_t gpu_end_tile = n_tiles;

    uint64_t cpu_begin_row = 0;
    uint64_t cpu_end_row = std::min<uint64_t>(orderkey.size(), cpu_tiles * TILE_ROWS);

    int64_t cpu_sum = 0;
    unsigned long long gpu_sum = 0;

    double cpu_ms = 0.0;
    double gpu_ms = 0.0;

    auto e2e_start = std::chrono::high_resolution_clock::now();

    std::thread cpu_thread;
    if (cpu_percent > 0) {
        cpu_thread = std::thread([&]() {
            auto t0 = std::chrono::high_resolution_clock::now();
            cpu_sum = cpu_spja_range_raw(orderkey, quantity, extendedprice,
                                         order_custkey, customer_nation,
                                         cpu_begin_row, cpu_end_row);
            auto t1 = std::chrono::high_resolution_clock::now();
            cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        });
    }

    if (gpu_percent > 0 && gpu_start_tile < gpu_end_tile) {
        auto tg0 = std::chrono::high_resolution_clock::now();

        // COMMON TIMING START:
        // H2D encoded GPU-owned slices
        copy_slice_to_device(s_orderkey, d_orderkey);
        copy_slice_to_device(s_quantity, d_quantity);
        copy_slice_to_device(s_extendedprice, d_extendedprice);

        // Result initialization is part of query-time GPU path.
        CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(unsigned long long)));

        uint64_t nblocks_slice = s_orderkey.nblocks;
        uint64_t global_block_begin = s_orderkey.global_block_begin;

        if (s_quantity.nblocks != nblocks_slice ||
            s_extendedprice.nblocks != nblocks_slice ||
            s_quantity.global_block_begin != global_block_begin ||
            s_extendedprice.global_block_begin != global_block_begin) {
            throw std::runtime_error("Packed slices do not match.");
        }

        uint64_t n_local_tiles = (nblocks_slice + ITEMS_PER_THREAD - 1) / ITEMS_PER_THREAD;

        int sm_count = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));

        int grid = static_cast<int>(std::min<uint64_t>(
            n_local_tiles, static_cast<uint64_t>(sm_count * 8)));
        grid = std::max(grid, 1);

        dpf_fused_spja_kernel_common<<<grid, BLOCK_THREADS>>>(
            d_orderkey.d_offsets, d_orderkey.d_data,
            d_quantity.d_offsets, d_quantity.d_data,
            d_extendedprice.d_offsets, d_extendedprice.d_data,
            global_block_begin,
            nblocks_slice,
            d_order_custkey, order_custkey.size(),
            d_customer_nation, customer_nation.size(),
            orderkey.size(),
            d_sum);

        CUDA_CHECK(cudaGetLastError());

        // D2H result is part of common timing.
        CUDA_CHECK(cudaMemcpy(&gpu_sum, d_sum, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize());

        auto tg1 = std::chrono::high_resolution_clock::now();
        gpu_ms = std::chrono::duration<double, std::milli>(tg1 - tg0).count();
    }

    if (cpu_percent > 0) {
        cpu_thread.join();
    }

    auto e2e_stop = std::chrono::high_resolution_clock::now();

    rr.cpu_ms = cpu_ms;
    rr.gpu_ms = gpu_ms;
    rr.e2e_ms = std::chrono::duration<double, std::milli>(e2e_stop - e2e_start).count();
    rr.result = cpu_sum + static_cast<int64_t>(gpu_sum);
    rr.match = (rr.result == reference);

    double input_gib = (static_cast<double>(orderkey.size()) * 3.0 * sizeof(int32_t))
                     / (1024.0 * 1024.0 * 1024.0);
    rr.eff_gib_s = input_gib / (rr.e2e_ms / 1000.0);

    return rr;
}

RunResult run_split_avg_common(
    int cpu_percent,
    int gpu_percent,
    const std::vector<int32_t>& orderkey,
    const std::vector<int32_t>& quantity,
    const std::vector<int32_t>& extendedprice,
    const std::vector<int32_t>& order_custkey,
    const std::vector<int32_t>& customer_nation,
    const PackedColumn& p_orderkey,
    const PackedColumn& p_quantity,
    const PackedColumn& p_extendedprice,
    DevicePackedBuffer& d_orderkey,
    DevicePackedBuffer& d_quantity,
    DevicePackedBuffer& d_extendedprice,
    const int32_t* d_order_custkey,
    const int32_t* d_customer_nation,
    unsigned long long* d_sum,
    uint64_t n_tiles,
    int64_t reference)
{
    uint64_t cpu_tiles = (n_tiles * static_cast<uint64_t>(cpu_percent)) / 100ULL;
    uint64_t gpu_start_tile = cpu_tiles;
    uint64_t gpu_end_tile = n_tiles;

    uint64_t block_begin = gpu_start_tile * ITEMS_PER_THREAD;
    uint64_t block_end = std::min<uint64_t>(p_orderkey.nblocks, gpu_end_tile * ITEMS_PER_THREAD);

    PackedSlice s_orderkey = make_slice(p_orderkey, block_begin, block_end);
    PackedSlice s_quantity = make_slice(p_quantity, block_begin, block_end);
    PackedSlice s_extendedprice = make_slice(p_extendedprice, block_begin, block_end);

    // Allocate once for this split. H2D copy still happens inside the timed GPU region.
    if (gpu_percent > 0 && s_orderkey.nblocks > 0) {
        allocate_device_buffer(d_orderkey, s_orderkey.block_offsets.size(), s_orderkey.data.size());
        allocate_device_buffer(d_quantity, s_quantity.block_offsets.size(), s_quantity.data.size());
        allocate_device_buffer(d_extendedprice, s_extendedprice.block_offsets.size(), s_extendedprice.data.size());
    }

    for (int i = 0; i < WARMUP_RUNS; ++i) {
        RunResult warm = run_split_once_common(
            cpu_percent, gpu_percent,
            orderkey, quantity, extendedprice,
            order_custkey, customer_nation,
            s_orderkey, s_quantity, s_extendedprice,
            d_orderkey, d_quantity, d_extendedprice,
            d_order_custkey, d_customer_nation,
            d_sum, n_tiles, reference);

        if (!warm.match) {
            std::cerr << "Warmup mismatch at split "
                      << cpu_percent << "/" << gpu_percent
                      << ": got " << warm.result
                      << ", expected " << reference << std::endl;
            std::exit(1);
        }
    }

    RunResult avg;
    avg.cpu_percent = cpu_percent;
    avg.gpu_percent = gpu_percent;
    avg.match = true;

    for (int i = 0; i < TIMED_RUNS; ++i) {
        RunResult r = run_split_once_common(
            cpu_percent, gpu_percent,
            orderkey, quantity, extendedprice,
            order_custkey, customer_nation,
            s_orderkey, s_quantity, s_extendedprice,
            d_orderkey, d_quantity, d_extendedprice,
            d_order_custkey, d_customer_nation,
            d_sum, n_tiles, reference);

        avg.cpu_ms += r.cpu_ms;
        avg.gpu_ms += r.gpu_ms;
        avg.e2e_ms += r.e2e_ms;
        avg.eff_gib_s += r.eff_gib_s;
        avg.result = r.result;
        avg.match = avg.match && r.match;
    }

    avg.cpu_ms /= TIMED_RUNS;
    avg.gpu_ms /= TIMED_RUNS;
    avg.e2e_ms /= TIMED_RUNS;
    avg.eff_gib_s /= TIMED_RUNS;

    return avg;
}

void write_csv(
    const std::string& path,
    const std::vector<RunResult>& rows,
    double input_mib,
    double compressed_mib,
    double compression_reduction)
{
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());

    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Cannot write CSV: " + path);
    }

    out << "system,cpu_percent,gpu_percent,input_mib,compressed_mib,"
        << "compression_reduction_percent,cpu_ms,gpu_ms,e2e_ms,eff_gib_s,match\n";

    for (const auto& r : rows) {
        out << "DPF_FUSED_BINPACK_COMMON_TIMING,"
            << r.cpu_percent << ","
            << r.gpu_percent << ","
            << std::fixed << std::setprecision(3) << input_mib << ","
            << std::fixed << std::setprecision(3) << compressed_mib << ","
            << std::fixed << std::setprecision(3) << compression_reduction << ","
            << std::fixed << std::setprecision(3) << r.cpu_ms << ","
            << std::fixed << std::setprecision(3) << r.gpu_ms << ","
            << std::fixed << std::setprecision(3) << r.e2e_ms << ","
            << std::fixed << std::setprecision(3) << r.eff_gib_s << ","
            << (r.match ? "YES" : "NO") << "\n";
    }
}

int main() {
    try {
        std::cout << "DPF-inspired fused SPJA x40 COMMON-TIMING benchmark\n";
        std::cout << "Encoding: DPF-style binpack/PFOR, block_size=128, miniblocks=4\n";
        std::cout << "Query: quantity > 25 AND customer_nation = 3, SUM(extendedprice)\n";
        std::cout << "GPU timing includes: H2D encoded GPU-owned slice + fused decode/SPJA kernel + D2H result\n";
        std::cout << "Packing/preprocessing time is excluded.\n";
        std::cout << "Lookup arrays are copied once and kept resident on GPU.\n\n";

        const std::string base = "data/tpch_columnar/";

        const std::string orderkey_path = base + "orderkey_sfx40.bin";
        const std::string quantity_path = base + "quantity_sfx40.bin";
        const std::string extendedprice_path = base + "extendedprice_sfx40.bin";
        const std::string order_custkey_path = base + "order_custkey_sfx40.bin";
        const std::string customer_nation_path = base + "customer_nation_sfx40.bin";

        std::cout << "Reading columns...\n";
        auto orderkey = read_binary_vector<int32_t>(orderkey_path);
        auto quantity = read_binary_vector<int32_t>(quantity_path);
        auto extendedprice = read_binary_vector<int32_t>(extendedprice_path);
        auto order_custkey = read_binary_vector<int32_t>(order_custkey_path);
        auto customer_nation = read_binary_vector<int32_t>(customer_nation_path);

        if (orderkey.size() != quantity.size() || orderkey.size() != extendedprice.size()) {
            throw std::runtime_error("Fact column row counts do not match.");
        }

        const uint64_t n_rows = orderkey.size();
        const uint64_t n_tiles = (n_rows + TILE_ROWS - 1) / TILE_ROWS;

        double input_mib = (static_cast<double>(n_rows) * 3.0 * sizeof(int32_t))
                         / (1024.0 * 1024.0);

        std::cout << "Rows: " << n_rows << "\n";
        std::cout << "Tiles: " << n_tiles << " (" << TILE_ROWS << " rows/tile)\n";
        std::cout << "Query input MiB: " << std::fixed << std::setprecision(3)
                  << input_mib << "\n\n";

        std::cout << "Packing fact columns with DPF-style binpack...\n";
        auto p_orderkey = pack_dpf_u32_from_i32(orderkey);
        auto p_quantity = pack_dpf_u32_from_i32(quantity);
        auto p_extendedprice = pack_dpf_u32_from_i32(extendedprice);

        if (p_orderkey.nblocks != p_quantity.nblocks || p_orderkey.nblocks != p_extendedprice.nblocks) {
            throw std::runtime_error("Packed column block counts do not match.");
        }

        uint64_t compressed_bytes =
            p_orderkey.bytes() + p_quantity.bytes() + p_extendedprice.bytes();

        double compressed_mib = static_cast<double>(compressed_bytes) / (1024.0 * 1024.0);
        double compression_reduction = 100.0 * (1.0 - (compressed_mib / input_mib));

        std::cout << "Compression statistics, fact columns only:\n";
        std::cout << "  Original MiB:   " << std::fixed << std::setprecision(3)
                  << input_mib << "\n";
        std::cout << "  Compressed MiB: " << std::fixed << std::setprecision(3)
                  << compressed_mib << "\n";
        std::cout << "  Reduction %:    " << std::fixed << std::setprecision(3)
                  << compression_reduction << "\n";
        std::cout << "  Note: lookup arrays are kept uncompressed/resident.\n\n";

        std::cout << "Computing CPU reference...\n";
        int64_t reference = cpu_spja_range_raw(orderkey, quantity, extendedprice,
                                               order_custkey, customer_nation,
                                               0, n_rows);
        std::cout << "Reference result: " << reference << "\n\n";

        std::cout << "Copying lookup arrays to GPU once...\n";
        int32_t* d_order_custkey = nullptr;
        int32_t* d_customer_nation = nullptr;
        unsigned long long* d_sum = nullptr;

        CUDA_CHECK(cudaMalloc(&d_order_custkey, order_custkey.size() * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_customer_nation, customer_nation.size() * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_sum, sizeof(unsigned long long)));

        CUDA_CHECK(cudaMemcpy(d_order_custkey, order_custkey.data(),
                              order_custkey.size() * sizeof(int32_t),
                              cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemcpy(d_customer_nation, customer_nation.data(),
                              customer_nation.size() * sizeof(int32_t),
                              cudaMemcpyHostToDevice));

        DevicePackedBuffer d_orderkey;
        DevicePackedBuffer d_quantity;
        DevicePackedBuffer d_extendedprice;

        std::vector<std::pair<int,int>> splits = {
            {100, 0},
            {75, 25},
            {50, 50},
            {25, 75},
            {0, 100}
        };

        std::vector<RunResult> results;

        std::cout << "Running CPU/GPU split experiments...\n";
        std::cout << "Warmup runs per split: " << WARMUP_RUNS << "\n";
        std::cout << "Timed runs per split: " << TIMED_RUNS << "\n";
        std::cout << "IMPORTANT: GPU ms includes H2D encoded slice + fused kernel + D2H result.\n\n";

        std::cout << std::left
                  << std::setw(12) << "Split"
                  << std::right
                  << std::setw(12) << "CPU ms"
                  << std::setw(12) << "GPU ms"
                  << std::setw(12) << "E2E ms"
                  << std::setw(14) << "Eff GiB/s"
                  << std::setw(10) << "MATCH"
                  << "\n";

        for (auto [cpu_pct, gpu_pct] : splits) {
            RunResult r = run_split_avg_common(
                cpu_pct, gpu_pct,
                orderkey, quantity, extendedprice,
                order_custkey, customer_nation,
                p_orderkey, p_quantity, p_extendedprice,
                d_orderkey, d_quantity, d_extendedprice,
                d_order_custkey, d_customer_nation,
                d_sum, n_tiles, reference);

            results.push_back(r);

            std::string split = std::to_string(cpu_pct) + "/" + std::to_string(gpu_pct);

            std::cout << std::left
                      << std::setw(12) << split
                      << std::right
                      << std::setw(12) << std::fixed << std::setprecision(3) << r.cpu_ms
                      << std::setw(12) << std::fixed << std::setprecision(3) << r.gpu_ms
                      << std::setw(12) << std::fixed << std::setprecision(3) << r.e2e_ms
                      << std::setw(14) << std::fixed << std::setprecision(3) << r.eff_gib_s
                      << std::setw(10) << (r.match ? "YES" : "NO")
                      << "\n";
        }

        bool all_match = true;
        for (const auto& r : results) all_match = all_match && r.match;

        std::string csv_path =
            "results/fastlanes_lz4nvcomp_dowda/csv/dpf_fused_spja_x40_common_timing_results.csv";

        write_csv(csv_path, results, input_mib, compressed_mib, compression_reduction);

        std::cout << "\nOverall correctness: " << (all_match ? "MATCH YES" : "MATCH NO") << "\n";
        std::cout << "CSV written: " << csv_path << "\n";

        free_device_buffer(d_orderkey);
        free_device_buffer(d_quantity);
        free_device_buffer(d_extendedprice);

        CUDA_CHECK(cudaFree(d_order_custkey));
        CUDA_CHECK(cudaFree(d_customer_nation));
        CUDA_CHECK(cudaFree(d_sum));

        return all_match ? 0 : 1;

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}


// nvcc -O3 -std=c++17 -arch=sm_86 \
//   src/benchmark/spja_dpf_fused_x40.cu \
//   -o bin/spja_dpf_fused_x40

// // ./bin/spja_dpf_fused_x40