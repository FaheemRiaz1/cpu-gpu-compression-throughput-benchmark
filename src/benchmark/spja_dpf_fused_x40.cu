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


// Fail immediately on CUDA runtime errors and report the source location.
#define CUDA_CHECK(call) do {                                      \
    cudaError_t err__ = (call);                                    \
    if (err__ != cudaSuccess) {                                    \
        std::cerr << "CUDA error: " << cudaGetErrorString(err__)   \
                  << " at " << __FILE__ << ":" << __LINE__         \
                  << std::endl;                                    \
        std::exit(1);                                              \
    }                                                              \
} while (0)


// DPF/binpack layout and benchmark configuration.
static constexpr int BLOCK_THREADS = 128;
static constexpr int ITEMS_PER_THREAD = 4;
static constexpr int BLOCK_SIZE = 128;
static constexpr int MINIBLOCK_COUNT = 4;

// Four encoded blocks are grouped into one 512-row processing tile.
static constexpr int TILE_ROWS =
    BLOCK_THREADS * ITEMS_PER_THREAD;

// Nation code selected by the SPJA predicate.
static constexpr int TARGET_NATION = 3;

static constexpr int WARMUP_RUNS = 1;
static constexpr int TIMED_RUNS = 5;


// Complete DPF-style packed representation of one fact-table column.
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


// Encoded subset corresponding to the block range assigned to the GPU.
// Offsets are local to the beginning of this slice.
struct PackedSlice {
    std::vector<uint32_t> block_offsets;
    std::vector<uint32_t> data;

    uint64_t global_block_begin = 0;
    uint64_t nblocks = 0;

    uint64_t bytes() const {
        return data.size() * sizeof(uint32_t)
             + block_offsets.size() * sizeof(uint32_t);
    }
};


// Reusable device storage for one packed column slice.
struct DevicePackedBuffer {
    uint32_t* d_offsets = nullptr;
    uint32_t* d_data = nullptr;

    uint64_t offsets_capacity = 0;
    uint64_t data_capacity = 0;
};


// Measurements and correctness information for one CPU/GPU split.
struct RunResult {
    int cpu_percent = 0;
    int gpu_percent = 0;

    double cpu_ms = 0.0;

    // Common GPU timing includes encoded H2D transfer,
    // fused decode/query execution, and scalar D2H result transfer.
    double gpu_ms = 0.0;

    double e2e_ms = 0.0;
    double eff_gib_s = 0.0;

    int64_t result = 0;
    bool match = false;
};


// Read one binary column directly into a typed host vector.
template <typename T>
std::vector<T> read_binary_vector(const std::string& path) {

    std::ifstream in(
        path,
        std::ios::binary | std::ios::ate
    );

    if (!in) {
        throw std::runtime_error(
            "Cannot open file: " + path
        );
    }


    std::streamsize bytes =
        in.tellg();


    if (
        bytes < 0 ||
        bytes %
            static_cast<std::streamsize>(
                sizeof(T)
            ) != 0
    ) {
        throw std::runtime_error(
            "Bad binary size for file: " + path
        );
    }


    in.seekg(
        0,
        std::ios::beg
    );


    std::vector<T> data(
        static_cast<size_t>(
            bytes / sizeof(T)
        )
    );


    if (!data.empty()) {

        in.read(
            reinterpret_cast<char*>(
                data.data()
            ),
            bytes
        );
    }


    return data;
}


// Return the number of bits required to represent an unsigned value.
static inline uint32_t bit_width_u32(uint32_t x) {

    if (x == 0) {
        return 0;
    }


    return 32u -
        static_cast<uint32_t>(
            __builtin_clz(x)
        );
}


// Encode one int32 fact column using the DPF-inspired block layout.
//
// Each 128-value block uses its minimum value as a frame-of-reference.
// Differences are bit-packed in four 32-value miniblocks.
PackedColumn pack_dpf_u32_from_i32(
    const std::vector<int32_t>& input
) {

    if (input.empty()) {

        throw std::runtime_error(
            "Cannot pack empty column."
        );
    }


    PackedColumn out;

    out.original_rows =
        input.size();

    out.nblocks =
        (
            input.size() +
            BLOCK_SIZE -
            1
        ) /
        BLOCK_SIZE;

    out.padded_rows =
        out.nblocks *
        BLOCK_SIZE;


    // Complete the final partial block so packing always operates
    // on exactly BLOCK_SIZE values.
    std::vector<uint32_t> padded(
        static_cast<size_t>(
            out.padded_rows
        )
    );


    for (size_t i = 0;
         i < input.size();
         ++i) {

        padded[i] =
            static_cast<uint32_t>(
                input[i]
            );
    }


    // Padding rows are not queried, but their encoded values must remain valid.
    uint32_t last =
        static_cast<uint32_t>(
            input.back()
        );


    for (size_t i = input.size();
         i < padded.size();
         ++i) {

        padded[i] = last;
    }


    out.block_offsets.resize(
        static_cast<size_t>(
            out.nblocks + 1
        )
    );


    // Small format header retained with the packed column.
    // Query kernels address blocks through block_offsets instead.
    out.data.push_back(
        BLOCK_SIZE
    );

    out.data.push_back(
        MINIBLOCK_COUNT
    );

    out.data.push_back(
        static_cast<uint32_t>(
            input.size()
        )
    );

    out.data.push_back(
        static_cast<uint32_t>(
            input[0]
        )
    );


    for (uint64_t b = 0;
         b < out.nblocks;
         ++b) {

        const uint64_t base =
            b * BLOCK_SIZE;


        out.block_offsets[
            static_cast<size_t>(b)
        ] =
            static_cast<uint32_t>(
                out.data.size()
            );


        // Use the block minimum as the frame-of-reference value.
        uint32_t min_val =
            padded[
                static_cast<size_t>(
                    base
                )
            ];


        for (int i = 1;
             i < BLOCK_SIZE;
             ++i) {

            min_val =
                std::min(
                    min_val,
                    padded[
                        static_cast<size_t>(
                            base + i
                        )
                    ]
                );
        }


        uint32_t diffs[BLOCK_SIZE];
        uint32_t max_diff = 0;


        for (int i = 0;
             i < BLOCK_SIZE;
             ++i) {

            diffs[i] =
                padded[
                    static_cast<size_t>(
                        base + i
                    )
                ] -
                min_val;


            max_diff =
                std::max(
                    max_diff,
                    diffs[i]
                );
        }


        uint32_t bw =
            bit_width_u32(
                max_diff
            );


        // Store the block reference.
        out.data.push_back(
            min_val
        );


        // All four miniblocks use the same bit width in this implementation.
        uint32_t bitwidth_word =
            bw |
            (bw << 8) |
            (bw << 16) |
            (bw << 24);


        out.data.push_back(
            bitwidth_word
        );


        // Each miniblock contains 32 values. A width of bw therefore
        // requires exactly bw 32-bit packed words per miniblock.
        for (int mb = 0;
             mb < MINIBLOCK_COUNT;
             ++mb) {

            if (bw == 0) {
                continue;
            }


            std::vector<uint32_t> words(
                bw,
                0
            );


            for (int j = 0;
                 j < 32;
                 ++j) {

                uint32_t v =
                    diffs[
                        mb * 32 + j
                    ];


                uint32_t bitpos =
                    static_cast<uint32_t>(j) *
                    bw;


                uint32_t word =
                    bitpos >> 5;


                uint32_t shift =
                    bitpos & 31;


                words[word] |=
                    (v << shift);


                // Continue into the next packed word when a value
                // crosses a 32-bit boundary.
                if (shift + bw > 32) {

                    words[word + 1] |=
                        (
                            v >>
                            (32 - shift)
                        );
                }
            }


            out.data.insert(
                out.data.end(),
                words.begin(),
                words.end()
            );
        }
    }


    out.block_offsets[
        static_cast<size_t>(
            out.nblocks
        )
    ] =
        static_cast<uint32_t>(
            out.data.size()
        );


    // The decoder reads two adjacent packed words as one 64-bit value.
    // Keep one padding word available after the final block.
    out.data.push_back(0);


    return out;
}


// Extract only the encoded block interval assigned to the GPU.
// This keeps H2D traffic proportional to the GPU-owned work.
PackedSlice make_slice(
    const PackedColumn& full,
    uint64_t block_begin,
    uint64_t block_end
) {

    if (
        block_begin > block_end ||
        block_end > full.nblocks
    ) {

        throw std::runtime_error(
            "Invalid packed slice range."
        );
    }


    PackedSlice s;

    s.global_block_begin =
        block_begin;

    s.nblocks =
        block_end -
        block_begin;


    if (s.nblocks == 0) {
        return s;
    }


    uint32_t data_begin =
        full.block_offsets[
            static_cast<size_t>(
                block_begin
            )
        ];


    uint32_t data_end =
        full.block_offsets[
            static_cast<size_t>(
                block_end
            )
        ];


    s.data.assign(
        full.data.begin() +
            data_begin,

        full.data.begin() +
            data_end
    );


    // Preserve the safe two-word read at the end of this local slice.
    s.data.push_back(0);


    s.block_offsets.resize(
        static_cast<size_t>(
            s.nblocks + 1
        )
    );


    // Convert global packed-data offsets into slice-local offsets.
    for (uint64_t i = 0;
         i <= s.nblocks;
         ++i) {

        s.block_offsets[
            static_cast<size_t>(i)
        ] =
            full.block_offsets[
                static_cast<size_t>(
                    block_begin + i
                )
            ] -
            data_begin;
    }


    return s;
}


// Grow reusable device buffers only when the current slice exceeds
// the capacity already allocated for a previous split.
void allocate_device_buffer(
    DevicePackedBuffer& d,
    uint64_t offsets_count,
    uint64_t data_count
) {

    if (
        offsets_count >
        d.offsets_capacity
    ) {

        if (d.d_offsets) {
            CUDA_CHECK(
                cudaFree(
                    d.d_offsets
                )
            );
        }


        CUDA_CHECK(
            cudaMalloc(
                &d.d_offsets,
                offsets_count *
                    sizeof(uint32_t)
            )
        );


        d.offsets_capacity =
            offsets_count;
    }


    if (
        data_count >
        d.data_capacity
    ) {

        if (d.d_data) {
            CUDA_CHECK(
                cudaFree(
                    d.d_data
                )
            );
        }


        CUDA_CHECK(
            cudaMalloc(
                &d.d_data,
                data_count *
                    sizeof(uint32_t)
            )
        );


        d.data_capacity =
            data_count;
    }
}


// Release one reusable packed-column device buffer.
void free_device_buffer(
    DevicePackedBuffer& d
) {

    if (d.d_offsets) {

        CUDA_CHECK(
            cudaFree(
                d.d_offsets
            )
        );
    }


    if (d.d_data) {

        CUDA_CHECK(
            cudaFree(
                d.d_data
            )
        );
    }


    d = {};
}


// Transfer one GPU-owned encoded slice.
// Calls to this function remain inside the measured GPU query region.
void copy_slice_to_device(
    const PackedSlice& h,
    DevicePackedBuffer& d
) {

    if (h.nblocks == 0) {
        return;
    }


    allocate_device_buffer(
        d,
        h.block_offsets.size(),
        h.data.size()
    );


    CUDA_CHECK(
        cudaMemcpy(
            d.d_offsets,
            h.block_offsets.data(),
            h.block_offsets.size() *
                sizeof(uint32_t),
            cudaMemcpyHostToDevice
        )
    );


    CUDA_CHECK(
        cudaMemcpy(
            d.d_data,
            h.data.data(),
            h.data.size() *
                sizeof(uint32_t),
            cudaMemcpyHostToDevice
        )
    );
}


// Decode one value directly from the packed column without
// materializing a full decompressed intermediate column.
__device__ __forceinline__
uint32_t dpf_decode_u32(
    const uint32_t* __restrict__ block_offsets,
    const uint32_t* __restrict__ data,
    uint32_t local_block_idx,
    uint32_t index_in_block
) {

    const uint32_t* block =
        data +
        block_offsets[
            local_block_idx
        ];


    uint32_t reference =
        block[0];


    uint32_t bitwidths =
        block[1];


    uint32_t miniblock_index =
        index_in_block >> 5;


    uint32_t index_into_miniblock =
        index_in_block & 31;


    uint32_t bitwidth =
        (
            bitwidths >>
            (
                miniblock_index <<
                3
            )
        ) &
        255u;


    if (bitwidth == 0) {
        return reference;
    }


    // Prefix offsets for the four equal-width miniblocks:
    // [0, bw, 2*bw, 3*bw].
    uint32_t miniblock_offsets =
        (bitwidths << 8) +
        (bitwidths << 16) +
        (bitwidths << 24);


    uint32_t miniblock_offset =
        (
            miniblock_offsets >>
            (
                miniblock_index <<
                3
            )
        ) &
        255u;


    uint32_t start_bit =
        bitwidth *
        index_into_miniblock;


    uint32_t start_word =
        2 +
        miniblock_offset +
        (start_bit >> 5);


    uint32_t shift =
        start_bit & 31;


    // Combine two adjacent packed words so values that cross a
    // 32-bit boundary can be decoded with the same operation.
    uint64_t two_words =
        (
            static_cast<uint64_t>(
                block[
                    start_word + 1
                ]
            ) <<
            32
        )
        |
        static_cast<uint64_t>(
            block[
                start_word
            ]
        );


    uint32_t mask =
        (bitwidth == 32)
            ? 0xffffffffu
            : (
                (1u << bitwidth) -
                1u
            );


    uint32_t element =
        static_cast<uint32_t>(
            (
                two_words >>
                shift
            ) &
            mask
        );


    return
        reference +
        element;
}


// Fused decode and SPJA kernel.
//
// Processing order avoids unnecessary decoding:
//   1. Decode quantity and apply quantity > 25.
//   2. Decode orderkey and perform the order-to-customer lookup.
//   3. Apply the target-nation predicate.
//   4. Decode extendedprice only for qualifying rows.
//   5. Aggregate directly on the GPU.
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

    unsigned long long* __restrict__ out_sum
) {

    __shared__
        unsigned long long
        block_sums[BLOCK_THREADS];


    const uint32_t tid =
        threadIdx.x;


    const uint64_t n_local_tiles =
        (
            nblocks_slice +
            ITEMS_PER_THREAD -
            1
        ) /
        ITEMS_PER_THREAD;


    // Grid-stride traversal allows a fixed launch configuration to
    // process all tiles in the GPU-owned slice.
    for (uint64_t tile = blockIdx.x;
         tile < n_local_tiles;
         tile += gridDim.x) {

        unsigned long long local =
            0;


        uint64_t first_local_block =
            tile *
            ITEMS_PER_THREAD;


        #pragma unroll
        for (int item = 0;
             item < ITEMS_PER_THREAD;
             ++item) {

            uint64_t local_block =
                first_local_block +
                item;


            if (
                local_block >=
                nblocks_slice
            ) {
                continue;
            }


            uint64_t global_block =
                global_block_begin +
                local_block;


            uint64_t row =
                global_block *
                BLOCK_SIZE +
                tid;


            if (row >= n_rows) {
                continue;
            }


            uint32_t lb =
                static_cast<uint32_t>(
                    local_block
                );


            // Apply the quantity predicate before decoding join keys
            // or extendedprice for this row.
            int32_t q =
                static_cast<int32_t>(
                    dpf_decode_u32(
                        quantity_offsets,
                        quantity_data,
                        lb,
                        tid
                    )
                );


            if (q <= 25) {
                continue;
            }


            int32_t ok =
                static_cast<int32_t>(
                    dpf_decode_u32(
                        orderkey_offsets,
                        orderkey_data,
                        lb,
                        tid
                    )
                );


            if (
                ok < 0 ||
                static_cast<uint64_t>(ok) >=
                    order_custkey_n
            ) {
                continue;
            }


            int32_t custkey =
                order_custkey[ok];


            if (
                custkey < 0 ||
                static_cast<uint64_t>(custkey) >=
                    customer_nation_n
            ) {
                continue;
            }


            if (
                customer_nation[
                    custkey
                ] !=
                TARGET_NATION
            ) {
                continue;
            }


            // Price decoding is delayed until the row has passed
            // both predicates and the lookup join.
            int32_t ep =
                static_cast<int32_t>(
                    dpf_decode_u32(
                        extendedprice_offsets,
                        extendedprice_data,
                        lb,
                        tid
                    )
                );


            local +=
                static_cast<unsigned long long>(
                    static_cast<int64_t>(
                        ep
                    )
                );
        }


        // Reduce thread-local aggregates within the CUDA block.
        block_sums[tid] =
            local;


        __syncthreads();


        for (uint32_t stride =
                 BLOCK_THREADS / 2;
             stride > 0;
             stride >>= 1) {

            if (tid < stride) {

                block_sums[tid] +=
                    block_sums[
                        tid + stride
                    ];
            }


            __syncthreads();
        }


        if (
            tid == 0 &&
            block_sums[0] != 0
        ) {

            atomicAdd(
                out_sum,
                block_sums[0]
            );
        }


        __syncthreads();
    }
}


// Scalar CPU implementation of the same SPJA query.
// It is used for the independent reference and by CPU worker threads.
int64_t cpu_spja_range_raw(
    const std::vector<int32_t>& orderkey,
    const std::vector<int32_t>& quantity,
    const std::vector<int32_t>& extendedprice,

    const std::vector<int32_t>& order_custkey,
    const std::vector<int32_t>& customer_nation,

    uint64_t begin,
    uint64_t end
) {

    int64_t sum =
        0;


    end =
        std::min<uint64_t>(
            end,
            orderkey.size()
        );


    for (uint64_t i = begin;
         i < end;
         ++i) {

        if (quantity[i] <= 25) {
            continue;
        }


        int32_t ok =
            orderkey[i];


        if (
            ok < 0 ||
            static_cast<uint64_t>(ok) >=
                order_custkey.size()
        ) {
            continue;
        }


        int32_t custkey =
            order_custkey[ok];


        if (
            custkey < 0 ||
            static_cast<uint64_t>(custkey) >=
                customer_nation.size()
        ) {
            continue;
        }


        if (
            customer_nation[
                custkey
            ] !=
            TARGET_NATION
        ) {
            continue;
        }


        sum +=
            static_cast<int64_t>(
                extendedprice[i]
            );
    }


    return sum;
}


// Execute the CPU-owned row interval in parallel using up to 36
// worker threads, bounded by hardware concurrency and row count.
int64_t cpu_spja_range_parallel(
    const std::vector<int32_t>& orderkey,
    const std::vector<int32_t>& quantity,
    const std::vector<int32_t>& extendedprice,

    const std::vector<int32_t>& order_custkey,
    const std::vector<int32_t>& customer_nation,

    uint64_t begin,
    uint64_t end
) {

    end =
        std::min<uint64_t>(
            end,
            orderkey.size()
        );


    if (begin >= end) {
        return 0;
    }


    const uint64_t total_rows =
        end -
        begin;


    const unsigned int detected_cpu_threads =
        std::thread::hardware_concurrency();


    const int requested_cpu_threads =
        (detected_cpu_threads > 0)
            ? static_cast<int>(
                std::min(
                    36u,
                    detected_cpu_threads
                )
              )
            : 36;


    const int actual_threads =
        std::max(
            1,
            std::min(
                requested_cpu_threads,
                static_cast<int>(
                    total_rows
                )
            )
        );


    std::vector<std::thread>
        workers;


    std::vector<int64_t>
        partial_sums(
            static_cast<size_t>(
                actual_threads
            ),
            0
        );


    workers.reserve(
        static_cast<size_t>(
            actual_threads
        )
    );


    // Divide the CPU-owned row range evenly across workers.
    for (int t = 0;
         t < actual_threads;
         ++t) {

        const uint64_t local_begin =
            begin +
            (
                total_rows *
                static_cast<uint64_t>(t)
            ) /
            static_cast<uint64_t>(
                actual_threads
            );


        const uint64_t local_end =
            begin +
            (
                total_rows *
                static_cast<uint64_t>(
                    t + 1
                )
            ) /
            static_cast<uint64_t>(
                actual_threads
            );


        workers.emplace_back(
            [&, t, local_begin, local_end]() {

                partial_sums[
                    static_cast<size_t>(t)
                ] =
                    cpu_spja_range_raw(
                        orderkey,
                        quantity,
                        extendedprice,

                        order_custkey,
                        customer_nation,

                        local_begin,
                        local_end
                    );
            }
        );
    }


    for (auto& worker :
         workers) {

        worker.join();
    }


    int64_t total_sum =
        0;


    for (int64_t partial :
         partial_sums) {

        total_sum +=
            partial;
    }


    return total_sum;
}


// Execute one CPU/GPU split.
//
// CPU processing runs in a separate host thread while the calling thread
// drives the GPU path. E2E timing therefore captures their overlap rather
// than simply adding CPU and GPU execution times.
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
    int64_t reference
) {

    RunResult rr;

    rr.cpu_percent =
        cpu_percent;

    rr.gpu_percent =
        gpu_percent;


    uint64_t cpu_tiles =
        (
            n_tiles *
            static_cast<uint64_t>(
                cpu_percent
            )
        ) /
        100ULL;


    uint64_t gpu_start_tile =
        cpu_tiles;


    uint64_t gpu_end_tile =
        n_tiles;


    uint64_t cpu_begin_row =
        0;


    uint64_t cpu_end_row =
        std::min<uint64_t>(
            orderkey.size(),
            cpu_tiles *
                TILE_ROWS
        );


    int64_t cpu_sum =
        0;


    unsigned long long gpu_sum =
        0;


    double cpu_ms =
        0.0;


    double gpu_ms =
        0.0;


    auto e2e_start =
        std::chrono::high_resolution_clock::now();


    std::thread cpu_thread;


    // Start CPU work separately so CPU and GPU portions can execute concurrently.
    if (cpu_percent > 0) {

        cpu_thread =
            std::thread(
                [&]() {

                    auto t0 =
                        std::chrono::high_resolution_clock::now();


                    cpu_sum =
                        cpu_spja_range_parallel(
                            orderkey,
                            quantity,
                            extendedprice,

                            order_custkey,
                            customer_nation,

                            cpu_begin_row,
                            cpu_end_row
                        );


                    auto t1 =
                        std::chrono::high_resolution_clock::now();


                    cpu_ms =
                        std::chrono::duration<
                            double,
                            std::milli
                        >(
                            t1 - t0
                        ).count();
                }
            );
    }


    if (
        gpu_percent > 0 &&
        gpu_start_tile <
            gpu_end_tile
    ) {

        auto tg0 =
            std::chrono::high_resolution_clock::now();


        // Common GPU timing begins before transferring the encoded
        // fact-column slices assigned to the GPU.
        copy_slice_to_device(
            s_orderkey,
            d_orderkey
        );


        copy_slice_to_device(
            s_quantity,
            d_quantity
        );


        copy_slice_to_device(
            s_extendedprice,
            d_extendedprice
        );


        // Aggregate initialization is part of the measured GPU path.
        CUDA_CHECK(
            cudaMemset(
                d_sum,
                0,
                sizeof(
                    unsigned long long
                )
            )
        );


        uint64_t nblocks_slice =
            s_orderkey.nblocks;


        uint64_t global_block_begin =
            s_orderkey.global_block_begin;


        // All fact-column slices must describe the same block interval.
        if (
            s_quantity.nblocks !=
                nblocks_slice ||

            s_extendedprice.nblocks !=
                nblocks_slice ||

            s_quantity.global_block_begin !=
                global_block_begin ||

            s_extendedprice.global_block_begin !=
                global_block_begin
        ) {

            throw std::runtime_error(
                "Packed slices do not match."
            );
        }


        uint64_t n_local_tiles =
            (
                nblocks_slice +
                ITEMS_PER_THREAD -
                1
            ) /
            ITEMS_PER_THREAD;


        int sm_count =
            0;


        CUDA_CHECK(
            cudaDeviceGetAttribute(
                &sm_count,
                cudaDevAttrMultiProcessorCount,
                0
            )
        );


        // Cap the grid at eight blocks per SM while ensuring that
        // no more blocks are launched than there are local tiles.
        int grid =
            static_cast<int>(
                std::min<uint64_t>(
                    n_local_tiles,
                    static_cast<uint64_t>(
                        sm_count * 8
                    )
                )
            );


        grid =
            std::max(
                grid,
                1
            );


        dpf_fused_spja_kernel_common<<<
            grid,
            BLOCK_THREADS
        >>>(
            d_orderkey.d_offsets,
            d_orderkey.d_data,

            d_quantity.d_offsets,
            d_quantity.d_data,

            d_extendedprice.d_offsets,
            d_extendedprice.d_data,

            global_block_begin,
            nblocks_slice,

            d_order_custkey,
            order_custkey.size(),

            d_customer_nation,
            customer_nation.size(),

            orderkey.size(),

            d_sum
        );


        CUDA_CHECK(
            cudaGetLastError()
        );


        // Return only the final scalar aggregate. Its D2H transfer
        // remains part of common GPU timing.
        CUDA_CHECK(
            cudaMemcpy(
                &gpu_sum,
                d_sum,
                sizeof(
                    unsigned long long
                ),
                cudaMemcpyDeviceToHost
            )
        );


        CUDA_CHECK(
            cudaDeviceSynchronize()
        );


        auto tg1 =
            std::chrono::high_resolution_clock::now();


        gpu_ms =
            std::chrono::duration<
                double,
                std::milli
            >(
                tg1 - tg0
            ).count();
    }


    if (cpu_percent > 0) {

        cpu_thread.join();
    }


    auto e2e_stop =
        std::chrono::high_resolution_clock::now();


    rr.cpu_ms =
        cpu_ms;


    rr.gpu_ms =
        gpu_ms;


    rr.e2e_ms =
        std::chrono::duration<
            double,
            std::milli
        >(
            e2e_stop -
            e2e_start
        ).count();


    // CPU and GPU ranges are disjoint, so their partial aggregates
    // combine into the complete query result.
    rr.result =
        cpu_sum +
        static_cast<int64_t>(
            gpu_sum
        );


    rr.match =
        (
            rr.result ==
            reference
        );


    // Effective throughput uses the logical size of the three int32
    // fact columns rather than the encoded physical byte count.
    double input_gib =
        (
            static_cast<double>(
                orderkey.size()
            ) *
            3.0 *
            sizeof(int32_t)
        ) /
        (
            1024.0 *
            1024.0 *
            1024.0
        );


    rr.eff_gib_s =
        input_gib /
        (
            rr.e2e_ms /
            1000.0
        );


    return rr;
}


// Prepare the encoded GPU slice for one split, perform the warm-up,
// and average the configured timed measurements.
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
    int64_t reference
) {

    uint64_t cpu_tiles =
        (
            n_tiles *
            static_cast<uint64_t>(
                cpu_percent
            )
        ) /
        100ULL;


    uint64_t gpu_start_tile =
        cpu_tiles;


    uint64_t gpu_end_tile =
        n_tiles;


    uint64_t block_begin =
        gpu_start_tile *
        ITEMS_PER_THREAD;


    uint64_t block_end =
        std::min<uint64_t>(
            p_orderkey.nblocks,
            gpu_end_tile *
                ITEMS_PER_THREAD
        );


    // Slice construction is preprocessing and remains outside
    // the per-run query timing.
    PackedSlice s_orderkey =
        make_slice(
            p_orderkey,
            block_begin,
            block_end
        );


    PackedSlice s_quantity =
        make_slice(
            p_quantity,
            block_begin,
            block_end
        );


    PackedSlice s_extendedprice =
        make_slice(
            p_extendedprice,
            block_begin,
            block_end
        );


    // Reserve sufficient GPU capacity once for the split.
    // The encoded H2D copies themselves remain inside timed execution.
    if (
        gpu_percent > 0 &&
        s_orderkey.nblocks > 0
    ) {

        allocate_device_buffer(
            d_orderkey,
            s_orderkey.block_offsets.size(),
            s_orderkey.data.size()
        );


        allocate_device_buffer(
            d_quantity,
            s_quantity.block_offsets.size(),
            s_quantity.data.size()
        );


        allocate_device_buffer(
            d_extendedprice,
            s_extendedprice.block_offsets.size(),
            s_extendedprice.data.size()
        );
    }


    // Warm-up results are checked for correctness but not included
    // in the reported averages.
    for (int i = 0;
         i < WARMUP_RUNS;
         ++i) {

        RunResult warm =
            run_split_once_common(
                cpu_percent,
                gpu_percent,

                orderkey,
                quantity,
                extendedprice,

                order_custkey,
                customer_nation,

                s_orderkey,
                s_quantity,
                s_extendedprice,

                d_orderkey,
                d_quantity,
                d_extendedprice,

                d_order_custkey,
                d_customer_nation,

                d_sum,
                n_tiles,
                reference
            );


        if (!warm.match) {

            std::cerr
                << "Warmup mismatch at split "
                << cpu_percent
                << "/"
                << gpu_percent
                << ": got "
                << warm.result
                << ", expected "
                << reference
                << std::endl;


            std::exit(1);
        }
    }


    RunResult avg;

    avg.cpu_percent =
        cpu_percent;

    avg.gpu_percent =
        gpu_percent;

    avg.match =
        true;


    for (int i = 0;
         i < TIMED_RUNS;
         ++i) {

        RunResult r =
            run_split_once_common(
                cpu_percent,
                gpu_percent,

                orderkey,
                quantity,
                extendedprice,

                order_custkey,
                customer_nation,

                s_orderkey,
                s_quantity,
                s_extendedprice,

                d_orderkey,
                d_quantity,
                d_extendedprice,

                d_order_custkey,
                d_customer_nation,

                d_sum,
                n_tiles,
                reference
            );


        avg.cpu_ms +=
            r.cpu_ms;


        avg.gpu_ms +=
            r.gpu_ms;


        avg.e2e_ms +=
            r.e2e_ms;


        avg.eff_gib_s +=
            r.eff_gib_s;


        avg.result =
            r.result;


        avg.match =
            avg.match &&
            r.match;
    }


    avg.cpu_ms /=
        TIMED_RUNS;


    avg.gpu_ms /=
        TIMED_RUNS;


    avg.e2e_ms /=
        TIMED_RUNS;


    avg.eff_gib_s /=
        TIMED_RUNS;


    return avg;
}


// Write averaged split timings, compression statistics, and correctness.
void write_csv(
    const std::string& path,
    const std::vector<RunResult>& rows,

    double input_mib,
    double compressed_mib,
    double compression_reduction
) {

    std::filesystem::create_directories(
        std::filesystem::path(
            path
        ).parent_path()
    );


    std::ofstream out(
        path
    );


    if (!out) {

        throw std::runtime_error(
            "Cannot write CSV: " +
            path
        );
    }


    out
        << "system,cpu_percent,gpu_percent,input_mib,compressed_mib,"
        << "compression_reduction_percent,cpu_ms,gpu_ms,e2e_ms,"
        << "eff_gib_s,match\n";


    for (const auto& r :
         rows) {

        out
            << "DPF_FUSED_BINPACK_COMMON_TIMING,"
            << r.cpu_percent
            << ","

            << r.gpu_percent
            << ","

            << std::fixed
            << std::setprecision(3)
            << input_mib
            << ","

            << std::fixed
            << std::setprecision(3)
            << compressed_mib
            << ","

            << std::fixed
            << std::setprecision(3)
            << compression_reduction
            << ","

            << std::fixed
            << std::setprecision(3)
            << r.cpu_ms
            << ","

            << std::fixed
            << std::setprecision(3)
            << r.gpu_ms
            << ","

            << std::fixed
            << std::setprecision(3)
            << r.e2e_ms
            << ","

            << std::fixed
            << std::setprecision(3)
            << r.eff_gib_s
            << ","

            << (
                r.match
                    ? "YES"
                    : "NO"
               )
            << "\n";
    }
}


int main() {

    try {

        std::cout
            << "DPF-inspired fused SPJA x40 COMMON-TIMING benchmark\n";

        std::cout
            << "Encoding: DPF-style binpack/PFOR, block_size=128, miniblocks=4\n";

        std::cout
            << "Query: quantity > 25 AND customer_nation = 3, SUM(extendedprice)\n";

        std::cout
            << "GPU timing includes: H2D encoded GPU-owned slice + fused decode/SPJA kernel + D2H result\n";

        std::cout
            << "Packing/preprocessing time is excluded.\n";

        std::cout
            << "Lookup arrays are copied once and kept resident on GPU.\n\n";


        // TPC-H-derived x40 columns used by this numeric SPJA workload.
        const std::string base =
            "data/tpch_columnar/";


        const std::string orderkey_path =
            base +
            "orderkey_sfx40.bin";


        const std::string quantity_path =
            base +
            "quantity_sfx40.bin";


        const std::string extendedprice_path =
            base +
            "extendedprice_sfx40.bin";


        const std::string order_custkey_path =
            base +
            "order_custkey_sfx40.bin";


        const std::string customer_nation_path =
            base +
            "customer_nation_sfx40.bin";


        std::cout
            << "Reading columns...\n";


        auto orderkey =
            read_binary_vector<int32_t>(
                orderkey_path
            );


        auto quantity =
            read_binary_vector<int32_t>(
                quantity_path
            );


        auto extendedprice =
            read_binary_vector<int32_t>(
                extendedprice_path
            );


        auto order_custkey =
            read_binary_vector<int32_t>(
                order_custkey_path
            );


        auto customer_nation =
            read_binary_vector<int32_t>(
                customer_nation_path
            );


        if (
            orderkey.size() !=
                quantity.size() ||

            orderkey.size() !=
                extendedprice.size()
        ) {

            throw std::runtime_error(
                "Fact column row counts do not match."
            );
        }


        const uint64_t n_rows =
            orderkey.size();


        const uint64_t n_tiles =
            (
                n_rows +
                TILE_ROWS -
                1
            ) /
            TILE_ROWS;


        // Logical query size covers the three int32 fact columns.
        double input_mib =
            (
                static_cast<double>(
                    n_rows
                ) *
                3.0 *
                sizeof(int32_t)
            ) /
            (
                1024.0 *
                1024.0
            );


        std::cout
            << "Rows: "
            << n_rows
            << "\n";


        std::cout
            << "Tiles: "
            << n_tiles
            << " ("
            << TILE_ROWS
            << " rows/tile)\n";


        std::cout
            << "Query input MiB: "
            << std::fixed
            << std::setprecision(3)
            << input_mib
            << "\n\n";


        // Pack all three fact columns before entering the timed experiments.
        std::cout
            << "Packing fact columns with DPF-style binpack...\n";


        auto p_orderkey =
            pack_dpf_u32_from_i32(
                orderkey
            );


        auto p_quantity =
            pack_dpf_u32_from_i32(
                quantity
            );


        auto p_extendedprice =
            pack_dpf_u32_from_i32(
                extendedprice
            );


        if (
            p_orderkey.nblocks !=
                p_quantity.nblocks ||

            p_orderkey.nblocks !=
                p_extendedprice.nblocks
        ) {

            throw std::runtime_error(
                "Packed column block counts do not match."
            );
        }


        // Compression statistics include only the three encoded
        // fact columns. Lookup arrays stay uncompressed.
        uint64_t compressed_bytes =
            p_orderkey.bytes() +
            p_quantity.bytes() +
            p_extendedprice.bytes();


        double compressed_mib =
            static_cast<double>(
                compressed_bytes
            ) /
            (
                1024.0 *
                1024.0
            );


        double compression_reduction =
            100.0 *
            (
                1.0 -
                (
                    compressed_mib /
                    input_mib
                )
            );


        std::cout
            << "Compression statistics, fact columns only:\n";


        std::cout
            << "  Original MiB:   "
            << std::fixed
            << std::setprecision(3)
            << input_mib
            << "\n";


        std::cout
            << "  Compressed MiB: "
            << std::fixed
            << std::setprecision(3)
            << compressed_mib
            << "\n";


        std::cout
            << "  Reduction %:    "
            << std::fixed
            << std::setprecision(3)
            << compression_reduction
            << "\n";


        std::cout
            << "  Note: lookup arrays are kept uncompressed/resident.\n\n";


        // Independent CPU execution provides the correctness reference
        // used by every CPU/GPU split.
        std::cout
            << "Computing CPU reference...\n";


        int64_t reference =
            cpu_spja_range_raw(
                orderkey,
                quantity,
                extendedprice,

                order_custkey,
                customer_nation,

                0,
                n_rows
            );


        std::cout
            << "Reference result: "
            << reference
            << "\n\n";


        // Lookup arrays are transferred once and remain resident because
        // their transfer is not part of per-query common timing.
        std::cout
            << "Copying lookup arrays to GPU once...\n";


        int32_t* d_order_custkey =
            nullptr;


        int32_t* d_customer_nation =
            nullptr;


        unsigned long long* d_sum =
            nullptr;


        CUDA_CHECK(
            cudaMalloc(
                &d_order_custkey,
                order_custkey.size() *
                    sizeof(int32_t)
            )
        );


        CUDA_CHECK(
            cudaMalloc(
                &d_customer_nation,
                customer_nation.size() *
                    sizeof(int32_t)
            )
        );


        CUDA_CHECK(
            cudaMalloc(
                &d_sum,
                sizeof(
                    unsigned long long
                )
            )
        );


        CUDA_CHECK(
            cudaMemcpy(
                d_order_custkey,
                order_custkey.data(),
                order_custkey.size() *
                    sizeof(int32_t),
                cudaMemcpyHostToDevice
            )
        );


        CUDA_CHECK(
            cudaMemcpy(
                d_customer_nation,
                customer_nation.data(),
                customer_nation.size() *
                    sizeof(int32_t),
                cudaMemcpyHostToDevice
            )
        );


        DevicePackedBuffer d_orderkey;
        DevicePackedBuffer d_quantity;
        DevicePackedBuffer d_extendedprice;


        // Standard split points used for CPU/GPU co-processing evaluation.
        std::vector<std::pair<int,int>> splits = {
            {100, 0},
            {75, 25},
            {50, 50},
            {25, 75},
            {0, 100}
        };


        std::vector<RunResult> results;


        std::cout
            << "Running CPU/GPU split experiments...\n";


        std::cout
            << "Warmup runs per split: "
            << WARMUP_RUNS
            << "\n";


        std::cout
            << "Timed runs per split: "
            << TIMED_RUNS
            << "\n";


        std::cout
            << "IMPORTANT: GPU ms includes H2D encoded slice + fused kernel + D2H result.\n\n";


        std::cout
            << std::left

            << std::setw(12)
            << "Split"

            << std::right

            << std::setw(12)
            << "CPU ms"

            << std::setw(12)
            << "GPU ms"

            << std::setw(12)
            << "E2E ms"

            << std::setw(14)
            << "Eff GiB/s"

            << std::setw(10)
            << "MATCH"

            << "\n";


        for (auto [cpu_pct, gpu_pct] :
             splits) {

            RunResult r =
                run_split_avg_common(
                    cpu_pct,
                    gpu_pct,

                    orderkey,
                    quantity,
                    extendedprice,

                    order_custkey,
                    customer_nation,

                    p_orderkey,
                    p_quantity,
                    p_extendedprice,

                    d_orderkey,
                    d_quantity,
                    d_extendedprice,

                    d_order_custkey,
                    d_customer_nation,

                    d_sum,
                    n_tiles,
                    reference
                );


            results.push_back(
                r
            );


            std::string split =
                std::to_string(
                    cpu_pct
                ) +
                "/" +
                std::to_string(
                    gpu_pct
                );


            std::cout
                << std::left

                << std::setw(12)
                << split

                << std::right

                << std::setw(12)
                << std::fixed
                << std::setprecision(3)
                << r.cpu_ms

                << std::setw(12)
                << std::fixed
                << std::setprecision(3)
                << r.gpu_ms

                << std::setw(12)
                << std::fixed
                << std::setprecision(3)
                << r.e2e_ms

                << std::setw(14)
                << std::fixed
                << std::setprecision(3)
                << r.eff_gib_s

                << std::setw(10)
                << (
                    r.match
                        ? "YES"
                        : "NO"
                   )

                << "\n";
        }


        bool all_match =
            true;


        for (const auto& r :
             results) {

            all_match =
                all_match &&
                r.match;
        }


        // Running the benchmark rewrites this file with the current
        // averaged measurements.
        std::string csv_path =
            "results/fastlanes_lz4nvcomp_dowda/csv/"
            "dpf_fused_spja_x40_common_timing_results.csv";


        write_csv(
            csv_path,
            results,
            input_mib,
            compressed_mib,
            compression_reduction
        );


        std::cout
            << "\nOverall correctness: "
            << (
                all_match
                    ? "MATCH YES"
                    : "MATCH NO"
               )
            << "\n";


        std::cout
            << "CSV written: "
            << csv_path
            << "\n";


        // Release reusable packed buffers and persistent lookup arrays.
        free_device_buffer(
            d_orderkey
        );


        free_device_buffer(
            d_quantity
        );


        free_device_buffer(
            d_extendedprice
        );


        CUDA_CHECK(
            cudaFree(
                d_order_custkey
            )
        );


        CUDA_CHECK(
            cudaFree(
                d_customer_nation
            )
        );


        CUDA_CHECK(
            cudaFree(
                d_sum
            )
        );


        return
            all_match
                ? 0
                : 1;
    }

    catch (const std::exception& e) {

        std::cerr
            << "ERROR: "
            << e.what()
            << std::endl;


        return 1;
    }
}


// Build:
//
// mkdir -p bin
//
// nvcc -O3 -std=c++17 -arch=sm_86 \
//     -Xcompiler -pthread \
//     src/benchmark/spja_dpf_fused_x40.cu \
//     -o bin/spja_dpf_fused_x40
//
// Run:
//
// ./bin/spja_dpf_fused_x40