#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

#include <cuda_runtime.h>
#include <lz4.h>
#include <nvcomp/lz4.h>

// Checks CUDA API calls and stops the program if a CUDA error occurs.
inline void cuda_check(cudaError_t e, const char* file, int line) {
    if (e != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(e)
                  << " at " << file << ":" << line << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// Checks nvCOMP API calls and stops the program if an nvCOMP error occurs.
inline void nvcomp_check(nvcompStatus_t s, const char* file, int line) {
    if (s != nvcompSuccess) {
        std::cerr << "nvCOMP error: status=" << static_cast<int>(s)
                  << " at " << file << ":" << line << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// Convenience macros to automatically report file name and line number.
#define CUDA_CHECK(x) cuda_check((x), __FILE__, __LINE__)
#define NVCOMP_CHECK(x) nvcomp_check((x), __FILE__, __LINE__)

// Different input data patterns used to test the effect of compressibility.
enum class DataMode {
    HIGH_COMPRESSIBLE,
    MEDIUM_COMPRESSIBLE,
    RANDOM_DATA
};

// Converts the data mode enum into a readable string for terminal and CSV output.
static const char* mode_to_string(DataMode mode) {
    switch (mode) {
        case DataMode::HIGH_COMPRESSIBLE:   return "HIGH";
        case DataMode::MEDIUM_COMPRESSIBLE: return "MEDIUM";
        case DataMode::RANDOM_DATA:         return "RANDOM";
        default:                            return "UNKNOWN";
    }
}

// Simple GPU compute kernel.
// Each CUDA thread multiplies one integer by 2.
__global__ void compute_kernel(int* data, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] *= 2;
    }
}

// Returns elapsed time between two chrono timestamps in milliseconds.
template <typename T1, typename T2>
double ms_between(const T1& a, const T2& b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// Converts bytes to GiB using 1024-based units.
static double bytes_to_gib(size_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
}

// Computes the arithmetic mean of a vector of values.
static double mean(const std::vector<double>& values) {
    if (values.empty()) return 0.0;
    double sum = 0.0;
    for (double v : values) sum += v;
    return sum / static_cast<double>(values.size());
}

// Computes sample standard deviation for repeated benchmark measurements.
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

// Fills the input array according to the selected compressibility mode.
static void fill_input(int* data, size_t n, DataMode mode) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 1000000);

    for (size_t i = 0; i < n; ++i) {
        switch (mode) {
            // Repeated small pattern, expected to compress very well.
            case DataMode::HIGH_COMPRESSIBLE:
                data[i] = 0;
                break;

            // Larger repeating pattern, expected to be moderately compressible.
            case DataMode::MEDIUM_COMPRESSIBLE:
		switch (i % 8) {
            	  case 0:
                     data[i] = dist(rng);
                     break;
                  default:
                     data[i] = static_cast<int>((i / 8) % 128);
                     break;
       		 }
                 break;

  //              data[i] = static_cast<int>((i * 17) % 10000);
//                break;

            // Random values, expected to compress poorly.
            case DataMode::RANDOM_DATA:
                data[i] = dist(rng);
                break;
        }
    }
}

int main() {
    try {
        // Select GPU device 0.
        CUDA_CHECK(cudaSetDevice(0));

        // Create folders for executable and result files.
        std::system("mkdir -p bin");
        std::system("mkdir -p results");

        // Create a CUDA stream used for asynchronous memory copies and kernel launches.
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        // Each input is split into 4 MB chunks for batched LZ4 compression/decompression.
        const size_t chunk_size = 1ULL << 22; // 4 MB

        // Warmup iterations are executed but excluded from final statistics.
        const int warmup = 2;

        // Number of measured repetitions.
        const int iterations = 10;

        // Input sizes in MB.
        // const std::vector<size_t> sizes_mb = {4, 16, 64, 256, 512, 1024, 2048};
        const std::vector<size_t> sizes_mb = {4, 16, 64, 256, 512, 1024};

        // Input data modes tested for each size.
        const std::vector<DataMode> modes = {
            DataMode::HIGH_COMPRESSIBLE,
            DataMode::MEDIUM_COMPRESSIBLE,
            DataMode::RANDOM_DATA
        };

        // Summary CSV containing averaged results.
        std::ofstream csv("results/lz4_nvcomp_pipeline/csv_file/parallel_cpu_lz4_nvcomp_full_pipeline_results.csv");
        csv << "Mode,MB,Ratio,"
               "Base_GBps_Avg,Base_GBps_StdDev,"
               "Pipe_Phys_GBps_Avg,Pipe_Phys_GBps_StdDev,"
               "Pipe_Eff_GBps_Avg,Pipe_Eff_GBps_StdDev,"
               "Base_ms_Avg,Base_ms_StdDev,"
               "CPUCompOnce_ms,"
               "H2DComp_ms_Avg,Decomp_ms_Avg,Kernel_ms_Avg,D2H_ms_Avg,"
               "Pipe_ms_Avg,Pipe_ms_StdDev,Winner\n";

        // Detailed CSV containing every measured run.
        std::ofstream detail_csv("results/lz4_nvcomp_pipeline/csv_file/parallel_cpu_lz4_nvcomp_full_pipeline_detailed_runs.csv");
        detail_csv << "Mode,MB,Run,"
                   << "Base_ms,Base_GBps,"
                   << "Pipe_ms,Pipe_Phys_GBps,Pipe_Eff_GBps,"
                   << "H2DComp_ms,Decomp_ms,Kernel_ms,D2H_ms\n";

        // Configure terminal number formatting.
        std::cout << std::fixed << std::setprecision(2);

        // Print table header.
        std::cout << std::left
                  << std::setw(10) << "MODE"
                  << std::setw(8)  << "MB"
                  << std::setw(10) << "RATIO"
                  << std::setw(18) << "BASE GB/s"
                  << std::setw(20) << "PIPE phys"
                  << std::setw(20) << "PIPE eff"
                  << std::setw(10) << "WINNER"
                  << "\n";

        std::cout << "--------------------------------------------------------------------------------------------------\n";

        // Run all experiments for every data mode and input size.
        for (DataMode mode : modes) {
            for (size_t mb : sizes_mb) {
                // Convert the current test size from MB to bytes.
                const size_t total_bytes = mb * 1024ULL * 1024ULL;

                // Number of int elements in the input buffer.
                const size_t n_ints = total_bytes / sizeof(int);

                // Original uncompressed size in GiB.
                const double total_gib = bytes_to_gib(total_bytes);

                // Number of chunks needed for the current input size.
                const size_t num_chunks = (total_bytes + chunk_size - 1) / chunk_size;

                // -------- host input/output --------
                int* host_in_ints = nullptr;
                int* host_out_ints = nullptr;

                // Allocate pinned host memory for faster CPU-GPU transfers.
                CUDA_CHECK(cudaMallocHost(&host_in_ints, total_bytes));
                CUDA_CHECK(cudaMallocHost(&host_out_ints, total_bytes));

                // Fill input according to the selected data mode.
                fill_input(host_in_ints, n_ints, mode);

                // Reinterpret integer input as bytes for LZ4 compression.
                const char* host_in_bytes = reinterpret_cast<const char*>(host_in_ints);

                // -------- CPU compression once --------
                // CPU compression is measured once before repeated GPU pipeline timing.
                auto comp_once_start = std::chrono::high_resolution_clock::now();

                // Per-chunk metadata used by CPU compression and nvCOMP decompression.
                std::vector<size_t> host_uncomp_sizes(num_chunks);
                std::vector<size_t> host_comp_sizes(num_chunks);
                std::vector<size_t> host_comp_offsets(num_chunks);
                std::vector<std::vector<char>> host_comp_chunks(num_chunks);

                // Total compressed size over all chunks.
                size_t total_comp_bytes = 0;

                // Compress the input chunk by chunk using CPU LZ4.
                for (size_t c = 0; c < num_chunks; ++c) {
                    const size_t offset = c * chunk_size;
                    const size_t this_uncomp_size =
                        std::min(chunk_size, total_bytes - offset);

                    host_uncomp_sizes[c] = this_uncomp_size;

                    // Maximum possible compressed size for this chunk.
                    const int max_comp_size =
                        LZ4_compressBound(static_cast<int>(this_uncomp_size));

                    host_comp_chunks[c].resize(static_cast<size_t>(max_comp_size));

                    // Compress this chunk using LZ4 on the CPU.
                    const int comp_size = LZ4_compress_default(
                        host_in_bytes + offset,
                        host_comp_chunks[c].data(),
                        static_cast<int>(this_uncomp_size),
                        max_comp_size);

                    if (comp_size <= 0) {
                        throw std::runtime_error("LZ4 compression failed.");
                    }

                    // Store compressed size and offset in the final flat compressed buffer.
                    host_comp_sizes[c] = static_cast<size_t>(comp_size);
                    host_comp_offsets[c] = total_comp_bytes;
                    total_comp_bytes += static_cast<size_t>(comp_size);
                }

                // Flat compressed buffer containing all compressed chunks consecutively.
                std::vector<char> host_comp_flat(total_comp_bytes);

                // Copy each compressed chunk into the flat compressed buffer.
                for (size_t c = 0; c < num_chunks; ++c) {
                    std::memcpy(
                        host_comp_flat.data() + host_comp_offsets[c],
                        host_comp_chunks[c].data(),
                        host_comp_sizes[c]);
                }

                auto comp_once_end = std::chrono::high_resolution_clock::now();

                // CPU compression time in milliseconds.
                const double cpu_comp_once_ms = ms_between(comp_once_start, comp_once_end);

                // Compression ratio: original bytes divided by compressed bytes.

		const double comp_ratio =
		 std::max(
        		0.0,
        		(1.0 - static_cast<double>(total_comp_bytes) / static_cast<double>(total_bytes)) * 100.0
    			);
//			(1.0 - static_cast<double>(total_comp_bytes) / static_cast<double>(total_bytes)) * 100.0;
                // -------- device allocations --------
                int* d_base = nullptr;
                char* d_comp_flat = nullptr;
                char* d_decomp_flat = nullptr;
                void** d_comp_ptrs = nullptr;
                void** d_decomp_ptrs = nullptr;
                size_t* d_comp_sizes = nullptr;
                size_t* d_uncomp_sizes = nullptr;
                size_t* d_actual_uncomp_sizes = nullptr;
                nvcompStatus_t* d_statuses = nullptr;
                void* d_temp = nullptr;

                // Baseline device buffer for uncompressed data.
                CUDA_CHECK(cudaMalloc(&d_base, total_bytes));

                // Device buffer for compressed input chunks.
                CUDA_CHECK(cudaMalloc(&d_comp_flat, total_comp_bytes));

                // Device buffer for decompressed output.
                CUDA_CHECK(cudaMalloc(&d_decomp_flat, total_bytes));

                // Device arrays containing per-chunk compressed and decompressed pointers.
                CUDA_CHECK(cudaMalloc(&d_comp_ptrs, num_chunks * sizeof(void*)));
                CUDA_CHECK(cudaMalloc(&d_decomp_ptrs, num_chunks * sizeof(void*)));

                // Device arrays containing per-chunk sizes.
                CUDA_CHECK(cudaMalloc(&d_comp_sizes, num_chunks * sizeof(size_t)));
                CUDA_CHECK(cudaMalloc(&d_uncomp_sizes, num_chunks * sizeof(size_t)));
                CUDA_CHECK(cudaMalloc(&d_actual_uncomp_sizes, num_chunks * sizeof(size_t)));

                // Device array containing per-chunk decompression status.
                CUDA_CHECK(cudaMalloc(&d_statuses, num_chunks * sizeof(nvcompStatus_t)));

                // Host arrays used to prepare device pointer arrays for nvCOMP.
                std::vector<void*> host_d_comp_ptrs(num_chunks);
                std::vector<void*> host_d_decomp_ptrs(num_chunks);

                // Build chunk pointer arrays.
                for (size_t c = 0; c < num_chunks; ++c) {
                    host_d_comp_ptrs[c] = d_comp_flat + host_comp_offsets[c];
                    host_d_decomp_ptrs[c] = d_decomp_flat + c * chunk_size;
                }

                // Copy compressed chunk pointers to device.
                CUDA_CHECK(cudaMemcpy(
                    d_comp_ptrs,
                    host_d_comp_ptrs.data(),
                    num_chunks * sizeof(void*),
                    cudaMemcpyHostToDevice));

                // Copy decompressed chunk pointers to device.
                CUDA_CHECK(cudaMemcpy(
                    d_decomp_ptrs,
                    host_d_decomp_ptrs.data(),
                    num_chunks * sizeof(void*),
                    cudaMemcpyHostToDevice));

                // Copy compressed chunk sizes to device.
                CUDA_CHECK(cudaMemcpy(
                    d_comp_sizes,
                    host_comp_sizes.data(),
                    num_chunks * sizeof(size_t),
                    cudaMemcpyHostToDevice));

                // Copy uncompressed chunk sizes to device.
                CUDA_CHECK(cudaMemcpy(
                    d_uncomp_sizes,
                    host_uncomp_sizes.data(),
                    num_chunks * sizeof(size_t),
                    cudaMemcpyHostToDevice));

                // -------- nvCOMP temp buffer --------
                size_t temp_bytes = 0;

                // Default options for batched LZ4 decompression.
                const nvcompBatchedLZ4DecompressOpts_t opts =
                    nvcompBatchedLZ4DecompressDefaultOpts;

                // Ask nvCOMP how much temporary GPU memory is required for decompression.
                NVCOMP_CHECK(
                    nvcompBatchedLZ4DecompressGetTempSizeSync(
                        (const void* const* const)d_comp_ptrs,
                        d_comp_sizes,
                        num_chunks,
                        chunk_size,
                        &temp_bytes,
                        total_bytes,
                        opts,
                        d_statuses,
                        stream
                    )
                );

                // Allocate temporary GPU workspace used internally by nvCOMP.
                CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));

                // -------- baseline timing --------
                std::vector<double> base_ms_runs;
                std::vector<double> base_gbps_runs;

                // Baseline pipeline:
                // uncompressed H2D transfer -> GPU compute -> uncompressed D2H transfer.
                for (int it = 0; it < warmup + iterations; ++it) {
                    auto t0 = std::chrono::high_resolution_clock::now();

                    // Copy uncompressed input from host to GPU.
                    CUDA_CHECK(cudaMemcpyAsync(
                        d_base, host_in_ints, total_bytes,
                        cudaMemcpyHostToDevice, stream));

                    // Run compute kernel on uncompressed GPU buffer.
                    compute_kernel<<<static_cast<int>((n_ints + 255) / 256), 256, 0, stream>>>(
                        d_base,
                        static_cast<int>(n_ints));
                    CUDA_CHECK(cudaGetLastError());

                    // Copy processed data back to host.
                    CUDA_CHECK(cudaMemcpyAsync(
                        host_out_ints, d_base, total_bytes,
                        cudaMemcpyDeviceToHost, stream));

                    // Wait until baseline operations finish.
                    CUDA_CHECK(cudaStreamSynchronize(stream));

                    auto t1 = std::chrono::high_resolution_clock::now();

                    // Ignore warmup runs and record only measured iterations.
                    if (it >= warmup) {
                        const double base_ms_run = ms_between(t0, t1);
                        const double base_gbps_run = total_gib / (base_ms_run / 1000.0);

                        base_ms_runs.push_back(base_ms_run);
                        base_gbps_runs.push_back(base_gbps_run);
                    }
                }

                // Average and standard deviation for baseline.
                const double base_ms = mean(base_ms_runs);
                const double base_ms_stddev = stddev_sample(base_ms_runs);
                const double base_gbps = mean(base_gbps_runs);
                const double base_gbps_stddev = stddev_sample(base_gbps_runs);

                // -------- pipeline timing --------
                std::vector<double> h2d_ms_runs;
                std::vector<double> decomp_ms_runs;
                std::vector<double> kernel_ms_runs;
                std::vector<double> d2h_ms_runs;
                std::vector<double> pipe_ms_runs;
                std::vector<double> pipe_phys_gbps_runs;
                std::vector<double> pipe_eff_gbps_runs;

                // Compressed pipeline:
                // compressed H2D transfer -> GPU decompression -> GPU compute -> D2H transfer.
                for (int it = 0; it < warmup + iterations; ++it) {
                    auto total_start = std::chrono::high_resolution_clock::now();

                    // Measure host-to-device transfer of compressed data.
                    auto h2d_start = std::chrono::high_resolution_clock::now();
                    CUDA_CHECK(cudaMemcpyAsync(
                        d_comp_flat, host_comp_flat.data(), total_comp_bytes,
                        cudaMemcpyHostToDevice, stream));
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                    auto h2d_end = std::chrono::high_resolution_clock::now();

                    // Measure GPU decompression time.
                    auto decomp_start = std::chrono::high_resolution_clock::now();

                    // Retrieve decompressed sizes for each compressed LZ4 chunk.
                    NVCOMP_CHECK(
                        nvcompBatchedLZ4GetDecompressSizeAsync(
                            (const void* const*)d_comp_ptrs,
                            d_comp_sizes,
                            d_uncomp_sizes,
                            num_chunks,
                            stream
                        )
                    );
                    CUDA_CHECK(cudaStreamSynchronize(stream));

                    // Decompress all chunks on the GPU using nvCOMP.
                    NVCOMP_CHECK(
                        nvcompBatchedLZ4DecompressAsync(
                            (const void* const*)d_comp_ptrs,
                            d_comp_sizes,
                            d_uncomp_sizes,
                            d_actual_uncomp_sizes,
                            num_chunks,
                            d_temp,
                            temp_bytes,
                            d_decomp_ptrs,
                            opts,
                            d_statuses,
                            stream
                        )
                    );
                    CUDA_CHECK(cudaStreamSynchronize(stream));

                    auto decomp_end = std::chrono::high_resolution_clock::now();

                    // Measure GPU compute time on decompressed data.
                    auto kernel_start = std::chrono::high_resolution_clock::now();

                    compute_kernel<<<static_cast<int>((n_ints + 255) / 256), 256, 0, stream>>>(
                        reinterpret_cast<int*>(d_decomp_flat),
                        static_cast<int>(n_ints));
                    CUDA_CHECK(cudaGetLastError());
                    CUDA_CHECK(cudaStreamSynchronize(stream));

                    auto kernel_end = std::chrono::high_resolution_clock::now();

                    // Measure device-to-host transfer of decompressed processed output.
                    auto d2h_start = std::chrono::high_resolution_clock::now();

                    CUDA_CHECK(cudaMemcpyAsync(
                        host_out_ints, d_decomp_flat, total_bytes,
                        cudaMemcpyDeviceToHost, stream));
                    CUDA_CHECK(cudaStreamSynchronize(stream));

                    auto d2h_end = std::chrono::high_resolution_clock::now();

                    auto total_end = std::chrono::high_resolution_clock::now();

                    // Ignore warmup runs and store timing/throughput for measured iterations.
                    if (it >= warmup) {
                        const int run_id = it - warmup + 1;

                        const double h2d_ms_run = ms_between(h2d_start, h2d_end);
                        const double decomp_ms_run = ms_between(decomp_start, decomp_end);
                        const double kernel_ms_run = ms_between(kernel_start, kernel_end);
                        const double d2h_ms_run = ms_between(d2h_start, d2h_end);
                        const double pipe_ms_run = ms_between(total_start, total_end);

                        // Physical throughput uses compressed bytes.
                        const double pipe_phys_gbps_run =
                            bytes_to_gib(total_comp_bytes) / (pipe_ms_run / 1000.0);

                        // Effective throughput uses original uncompressed bytes.
                        const double pipe_eff_gbps_run =
                            total_gib / (pipe_ms_run / 1000.0);

                        h2d_ms_runs.push_back(h2d_ms_run);
                        decomp_ms_runs.push_back(decomp_ms_run);
                        kernel_ms_runs.push_back(kernel_ms_run);
                        d2h_ms_runs.push_back(d2h_ms_run);
                        pipe_ms_runs.push_back(pipe_ms_run);
                        pipe_phys_gbps_runs.push_back(pipe_phys_gbps_run);
                        pipe_eff_gbps_runs.push_back(pipe_eff_gbps_run);

                        // Write detailed run data.
                        detail_csv << mode_to_string(mode) << ","
                                   << mb << ","
                                   << run_id << ","
                                   << base_ms_runs[run_id - 1] << ","
                                   << base_gbps_runs[run_id - 1] << ","
                                   << pipe_ms_run << ","
                                   << pipe_phys_gbps_run << ","
                                   << pipe_eff_gbps_run << ","
                                   << h2d_ms_run << ","
                                   << decomp_ms_run << ","
                                   << kernel_ms_run << ","
                                   << d2h_ms_run << "\n";
                    }
                }

                // Average individual pipeline stages.
                const double avg_h2d_ms = mean(h2d_ms_runs);
                const double avg_decomp_ms = mean(decomp_ms_runs);
                const double avg_kernel_ms = mean(kernel_ms_runs);
                const double avg_d2h_ms = mean(d2h_ms_runs);

                // Average total pipeline time.
                const double avg_pipe_ms = mean(pipe_ms_runs);
                const double pipe_ms_stddev = stddev_sample(pipe_ms_runs);

                // Average physical pipeline throughput.
                const double pipe_phys_gbps = mean(pipe_phys_gbps_runs);
                const double pipe_phys_gbps_stddev = stddev_sample(pipe_phys_gbps_runs);

                // Average effective pipeline throughput.
                const double pipe_eff_gbps = mean(pipe_eff_gbps_runs);
                const double pipe_eff_gbps_stddev = stddev_sample(pipe_eff_gbps_runs);

                // Winner is decided using effective pipeline throughput vs baseline throughput.
                const std::string winner =
                    (pipe_eff_gbps > base_gbps) ? "PIPE" : "BASE";

                // Print green for PIPE winner and red for BASE winner.
                std::cout << (winner == "PIPE" ? "\033[32m" : "\033[31m");

                // Print mode, size, and compression ratio.
                std::cout << std::left
                          << std::setw(10) << mode_to_string(mode)
                          << std::setw(8)  << mb
                          << std::setw(10) << comp_ratio;

                // Print baseline throughput.
                std::cout << std::right
                          << std::setw(7) << base_gbps
                          << " ± "
                          << std::setw(6) << base_gbps_stddev
                          << "   ";

                // Print physical pipeline throughput.
                std::cout << std::setw(7) << pipe_phys_gbps
                          << " ± "
                          << std::setw(6) << pipe_phys_gbps_stddev
                          << "   ";

                // Print effective pipeline throughput.
                std::cout << std::setw(7) << pipe_eff_gbps
                          << " ± "
                          << std::setw(6) << pipe_eff_gbps_stddev
                          << "   ";

                // Print winner.
                std::cout << std::left
                          << std::setw(10) << winner
                          << "\n";

                // Reset terminal color.
                std::cout << "\033[0m";

                // Write aggregated results to summary CSV.
                csv << mode_to_string(mode) << ","
                    << mb << ","
                    << comp_ratio << ","
                    << base_gbps << ","
                    << base_gbps_stddev << ","
                    << pipe_phys_gbps << ","
                    << pipe_phys_gbps_stddev << ","
                    << pipe_eff_gbps << ","
                    << pipe_eff_gbps_stddev << ","
                    << base_ms << ","
                    << base_ms_stddev << ","
                    << cpu_comp_once_ms << ","
                    << avg_h2d_ms << ","
                    << avg_decomp_ms << ","
                    << avg_kernel_ms << ","
                    << avg_d2h_ms << ","
                    << avg_pipe_ms << ","
                    << pipe_ms_stddev << ","
                    << winner << "\n";

                // Free all device and pinned host memory for this experiment.
                CUDA_CHECK(cudaFree(d_base));
                CUDA_CHECK(cudaFree(d_comp_flat));
                CUDA_CHECK(cudaFree(d_decomp_flat));
                CUDA_CHECK(cudaFree(d_comp_ptrs));
                CUDA_CHECK(cudaFree(d_decomp_ptrs));
                CUDA_CHECK(cudaFree(d_comp_sizes));
                CUDA_CHECK(cudaFree(d_uncomp_sizes));
                CUDA_CHECK(cudaFree(d_actual_uncomp_sizes));
                CUDA_CHECK(cudaFree(d_statuses));
                CUDA_CHECK(cudaFree(d_temp));
                CUDA_CHECK(cudaFreeHost(host_in_ints));
                CUDA_CHECK(cudaFreeHost(host_out_ints));
            }
        }

        // Close result files and destroy CUDA stream.
        csv.close();
        detail_csv.close();
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    }
    catch (const std::exception& e) {
        // Print exception message if any C++ runtime error occurs.
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
}
