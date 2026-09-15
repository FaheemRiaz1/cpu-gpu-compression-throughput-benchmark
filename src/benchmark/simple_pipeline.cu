// Early RLE prototype used to compare an uncompressed CPU-to-GPU path
// against a compressed path with GPU-side decompression.

#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <string>
#include <cstdlib>
#include <cuda_runtime.h>

// Fail fast on CUDA runtime errors.
#define CHECK_CUDA(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

// One run-length encoded pair.
struct RLEPair {
    int value;
    int count;
};

// Generate deterministic runs whose length controls compressibility.
std::vector<int> generate_data(size_t N, int run_len) {
    std::vector<int> data(N);
    int val = 1;

    for (size_t i = 0; i < N; i++) {
        data[i] = val;
        if ((i + 1) % run_len == 0) val++;
    }
    return data;
}

// CPU-side run-length encoding.
std::vector<RLEPair> compress(const std::vector<int>& input) {
    std::vector<RLEPair> out;

    int curr = input[0];
    int count = 1;

    for (size_t i = 1; i < input.size(); i++) {
        if (input[i] == curr) count++;
        else {
            out.push_back({curr, count});
            curr = input[i];
            count = 1;
        }
    }
    out.push_back({curr, count});
    return out;
}

// GPU-side RLE decompression.
// Each thread handles one pair; computing the output offset requires
// scanning all preceding pair lengths, giving this prototype O(n^2) work.
__global__ void decompress_kernel(RLEPair* comp, int* out, int num_pairs, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_pairs) {
        int start = 0;

        // O(n²) bottleneck
        for (int i = 0; i < idx; i++)
            start += comp[i].count;

        for (int j = 0; j < comp[idx].count; j++) {
            int out_idx = start + j;

            int val = comp[idx].value;

            if (out_idx < N)
                out[out_idx] = val;
        }
    }
}

// Lightweight GPU computation applied after transfer/decompression.
__global__ void compute_kernel(int* data, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        int val = data[i];

        // Repeated arithmetic keeps the compute stage identical for both paths.
        for (int k = 0; k < 50; k++) {
            val = val * 2 + 1;
            val = val / 2;
        }

        data[i] = val;
    }
}

int main() {
    std::system("mkdir -p results");
    std::system("mkdir -p results/simple_pipeline");
    std::system("mkdir -p results/simple_pipeline/csv_file");
    std::system("mkdir -p results/graphs");

    // Store throughput measurements used by the simple-pipeline plots.
    std::ofstream csv("results/simple_pipeline/csv_file/simple_results.csv");

    if (!csv) {
        std::cerr << "Could not open simple_results.csv\n";
        return 1;
    }

    csv << "MB,RunLen,BaselineGBs,CompressedGBs\n";

    std::vector<int> run_lengths = {2, 32, 128};
    std::vector<int> sizes = {1, 4, 16, 64, 128};

    std::cout << "\n======= SIMPLE PIPELINE =======\n\n";

    std::cout << std::setw(6) << "MB"
              << std::setw(8) << "RunLen"
              << std::setw(14) << "Base GB/s"
              << std::setw(14) << "Comp GB/s"
              << std::setw(10) << "Winner"
              << "\n";

    std::cout << "-------------------------------------------------------------\n";
    std::cout << std::fixed;

    for (auto run_len : run_lengths) {

        for (auto mb : sizes) {

            size_t bytes = mb * 1024 * 1024;
            size_t N = bytes / sizeof(int);

            // Preserve the same logical input for both benchmark paths.
            auto original_data = generate_data(N, run_len);
            auto data = original_data;

            int* d_data;
            CHECK_CUDA(cudaMalloc(&d_data, bytes));

            // Baseline path: raw H2D -> GPU compute -> raw D2H.
            auto start = std::chrono::high_resolution_clock::now();

            cudaMemcpy(d_data, data.data(), bytes, cudaMemcpyHostToDevice);

            compute_kernel<<<(N+255)/256, 256>>>(d_data, N);

            cudaMemcpy(data.data(), d_data, bytes, cudaMemcpyDeviceToHost);

            cudaDeviceSynchronize();

            auto end = std::chrono::high_resolution_clock::now();

            double base_ms = std::chrono::duration<double, std::milli>(end - start).count();
            double base_gbps = (bytes / 1e9) / (base_ms / 1000.0);

            // Compressed path: CPU RLE compression -> compressed H2D ->
            // GPU decompression -> GPU compute -> raw D2H.
            data = original_data;  // Reset input before measuring the second path.

            start = std::chrono::high_resolution_clock::now();

            auto comp = compress(data);

            RLEPair* d_comp;
            CHECK_CUDA(cudaMalloc(&d_comp, comp.size() * sizeof(RLEPair)));

            cudaMemcpy(d_comp, comp.data(),
                       comp.size() * sizeof(RLEPair),
                       cudaMemcpyHostToDevice);

            decompress_kernel<<<(comp.size()+255)/256, 256>>>(
                d_comp, d_data, comp.size(), N);

            compute_kernel<<<(N+255)/256, 256>>>(d_data, N);

            cudaMemcpy(data.data(), d_data, bytes, cudaMemcpyDeviceToHost);

            cudaDeviceSynchronize();

            end = std::chrono::high_resolution_clock::now();

            double comp_ms = std::chrono::duration<double, std::milli>(end - start).count();
            double comp_gbps = (bytes / 1e9) / (comp_ms / 1000.0);

            // Compare end-to-end effective throughput of the two paths.
            std::string winner = (comp_gbps > base_gbps) ? "COMP" : "BASE";

            if (winner == "COMP") std::cout << "\033[32m";
            else std::cout << "\033[31m";

            std::cout << std::setw(6) << mb
                      << std::setw(8) << run_len
                      << std::setw(14) << std::setprecision(3) << base_gbps
                      << std::setw(14) << comp_gbps
                      << std::setw(10) << winner
                      << "\n";

            std::cout << "\033[0m";

            // Save the same measured throughputs printed above.
            csv << mb << ","
                << run_len << ","
                << base_gbps << ","
                << comp_gbps << "\n";

            cudaFree(d_comp);
            cudaFree(d_data);
        }

        std::cout << "-------------------------------------------------------------\n";
    }

    return 0;
}

// Build:
// mkdir -p bin
// nvcc -O2 src/benchmark/simple_pipeline.cu -o bin/simple_pipeline
//
// Run:
// ./bin/simple_pipeline
