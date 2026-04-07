#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                      << " at line " << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while (0)

__global__ void memory_kernel(int* data, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = data[idx] + 1;
    }
}

int main() {
    std::vector<size_t> sizes_mb = {1, 4, 16, 64, 256, 512};

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Size(MB)\tCPU(ms)\t\tH2D(ms)\t\tD2H(ms)\t\tGPU(ms)\t\tCPU(GB/s)\tH2D(GB/s)\tD2H(GB/s)\tGPU(GB/s)\n";

    for (size_t size_mb : sizes_mb) {
        size_t bytes = size_mb * 1024 * 1024;
        size_t N = bytes / sizeof(int);

        std::vector<int> h_src(N, 1);
        std::vector<int> h_dst(N, 0);
        int* d_data = nullptr;

        CHECK_CUDA(cudaMalloc(&d_data, bytes));
        CHECK_CUDA(cudaMemcpy(d_data, h_src.data(), bytes, cudaMemcpyHostToDevice));

        // ================= CPU Benchmark: RAM copy =================
        auto cpu_start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < N; i++) {
            h_dst[i] = h_src[i];
        }
        auto cpu_end = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

        // ================= Warm-up Transfer =================
        CHECK_CUDA(cudaMemcpy(d_data, h_src.data(), bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(h_src.data(), d_data, bytes, cudaMemcpyDeviceToHost));

        // ================= H2D =================
        CHECK_CUDA(cudaEventRecord(start));
        CHECK_CUDA(cudaMemcpy(d_data, h_src.data(), bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float h2d_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&h2d_ms, start, stop));

        // ================= D2H =================
        CHECK_CUDA(cudaEventRecord(start));
        CHECK_CUDA(cudaMemcpy(h_src.data(), d_data, bytes, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float d2h_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&d2h_ms, start, stop));

        // ================= GPU Memory Kernel =================
        int threads = 256;
        int blocks = (N + threads - 1) / threads;

        memory_kernel<<<blocks, threads>>>(d_data, N);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaEventRecord(start));
        memory_kernel<<<blocks, threads>>>(d_data, N);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float gpu_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&gpu_ms, start, stop));

        // CPU copy does read + write, so count 2 * bytes
        double cpu_gbps = ((2.0 * bytes) / 1e9) / (cpu_ms / 1000.0);
        double h2d_gbps = (bytes / 1e9) / (h2d_ms / 1000.0);
        double d2h_gbps = (bytes / 1e9) / (d2h_ms / 1000.0);
        double gpu_gbps = ((2.0 * bytes) / 1e9) / (gpu_ms / 1000.0);

        std::cout << size_mb << "\t\t"
                  << cpu_ms << "\t\t"
                  << h2d_ms << "\t\t"
                  << d2h_ms << "\t\t"
                  << gpu_ms << "\t\t"
                  << cpu_gbps << "\t\t"
                  << h2d_gbps << "\t\t"
                  << d2h_gbps << "\t\t"
                  << gpu_gbps << "\n";

        CHECK_CUDA(cudaFree(d_data));
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}
