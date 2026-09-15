#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cuda_runtime.h>

// Abort immediately when a CUDA operation fails and report
// the corresponding runtime error and source line.
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                      << " at line " << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while (0)


// Simple GPU memory workload used for the device-side baseline.
// Each thread reads and updates one integer.
__global__ void memory_kernel(int* data, size_t N) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        data[idx] = data[idx] + 1;
    }
}


int main() {

    // Input sizes used to observe how transfer and memory-processing
    // performance changes as the working set grows.
    std::vector<size_t> sizes_mb = {1, 4, 16, 64, 256, 512};

    cudaEvent_t start, stop;

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    std::cout << std::fixed << std::setprecision(3);

    std::cout
        << "Size(MB)\tCPU(ms)\t\tH2D(ms)\t\tD2H(ms)\t\tGPU(ms)\t\t"
        << "CPU(GB/s)\tH2D(GB/s)\tD2H(GB/s)\tGPU(GB/s)\n";


    for (size_t size_mb : sizes_mb) {

        size_t bytes = size_mb * 1024 * 1024;
        size_t N = bytes / sizeof(int);

        // Host buffers used for the CPU-copy measurement.
        std::vector<int> h_src(N, 1);
        std::vector<int> h_dst(N, 0);

        int* d_data = nullptr;

        CHECK_CUDA(cudaMalloc(&d_data, bytes));

        // Initialize the device buffer before timing individual operations.
        CHECK_CUDA(cudaMemcpy(
            d_data,
            h_src.data(),
            bytes,
            cudaMemcpyHostToDevice
        ));


        // Measure a sequential copy between two host-memory buffers.
        auto cpu_start =
            std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < N; i++) {
            h_dst[i] = h_src[i];
        }

        auto cpu_end =
            std::chrono::high_resolution_clock::now();

        double cpu_ms =
            std::chrono::duration<double, std::milli>(
                cpu_end - cpu_start
            ).count();


        // Warm up both transfer directions before collecting PCIe timings.
        CHECK_CUDA(cudaMemcpy(
            d_data,
            h_src.data(),
            bytes,
            cudaMemcpyHostToDevice
        ));

        CHECK_CUDA(cudaMemcpy(
            h_src.data(),
            d_data,
            bytes,
            cudaMemcpyDeviceToHost
        ));


        // Host-to-device transfer time.
        CHECK_CUDA(cudaEventRecord(start));

        CHECK_CUDA(cudaMemcpy(
            d_data,
            h_src.data(),
            bytes,
            cudaMemcpyHostToDevice
        ));

        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float h2d_ms = 0.0f;

        CHECK_CUDA(cudaEventElapsedTime(
            &h2d_ms,
            start,
            stop
        ));


        // Device-to-host transfer time.
        CHECK_CUDA(cudaEventRecord(start));

        CHECK_CUDA(cudaMemcpy(
            h_src.data(),
            d_data,
            bytes,
            cudaMemcpyDeviceToHost
        ));

        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float d2h_ms = 0.0f;

        CHECK_CUDA(cudaEventElapsedTime(
            &d2h_ms,
            start,
            stop
        ));


        // Configure enough CUDA threads to cover the complete input.
        int threads = 256;
        int blocks = (N + threads - 1) / threads;

        // Warm up the GPU kernel before measuring its execution time.
        memory_kernel<<<blocks, threads>>>(d_data, N);

        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());


        // Measure the device-memory processing kernel with CUDA events.
        CHECK_CUDA(cudaEventRecord(start));

        memory_kernel<<<blocks, threads>>>(d_data, N);

        CHECK_CUDA(cudaGetLastError());

        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float gpu_ms = 0.0f;

        CHECK_CUDA(cudaEventElapsedTime(
            &gpu_ms,
            start,
            stop
        ));


        // CPU and GPU memory operations read and write the full buffer,
        // so their effective traffic is represented as 2 * input bytes.
        double cpu_gbps =
            ((2.0 * bytes) / 1e9) /
            (cpu_ms / 1000.0);

        // H2D and D2H each represent a one-directional transfer.
        double h2d_gbps =
            (bytes / 1e9) /
            (h2d_ms / 1000.0);

        double d2h_gbps =
            (bytes / 1e9) /
            (d2h_ms / 1000.0);

        double gpu_gbps =
            ((2.0 * bytes) / 1e9) /
            (gpu_ms / 1000.0);


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