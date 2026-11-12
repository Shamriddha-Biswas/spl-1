#include "../include/processing.h"

#ifdef __CUDACC__

__global__ void kernel_block_benchmark(const uint8_t* d_src, uint8_t* d_dst, int width, int height, int channels, const float* d_kernel, int ksize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    int r = ksize / 2;

    for (int c = 0; c < channels; ++c) {
        float sum = 0.0f;
        for (int ky = -r; ky <= r; ++ky) {
            int py = y + ky;
            if (py < 0) py = 0;
            else if (py >= height) py = height - 1;

            for (int kx = -r; kx <= r; ++kx) {
                int px = x + kx;
                if (px < 0) px = 0;
                else if (px >= width) px = width - 1;

                float weight = d_kernel[(ky + r) * ksize + (kx + r)];
                uint64_t idx = ((uint64_t)py * (uint64_t)width + (uint64_t)px) * (uint64_t)channels + (uint64_t)c;
                sum += (float)d_src[idx] * weight;
            }
        }
        int out_val = (int)(sum + 0.5f);
        if (out_val < 0) out_val = 0;
        else if (out_val > 255) out_val = 255;

        uint64_t dst_idx = ((uint64_t)y * (uint64_t)width + (uint64_t)x) * (uint64_t)channels + (uint64_t)c;
        d_dst[dst_idx] = (uint8_t)out_val;
    }
}

#endif

void run_block_size_experiment(const Image* src, int ksize, float sigma) {
    if (!src || !src->data || src->width <= 0 || src->height <= 0) {
        std::cout << "Error: Invalid image for block size experiment." << std::endl;
        return;
    }

#ifdef __CUDACC__
    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count <= 0) {
        std::cout << "Notice: CUDA device not available for block size experiment." << std::endl;
        return;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    struct BlockConfig {
        int bx;
        int by;
        const char* label;
    };

    BlockConfig configs[4] = {
        { 8, 8, "8 x 8 (64 threads)" },
        { 16, 16, "16 x 16 (256 threads)" },
        { 32, 8, "32 x 8 (256 threads)" },
        { 32, 16, "32 x 16 (512 threads)" }
    };

    KernelMatrix kernel = create_gaussian_kernel_2d(ksize, sigma);
    if (!kernel.weights) return;

    size_t img_bytes = src->data_size;
    size_t kernel_bytes = (size_t)ksize * (size_t)ksize * sizeof(float);

    uint8_t* d_src = nullptr;
    uint8_t* d_dst = nullptr;
    float* d_kernel = nullptr;

    if (cudaMalloc((void**)&d_src, img_bytes) != cudaSuccess ||
        cudaMalloc((void**)&d_dst, img_bytes) != cudaSuccess ||
        cudaMalloc((void**)&d_kernel, kernel_bytes) != cudaSuccess) {
        if (d_src) cudaFree(d_src);
        if (d_dst) cudaFree(d_dst);
        if (d_kernel) cudaFree(d_kernel);
        free_kernel_matrix(&kernel);
        return;
    }

    cudaMemcpy(d_src, src->data, img_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel.weights, kernel_bytes, cudaMemcpyHostToDevice);

    std::cout << "\n=================================================" << std::endl;
    std::cout << "       CUDA BLOCK CONFIGURATION EXPERIMENT       " << std::endl;
    std::cout << "=================================================" << std::endl;
    std::cout << "GPU Device Name       : " << prop.name << std::endl;
    std::cout << "Image Resolution      : " << src->width << " x " << src->height << std::endl;
    std::cout << "Kernel Filter Size    : " << ksize << " x " << ksize << " (Sigma: " << sigma << ")" << std::endl;
    std::cout << "Max Threads Per Block : " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "-------------------------------------------------" << std::endl;
    std::cout << "Block Shape     Threads/Block    Kernel Time (ms)" << std::endl;
    std::cout << "-------------------------------------------------" << std::endl;

    double best_time = 1e9;
    std::string best_config = "None";

    for (int i = 0; i < 4; ++i) {
        int bx = configs[i].bx;
        int by = configs[i].by;
        int threads = bx * by;

        if (threads > prop.maxThreadsPerBlock || bx > prop.maxThreadsDim[0] || by > prop.maxThreadsDim[1]) {
            std::cout << configs[i].label << "  : Unsupported by device limits." << std::endl;
            continue;
        }

        dim3 block(bx, by);
        dim3 grid((src->width + bx - 1) / bx, (src->height + by - 1) / by);

        kernel_block_benchmark<<<grid, block>>>(d_src, d_dst, src->width, src->height, src->channels, d_kernel, ksize);
        cudaDeviceSynchronize();

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        const int runs = 5;
        for (int r = 0; r < runs; ++r) {
            kernel_block_benchmark<<<grid, block>>>(d_src, d_dst, src->width, src->height, src->channels, d_kernel, ksize);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float total_ms = 0.0f;
        cudaEventElapsedTime(&total_ms, start, stop);
        double avg_ms = (double)total_ms / (double)runs;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        std::cout << configs[i].label << "       " << avg_ms << " ms" << std::endl;

        if (avg_ms < best_time) {
            best_time = avg_ms;
            best_config = configs[i].label;
        }
    }

    std::cout << "-------------------------------------------------" << std::endl;
    std::cout << "Optimal Block Configuration: " << best_config << " (" << best_time << " ms)" << std::endl;
    std::cout << "=================================================\n" << std::endl;

    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFree(d_kernel);
    free_kernel_matrix(&kernel);
#else
    std::cout << "Notice: Built without CUDA compiler. Block experiments require nvcc build." << std::endl;
#endif
}

