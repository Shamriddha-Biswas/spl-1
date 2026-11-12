#include "../include/processing.h"

#ifdef __CUDACC__

__global__ void kernel_gaussian_shared(const uint8_t* d_src, uint8_t* d_dst, int width, int height, int channels, const float* d_kernel, int ksize) {
    __shared__ float s_tile[24][24];

    int r = ksize / 2;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int gx = blockIdx.x * blockDim.x + tx;
    int gy = blockIdx.y * blockDim.y + ty;

    int smem_dim = 16 + 2 * r;

    for (int c = 0; c < channels; ++c) {
        for (int sy = ty; sy < smem_dim; sy += blockDim.y) {
            int cur_gy = blockIdx.y * blockDim.y + sy - r;
            if (cur_gy < 0) cur_gy = 0;
            else if (cur_gy >= height) cur_gy = height - 1;

            for (int sx = tx; sx < smem_dim; sx += blockDim.x) {
                int cur_gx = blockIdx.x * blockDim.x + sx - r;
                if (cur_gx < 0) cur_gx = 0;
                else if (cur_gx >= width) cur_gx = width - 1;

                uint64_t idx = ((uint64_t)cur_gy * (uint64_t)width + (uint64_t)cur_gx) * (uint64_t)channels + (uint64_t)c;
                s_tile[sy][sx] = (float)d_src[idx];
            }
        }

        __syncthreads();

        if (gx < width && gy < height) {
            float sum = 0.0f;
            for (int ky = -r; ky <= r; ++ky) {
                for (int kx = -r; kx <= r; ++kx) {
                    float weight = d_kernel[(ky + r) * ksize + (kx + r)];
                    float val = s_tile[ty + r + ky][tx + r + kx];
                    sum += val * weight;
                }
            }
            int out_val = (int)(sum + 0.5f);
            if (out_val < 0) out_val = 0;
            else if (out_val > 255) out_val = 255;

            uint64_t dst_idx = ((uint64_t)gy * (uint64_t)width + (uint64_t)gx) * (uint64_t)channels + (uint64_t)c;
            d_dst[dst_idx] = (uint8_t)out_val;
        }

        __syncthreads();
    }
}

#endif

bool gpu_gaussian_shared_memory(const Image* src, Image* dst, int ksize, float sigma, BenchmarkMetrics* metrics) {
    if (!src || !dst || !src->data || !dst->data) {
        return false;
    }
    if (ksize <= 0 || (ksize % 2) == 0 || sigma <= 0.0f || ksize > 7) {
        return false;
    }

#ifdef __CUDACC__
    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count <= 0) {
        return false;
    }

    KernelMatrix kernel = create_gaussian_kernel_2d(ksize, sigma);
    if (!kernel.weights) return false;

    size_t img_bytes = src->data_size;
    size_t kernel_bytes = (size_t)ksize * (size_t)ksize * sizeof(float);

    uint8_t* d_src = nullptr;
    uint8_t* d_dst = nullptr;
    float* d_kernel = nullptr;

    if (cudaMalloc((void**)&d_src, img_bytes) != cudaSuccess) {
        free_kernel_matrix(&kernel);
        return false;
    }
    if (cudaMalloc((void**)&d_dst, img_bytes) != cudaSuccess) {
        cudaFree(d_src);
        free_kernel_matrix(&kernel);
        return false;
    }
    if (cudaMalloc((void**)&d_kernel, kernel_bytes) != cudaSuccess) {
        cudaFree(d_src);
        cudaFree(d_dst);
        free_kernel_matrix(&kernel);
        return false;
    }

    cudaEvent_t start_h2d, stop_h2d, start_kernel, stop_kernel, start_d2h, stop_d2h;
    cudaEventCreate(&start_h2d);
    cudaEventCreate(&stop_h2d);
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&stop_kernel);
    cudaEventCreate(&start_d2h);
    cudaEventCreate(&stop_d2h);

    cudaEventRecord(start_h2d);
    cudaMemcpy(d_src, src->data, img_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel.weights, kernel_bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(stop_h2d);
    cudaEventSynchronize(stop_h2d);

    dim3 block(16, 16);
    dim3 grid((src->width + block.x - 1) / block.x, (src->height + block.y - 1) / block.y);

    cudaEventRecord(start_kernel);
    kernel_gaussian_shared<<<grid, block>>>(d_src, d_dst, src->width, src->height, src->channels, d_kernel, ksize);
    cudaEventRecord(stop_kernel);
    cudaEventSynchronize(stop_kernel);

    cudaEventRecord(start_d2h);
    cudaMemcpy(dst->data, d_dst, img_bytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop_d2h);
    cudaEventSynchronize(stop_d2h);

    if (metrics) {
        float h2d_ms = 0.0f, kernel_ms = 0.0f, d2h_ms = 0.0f;
        cudaEventElapsedTime(&h2d_ms, start_h2d, stop_h2d);
        cudaEventElapsedTime(&kernel_ms, start_kernel, stop_kernel);
        cudaEventElapsedTime(&d2h_ms, start_d2h, stop_d2h);
        metrics->h2d_time_ms = (double)h2d_ms;
        metrics->opt_kernel_time_ms = (double)kernel_ms;
        metrics->gpu_kernel_time_ms = (double)kernel_ms;
        metrics->d2h_time_ms = (double)d2h_ms;
        metrics->gpu_total_time_ms = (double)(h2d_ms + kernel_ms + d2h_ms);
    }

    cudaEventDestroy(start_h2d);
    cudaEventDestroy(stop_h2d);
    cudaEventDestroy(start_kernel);
    cudaEventDestroy(stop_kernel);
    cudaEventDestroy(start_d2h);
    cudaEventDestroy(stop_d2h);

    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFree(d_kernel);
    free_kernel_matrix(&kernel);
    return true;
#else
    return false;
#endif
}

