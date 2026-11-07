#include "../include/processing.h"

#ifdef __CUDACC__

__device__ static inline uint8_t gpu_clamp_u8(int v) {
    if (v < 0) return 0;
    if (v > 255) return 255;
    return (uint8_t)v;
}

__global__ void kernel_convolve_2d_global(const uint8_t* d_src, uint8_t* d_dst, int width, int height, int channels, const float* d_kernel, int ksize) {
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
        uint64_t dst_idx = ((uint64_t)y * (uint64_t)width + (uint64_t)x) * (uint64_t)channels + (uint64_t)c;
        d_dst[dst_idx] = gpu_clamp_u8(out_val);
    }
}

__global__ void kernel_convolve_1d_h(const uint8_t* d_src, float* d_tmp, int width, int height, int channels, const float* d_kernel_1d, int ksize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    int r = ksize / 2;

    for (int c = 0; c < channels; ++c) {
        float sum = 0.0f;
        for (int kx = -r; kx <= r; ++kx) {
            int px = x + kx;
            if (px < 0) px = 0;
            else if (px >= width) px = width - 1;

            float weight = d_kernel_1d[kx + r];
            uint64_t idx = ((uint64_t)y * (uint64_t)width + (uint64_t)px) * (uint64_t)channels + (uint64_t)c;
            sum += (float)d_src[idx] * weight;
        }
        uint64_t out_idx = ((uint64_t)y * (uint64_t)width + (uint64_t)x) * (uint64_t)channels + (uint64_t)c;
        d_tmp[out_idx] = sum;
    }
}

__global__ void kernel_convolve_1d_v(const float* d_tmp, uint8_t* d_dst, int width, int height, int channels, const float* d_kernel_1d, int ksize) {
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

            float weight = d_kernel_1d[ky + r];
            uint64_t idx = ((uint64_t)py * (uint64_t)width + (uint64_t)x) * (uint64_t)channels + (uint64_t)c;
            sum += d_tmp[idx] * weight;
        }
        int out_val = (int)(sum + 0.5f);
        uint64_t dst_idx = ((uint64_t)y * (uint64_t)width + (uint64_t)x) * (uint64_t)channels + (uint64_t)c;
        d_dst[dst_idx] = gpu_clamp_u8(out_val);
    }
}

#endif

bool gpu_convolve_2d_raw(const Image* src, Image* dst, const float* h_kernel, int ksize, BenchmarkMetrics* metrics) {
    if (!src || !dst || !src->data || !dst->data || !h_kernel || ksize <= 0) {
        return false;
    }

#ifdef __CUDACC__
    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count <= 0) {
        return false;
    }

    size_t img_bytes = src->data_size;
    size_t kernel_bytes = (size_t)ksize * (size_t)ksize * sizeof(float);

    uint8_t* d_src = nullptr;
    uint8_t* d_dst = nullptr;
    float* d_kernel = nullptr;

    if (cudaMalloc((void**)&d_src, img_bytes) != cudaSuccess) return false;
    if (cudaMalloc((void**)&d_dst, img_bytes) != cudaSuccess) {
        cudaFree(d_src);
        return false;
    }
    if (cudaMalloc((void**)&d_kernel, kernel_bytes) != cudaSuccess) {
        cudaFree(d_src);
        cudaFree(d_dst);
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
    cudaMemcpy(d_kernel, h_kernel, kernel_bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(stop_h2d);
    cudaEventSynchronize(stop_h2d);

    dim3 block(16, 16);
    dim3 grid((src->width + block.x - 1) / block.x, (src->height + block.y - 1) / block.y);

    cudaEventRecord(start_kernel);
    kernel_convolve_2d_global<<<grid, block>>>(d_src, d_dst, src->width, src->height, src->channels, d_kernel, ksize);
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
    return true;
#else
    return false;
#endif
}

bool gpu_convolve_1d_separable_raw(const Image* src, Image* dst, const float* h_kernel_1d, int ksize, BenchmarkMetrics* metrics) {
    if (!src || !dst || !src->data || !dst->data || !h_kernel_1d || ksize <= 0) {
        return false;
    }

#ifdef __CUDACC__
    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count <= 0) {
        return false;
    }

    size_t img_bytes = src->data_size;
    size_t tmp_bytes = (size_t)src->width * (size_t)src->height * (size_t)src->channels * sizeof(float);
    size_t kernel_bytes = (size_t)ksize * sizeof(float);

    uint8_t* d_src = nullptr;
    uint8_t* d_dst = nullptr;
    float* d_tmp = nullptr;
    float* d_kernel_1d = nullptr;

    if (cudaMalloc((void**)&d_src, img_bytes) != cudaSuccess) return false;
    if (cudaMalloc((void**)&d_dst, img_bytes) != cudaSuccess) {
        cudaFree(d_src);
        return false;
    }
    if (cudaMalloc((void**)&d_tmp, tmp_bytes) != cudaSuccess) {
        cudaFree(d_src);
        cudaFree(d_dst);
        return false;
    }
    if (cudaMalloc((void**)&d_kernel_1d, kernel_bytes) != cudaSuccess) {
        cudaFree(d_src);
        cudaFree(d_dst);
        cudaFree(d_tmp);
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
    cudaMemcpy(d_kernel_1d, h_kernel_1d, kernel_bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(stop_h2d);
    cudaEventSynchronize(stop_h2d);

    dim3 block(16, 16);
    dim3 grid((src->width + block.x - 1) / block.x, (src->height + block.y - 1) / block.y);

    cudaEventRecord(start_kernel);
    kernel_convolve_1d_h<<<grid, block>>>(d_src, d_tmp, src->width, src->height, src->channels, d_kernel_1d, ksize);
    kernel_convolve_1d_v<<<grid, block>>>(d_tmp, d_dst, src->width, src->height, src->channels, d_kernel_1d, ksize);
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
    cudaFree(d_tmp);
    cudaFree(d_kernel_1d);
    return true;
#else
    return false;
#endif
}

