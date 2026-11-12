#include "../include/processing.h"

#ifdef __CUDACC__

extern __global__ void kernel_convolve_2d_global(const uint8_t* d_src, uint8_t* d_dst, int width, int height, int channels, const float* d_kernel, int ksize);
extern __global__ void kernel_sobel_gradient(const uint8_t* d_src, uint8_t* d_dst, int width, int height, int channels);
extern __global__ void kernel_mean_filter(const uint8_t* d_src, uint8_t* d_dst, int width, int height, int channels, int window_size);
extern __global__ void kernel_median_filter(const uint8_t* d_src, uint8_t* d_dst, int width, int height, int channels, int window_size);
extern __global__ void kernel_grayscale(const uint8_t* d_src, uint8_t* d_dst, uint64_t total_pixels, int src_channels, int dst_channels);
extern __global__ void kernel_negative(const uint8_t* d_src, uint8_t* d_dst, uint64_t total_bytes);

#endif

bool gpu_pipeline_execute(const Image* src, Image* dst, const FilterType* filters, int num_filters, const FilterConfig* configs, BenchmarkMetrics* metrics) {
    if (!src || !dst || !src->data || !dst->data || !filters || num_filters <= 0 || !configs) {
        return false;
    }

#ifdef __CUDACC__
    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count <= 0) {
        return false;
    }

    size_t img_bytes = src->data_size;

    uint8_t* d_buf1 = nullptr;
    uint8_t* d_buf2 = nullptr;

    if (cudaMalloc((void**)&d_buf1, img_bytes) != cudaSuccess) return false;
    if (cudaMalloc((void**)&d_buf2, img_bytes) != cudaSuccess) {
        cudaFree(d_buf1);
        return false;
    }

    cudaEvent_t start_total, stop_total;
    cudaEventCreate(&start_total);
    cudaEventCreate(&stop_total);

    cudaEventRecord(start_total);
    cudaMemcpy(d_buf1, src->data, img_bytes, cudaMemcpyHostToDevice);

    uint8_t* current_in = d_buf1;
    uint8_t* current_out = d_buf2;

    dim3 block(16, 16);
    dim3 grid((src->width + block.x - 1) / block.x, (src->height + block.y - 1) / block.y);

    std::cout << "\n-------------------------------------------------" << std::endl;
    std::cout << "          EXECUTING GPU PROCESSING PIPELINE      " << std::endl;
    std::cout << "-------------------------------------------------" << std::endl;

    for (int step = 0; step < num_filters; ++step) {
        FilterType ftype = filters[step];
        const FilterConfig& cfg = configs[step];

        cudaEvent_t step_start, step_stop;
        cudaEventCreate(&step_start);
        cudaEventCreate(&step_stop);

        cudaEventRecord(step_start);

        if (ftype == FILTER_GAUSSIAN) {
            KernelMatrix kmat = create_gaussian_kernel_2d(cfg.kernel_size, cfg.sigma);
            float* d_k = nullptr;
            size_t k_bytes = cfg.kernel_size * cfg.kernel_size * sizeof(float);
            cudaMalloc((void**)&d_k, k_bytes);
            cudaMemcpy(d_k, kmat.weights, k_bytes, cudaMemcpyHostToDevice);

            kernel_convolve_2d_global<<<grid, block>>>(current_in, current_out, src->width, src->height, src->channels, d_k, cfg.kernel_size);

            cudaDeviceSynchronize();
            cudaFree(d_k);
            free_kernel_matrix(&kmat);
            std::cout << "Step " << (step + 1) << ": Gaussian Blur (" << cfg.kernel_size << "x" << cfg.kernel_size << ")";
        } else if (ftype == FILTER_SHARPEN) {
            float s = cfg.strength;
            float h_k[9] = { 0.0f, -s, 0.0f, -s, 1.0f + 4.0f * s, -s, 0.0f, -s, 0.0f };
            float* d_k = nullptr;
            cudaMalloc((void**)&d_k, 9 * sizeof(float));
            cudaMemcpy(d_k, h_k, 9 * sizeof(float), cudaMemcpyHostToDevice);

            kernel_convolve_2d_global<<<grid, block>>>(current_in, current_out, src->width, src->height, src->channels, d_k, 3);

            cudaDeviceSynchronize();
            cudaFree(d_k);
            std::cout << "Step " << (step + 1) << ": Sharpen (Strength: " << s << ")";
        } else if (ftype == FILTER_SOBEL) {
            kernel_sobel_gradient<<<grid, block>>>(current_in, current_out, src->width, src->height, src->channels);
            cudaDeviceSynchronize();
            std::cout << "Step " << (step + 1) << ": Sobel Edge Detection";
        } else if (ftype == FILTER_MEAN) {
            kernel_mean_filter<<<grid, block>>>(current_in, current_out, src->width, src->height, src->channels, cfg.window_size);
            cudaDeviceSynchronize();
            std::cout << "Step " << (step + 1) << ": Mean Noise Filter (" << cfg.window_size << "x" << cfg.window_size << ")";
        } else if (ftype == FILTER_MEDIAN) {
            kernel_median_filter<<<grid, block>>>(current_in, current_out, src->width, src->height, src->channels, cfg.window_size);
            cudaDeviceSynchronize();
            std::cout << "Step " << (step + 1) << ": Median Noise Filter (" << cfg.window_size << "x" << cfg.window_size << ")";
        } else if (ftype == FILTER_GRAYSCALE) {
            int block1d = 256;
            int grid1d = (int)((src->pixel_count + block1d - 1) / block1d);
            kernel_grayscale<<<grid1d, block1d>>>(current_in, current_out, src->pixel_count, src->channels, dst->channels);
            cudaDeviceSynchronize();
            std::cout << "Step " << (step + 1) << ": Grayscale Conversion";
        } else if (ftype == FILTER_NEGATIVE) {
            int block1d = 256;
            int grid1d = (int)((src->data_size + block1d - 1) / block1d);
            kernel_negative<<<grid1d, block1d>>>(current_in, current_out, src->data_size);
            cudaDeviceSynchronize();
            std::cout << "Step " << (step + 1) << ": Negative Inversion";
        } else {
            cudaMemcpy(current_out, current_in, img_bytes, cudaMemcpyDeviceToDevice);
            std::cout << "Step " << (step + 1) << ": Identity Pass";
        }

        cudaEventRecord(step_stop);
        cudaEventSynchronize(step_stop);

        float step_ms = 0.0f;
        cudaEventElapsedTime(&step_ms, step_start, step_stop);
        std::cout << " -> " << step_ms << " ms" << std::endl;

        cudaEventDestroy(step_start);
        cudaEventDestroy(step_stop);

        uint8_t* temp = current_in;
        current_in = current_out;
        current_out = temp;
    }

    cudaMemcpy(dst->data, current_in, img_bytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop_total);
    cudaEventSynchronize(stop_total);

    float total_ms = 0.0f;
    cudaEventElapsedTime(&total_ms, start_total, stop_total);

    std::cout << "-------------------------------------------------" << std::endl;
    std::cout << "Total Pipeline Processing Time (inc. Transfers): " << total_ms << " ms" << std::endl;
    std::cout << "-------------------------------------------------\n" << std::endl;

    if (metrics) {
        metrics->gpu_total_time_ms = (double)total_ms;
    }

    cudaEventDestroy(start_total);
    cudaEventDestroy(stop_total);

    cudaFree(d_buf1);
    cudaFree(d_buf2);
    return true;
#else
    return false;
#endif
}

