#include "../include/processing.h"

bool gpu_gaussian_separable(const Image* src, Image* dst, int ksize, float sigma, BenchmarkMetrics* metrics) {
    if (!src || !dst || !src->data || !dst->data) {
        return false;
    }
    if (ksize <= 0 || (ksize % 2) == 0 || sigma <= 0.0f) {
        return false;
    }

    float* kernel_1d = create_gaussian_kernel_1d(ksize, sigma);
    if (!kernel_1d) {
        return false;
    }

    bool success = gpu_convolve_1d_separable_raw(src, dst, kernel_1d, ksize, metrics);

    std::free(kernel_1d);
    return success;
}

