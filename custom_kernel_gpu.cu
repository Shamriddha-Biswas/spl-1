#include "../include/processing.h"

bool gpu_custom_convolution(const Image* src, Image* dst, const KernelMatrix* kernel, bool normalize, BenchmarkMetrics* metrics) {
    if (!src || !dst || !src->data || !dst->data) {
        return false;
    }
    if (!validate_kernel_matrix(kernel)) {
        return false;
    }

    int ksize = kernel->width;
    std::vector<float> norm_weights(ksize * ksize);

    float sum = 0.0f;
    for (int i = 0; i < ksize * ksize; ++i) {
        sum += kernel->weights[i];
    }

    if (normalize && manual_abs(sum) > 1e-6f) {
        float inv_sum = 1.0f / sum;
        for (int i = 0; i < ksize * ksize; ++i) {
            norm_weights[i] = kernel->weights[i] * inv_sum;
        }
    } else {
        for (int i = 0; i < ksize * ksize; ++i) {
            norm_weights[i] = kernel->weights[i];
        }
    }

    return gpu_convolve_2d_raw(src, dst, norm_weights.data(), ksize, metrics);
}

