#include "../include/processing.h"

bool cpu_gaussian_blur(const Image* src, Image* dst, int ksize, float sigma) {
    if (!src || !dst || !src->data || !dst->data) {
        return false;
    }
    if (ksize <= 0 || (ksize % 2) == 0 || sigma <= 0.0f) {
        return false;
    }

    KernelMatrix kernel = create_gaussian_kernel_2d(ksize, sigma);
    if (!kernel.weights) {
        return false;
    }

    cpu_convolve_2d_raw(src, dst, kernel.weights, ksize);

    free_kernel_matrix(&kernel);
    return true;
}

bool cpu_gaussian_separable(const Image* src, Image* dst, int ksize, float sigma) {
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

    cpu_convolve_1d_separable(src, dst, kernel_1d, ksize);

    std::free(kernel_1d);
    return true;
}

