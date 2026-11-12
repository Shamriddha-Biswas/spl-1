#include "../include/processing.h"
#include <cassert>

int main() {
    std::cout << "[TEST] Running Matrix Operations & Indexing Test..." << std::endl;

    assert(manual_clamp_u8(-10) == 0);
    assert(manual_clamp_u8(260) == 255);
    assert(manual_clamp_u8(128) == 128);

    KernelMatrix k2d = create_gaussian_kernel_2d(3, 1.0f);
    assert(k2d.weights != nullptr);
    assert(k2d.width == 3 && k2d.height == 3);

    float sum2d = 0.0f;
    for (int i = 0; i < 9; ++i) sum2d += k2d.weights[i];
    assert(manual_abs(sum2d - 1.0f) < 1e-4f);
    free_kernel_matrix(&k2d);

    float* k1d = create_gaussian_kernel_1d(5, 1.5f);
    assert(k1d != nullptr);
    float sum1d = 0.0f;
    for (int i = 0; i < 5; ++i) sum1d += k1d[i];
    assert(manual_abs(sum1d - 1.0f) < 1e-4f);
    std::free(k1d);

    Image src = allocate_image(3, 3, 1, FORMAT_PGM);
    for (int i = 0; i < 9; ++i) src.data[i] = 100;
    Image dst = allocate_image(3, 3, 1, FORMAT_PGM);

    float identity_kernel[9] = {
        0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f
    };

    cpu_convolve_2d_raw(&src, &dst, identity_kernel, 3);
    for (int i = 0; i < 9; ++i) {
        assert(dst.data[i] == 100);
    }

    free_image(&src);
    free_image(&dst);

    std::cout << "[PASS] Matrix operations and boundary indexing verified successfully!" << std::endl;
    return 0;
}

