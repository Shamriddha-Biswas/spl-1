#include "../include/processing.h"

bool cpu_sharpen(const Image* src, Image* dst, float strength) {
    if (!src || !dst || !src->data || !dst->data) {
        return false;
    }
    if (strength < 0.0f) {
        strength = 1.0f;
    }

    float kernel[9] = {
        0.0f, -strength, 0.0f,
        -strength, 1.0f + 4.0f * strength, -strength,
        0.0f, -strength, 0.0f
    };

    cpu_convolve_2d_raw(src, dst, kernel, 3);
    return true;
}

