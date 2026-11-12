#include "../include/processing.h"

void cpu_convolve_2d_raw(const Image* src, Image* dst, const float* kernel, int ksize) {
    if (!src || !dst || !src->data || !dst->data || !kernel || ksize <= 0) {
        return;
    }

    int r = ksize / 2;
    int w = src->width;
    int h = src->height;
    int c_count = src->channels;

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            for (int c = 0; c < c_count; ++c) {
                float sum = 0.0f;
                for (int ky = -r; ky <= r; ++ky) {
                    int py = y + ky;
                    if (py < 0) py = 0;
                    else if (py >= h) py = h - 1;

                    for (int kx = -r; kx <= r; ++kx) {
                        int px = x + kx;
                        if (px < 0) px = 0;
                        else if (px >= w) px = w - 1;

                        float weight = kernel[(ky + r) * ksize + (kx + r)];
                        uint64_t idx = ((uint64_t)py * (uint64_t)w + (uint64_t)px) * (uint64_t)c_count + (uint64_t)c;
                        sum += (float)src->data[idx] * weight;
                    }
                }
                int out_val = (int)std::round(sum);
                set_pixel(dst, x, y, c, manual_clamp_u8(out_val));
            }
        }
    }
}

void cpu_convolve_1d_separable(const Image* src, Image* dst, const float* kernel_1d, int ksize) {
    if (!src || !dst || !src->data || !dst->data || !kernel_1d || ksize <= 0) {
        return;
    }

    int r = ksize / 2;
    int w = src->width;
    int h = src->height;
    int c_count = src->channels;

    std::vector<float> temp_buffer((size_t)w * (size_t)h * (size_t)c_count, 0.0f);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            for (int c = 0; c < c_count; ++c) {
                float sum = 0.0f;
                for (int kx = -r; kx <= r; ++kx) {
                    int px = x + kx;
                    if (px < 0) px = 0;
                    else if (px >= w) px = w - 1;

                    float weight = kernel_1d[kx + r];
                    uint64_t idx = ((uint64_t)y * (uint64_t)w + (uint64_t)px) * (uint64_t)c_count + (uint64_t)c;
                    sum += (float)src->data[idx] * weight;
                }
                uint64_t out_idx = ((uint64_t)y * (uint64_t)w + (uint64_t)x) * (uint64_t)c_count + (uint64_t)c;
                temp_buffer[out_idx] = sum;
            }
        }
    }

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            for (int c = 0; c < c_count; ++c) {
                float sum = 0.0f;
                for (int ky = -r; ky <= r; ++ky) {
                    int py = y + ky;
                    if (py < 0) py = 0;
                    else if (py >= h) py = h - 1;

                    float weight = kernel_1d[ky + r];
                    uint64_t idx = ((uint64_t)py * (uint64_t)w + (uint64_t)x) * (uint64_t)c_count + (uint64_t)c;
                    sum += temp_buffer[idx] * weight;
                }
                int out_val = (int)std::round(sum);
                set_pixel(dst, x, y, c, manual_clamp_u8(out_val));
            }
        }
    }
}

