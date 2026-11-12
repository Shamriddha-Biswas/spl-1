#include "../include/processing.h"

bool cpu_sobel(const Image* src, Image* dst) {
    if (!src || !dst || !src->data || !dst->data) {
        return false;
    }

    int w = src->width;
    int h = src->height;
    int c_count = src->channels;

    const int sobel_x[9] = {
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1
    };

    const int sobel_y[9] = {
        -1, -2, -1,
         0,  0,  0,
         1,  2,  1
    };

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            for (int c = 0; c < c_count; ++c) {
                int gx = 0;
                int gy = 0;

                for (int ky = -1; ky <= 1; ++ky) {
                    int py = y + ky;
                    if (py < 0) py = 0;
                    else if (py >= h) py = h - 1;

                    for (int kx = -1; kx <= 1; ++kx) {
                        int px = x + kx;
                        if (px < 0) px = 0;
                        else if (px >= w) px = w - 1;

                        int weight_x = sobel_x[(ky + 1) * 3 + (kx + 1)];
                        int weight_y = sobel_y[(ky + 1) * 3 + (kx + 1)];

                        uint64_t idx = ((uint64_t)py * (uint64_t)w + (uint64_t)px) * (uint64_t)c_count + (uint64_t)c;
                        int val = (int)src->data[idx];

                        gx += val * weight_x;
                        gy += val * weight_y;
                    }
                }

                float mag = manual_sqrt((float)(gx * gx + gy * gy));
                int out_val = (int)std::round(mag);
                set_pixel(dst, x, y, c, manual_clamp_u8(out_val));
            }
        }
    }

    return true;
}

