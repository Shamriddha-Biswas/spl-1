#include "../include/processing.h"

bool cpu_median_filter(const Image* src, Image* dst, int window_size) {
    if (!src || !dst || !src->data || !dst->data) {
        return false;
    }
    if (window_size <= 0 || (window_size % 2) == 0) {
        window_size = 3;
    }

    int r = window_size / 2;
    int w = src->width;
    int h = src->height;
    int c_count = src->channels;
    int area = window_size * window_size;

    std::vector<uint8_t> window_buf(area);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            for (int c = 0; c < c_count; ++c) {
                int count = 0;

                for (int ky = -r; ky <= r; ++ky) {
                    int py = y + ky;
                    if (py < 0) py = 0;
                    else if (py >= h) py = h - 1;

                    for (int kx = -r; kx <= r; ++kx) {
                        int px = x + kx;
                        if (px < 0) px = 0;
                        else if (px >= w) px = w - 1;

                        uint64_t idx = ((uint64_t)py * (uint64_t)w + (uint64_t)px) * (uint64_t)c_count + (uint64_t)c;
                        window_buf[count++] = src->data[idx];
                    }
                }

                manual_sort_asc(window_buf.data(), area);
                uint8_t median_val = window_buf[area / 2];
                set_pixel(dst, x, y, c, median_val);
            }
        }
    }

    return true;
}

