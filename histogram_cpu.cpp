#include "../include/processing.h"

bool cpu_calculate_histogram(const Image* src, uint32_t* hist_r, uint32_t* hist_g, uint32_t* hist_b) {
    if (!src || !src->data || !hist_r || !hist_g || !hist_b) {
        return false;
    }

    std::memset(hist_r, 0, 256 * sizeof(uint32_t));
    std::memset(hist_g, 0, 256 * sizeof(uint32_t));
    std::memset(hist_b, 0, 256 * sizeof(uint32_t));

    uint64_t total_pixels = src->pixel_count;
    int channels = src->channels;

    if (channels == 1) {
        for (uint64_t i = 0; i < total_pixels; ++i) {
            uint8_t val = src->data[i];
            hist_r[val]++;
            hist_g[val]++;
            hist_b[val]++;
        }
    } else {
        for (uint64_t i = 0; i < total_pixels; ++i) {
            uint8_t r = src->data[i * channels + 0];
            uint8_t g = src->data[i * channels + 1];
            uint8_t b = src->data[i * channels + 2];
            hist_r[r]++;
            hist_g[g]++;
            hist_b[b]++;
        }
    }

    return true;
}

