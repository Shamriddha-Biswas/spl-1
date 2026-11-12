#include "../include/processing.h"

bool cpu_contrast_stretching(const Image* src, Image* dst) {
    if (!src || !dst || !src->data || !dst->data) {
        return false;
    }

    int c_count = src->channels;
    uint64_t total_pixels = src->pixel_count;

    for (int c = 0; c < c_count; ++c) {
        uint8_t min_val = 255;
        uint8_t max_val = 0;

        for (uint64_t i = 0; i < total_pixels; ++i) {
            uint8_t val = src->data[i * c_count + c];
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
        }

        if (max_val <= min_val) {
            for (uint64_t i = 0; i < total_pixels; ++i) {
                dst->data[i * c_count + c] = src->data[i * c_count + c];
            }
            continue;
        }

        float scale = 255.0f / (float)(max_val - min_val);

        for (uint64_t i = 0; i < total_pixels; ++i) {
            uint8_t val = src->data[i * c_count + c];
            int stretched = (int)std::round(((float)val - (float)min_val) * scale);
            dst->data[i * c_count + c] = manual_clamp_u8(stretched);
        }
    }

    return true;
}

bool cpu_histogram_equalization(const Image* src, Image* dst) {
    if (!src || !dst || !src->data || !dst->data) {
        return false;
    }

    uint32_t hist_r[256];
    uint32_t hist_g[256];
    uint32_t hist_b[256];

    if (!cpu_calculate_histogram(src, hist_r, hist_g, hist_b)) {
        return false;
    }

    uint32_t* hists[3] = { hist_r, hist_g, hist_b };
    uint8_t lut[3][256];
    uint64_t total_pixels = src->pixel_count;

    for (int c = 0; c < src->channels; ++c) {
        uint32_t cdf[256];
        cdf[0] = hists[c][0];
        for (int i = 1; i < 256; ++i) {
            cdf[i] = cdf[i - 1] + hists[c][i];
        }

        uint32_t cdf_min = 0;
        for (int i = 0; i < 256; ++i) {
            if (cdf[i] > 0) {
                cdf_min = cdf[i];
                break;
            }
        }

        if (total_pixels <= cdf_min) {
            for (int i = 0; i < 256; ++i) lut[c][i] = (uint8_t)i;
            continue;
        }

        float denom = (float)(total_pixels - cdf_min);
        for (int i = 0; i < 256; ++i) {
            if (cdf[i] < cdf_min) {
                lut[c][i] = 0;
            } else {
                int eq = (int)std::round(((float)(cdf[i] - cdf_min) / denom) * 255.0f);
                lut[c][i] = manual_clamp_u8(eq);
            }
        }
    }

    for (uint64_t i = 0; i < total_pixels; ++i) {
        for (int c = 0; c < src->channels; ++c) {
            uint8_t in_val = src->data[i * src->channels + c];
            dst->data[i * src->channels + c] = lut[c][in_val];
        }
    }

    return true;
}

bool cpu_auto_enhance(const Image* src, Image* dst) {
    if (!src || !dst || !src->data || !dst->data) {
        return false;
    }

    double sum = 0.0;
    uint64_t total_samples = src->pixel_count * (uint64_t)src->channels;

    for (uint64_t i = 0; i < total_samples; ++i) {
        sum += (double)src->data[i];
    }
    double mean = sum / (double)total_samples;

    double variance_sum = 0.0;
    for (uint64_t i = 0; i < total_samples; ++i) {
        double diff = (double)src->data[i] - mean;
        variance_sum += diff * diff;
    }
    double stddev = std::sqrt(variance_sum / (double)total_samples);

    if (stddev < 35.0) {
        return cpu_histogram_equalization(src, dst);
    } else {
        return cpu_contrast_stretching(src, dst);
    }
}

