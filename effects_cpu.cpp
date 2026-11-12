#include "../include/processing.h"

bool cpu_grayscale(const Image* src, Image* dst) {
    if (!src || !dst || !src->data || !dst->data) {
        return false;
    }

    uint64_t total = src->pixel_count;
    int channels = src->channels;

    if (channels == 1) {
        std::memcpy(dst->data, src->data, src->data_size);
        return true;
    }

    for (uint64_t i = 0; i < total; ++i) {
        uint8_t r = src->data[i * channels + 0];
        uint8_t g = src->data[i * channels + 1];
        uint8_t b = src->data[i * channels + 2];
        uint8_t gray = (uint8_t)((299 * (uint32_t)r + 587 * (uint32_t)g + 114 * (uint32_t)b) / 1000);
        for (int c = 0; c < dst->channels; ++c) {
            dst->data[i * dst->channels + c] = gray;
        }
    }
    return true;
}

bool cpu_negative(const Image* src, Image* dst) {
    if (!src || !dst || !src->data || !dst->data) {
        return false;
    }
    uint64_t total_bytes = src->data_size;
    for (uint64_t i = 0; i < total_bytes; ++i) {
        dst->data[i] = 255 - src->data[i];
    }
    return true;
}

bool cpu_box_blur(const Image* src, Image* dst, int ksize) {
    return cpu_mean_filter(src, dst, ksize);
}

bool cpu_emboss(const Image* src, Image* dst) {
    if (!src || !dst || !src->data || !dst->data) {
        return false;
    }

    float kernel[9] = {
        -2.0f, -1.0f,  0.0f,
        -1.0f,  1.0f,  1.0f,
         0.0f,  1.0f,  2.0f
    };

    int r = 1;
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

                        float weight = kernel[(ky + r) * 3 + (kx + r)];
                        uint64_t idx = ((uint64_t)py * (uint64_t)w + (uint64_t)px) * (uint64_t)c_count + (uint64_t)c;
                        sum += (float)src->data[idx] * weight;
                    }
                }
                int out_val = (int)std::round(sum + 128.0f);
                set_pixel(dst, x, y, c, manual_clamp_u8(out_val));
            }
        }
    }
    return true;
}

bool cpu_outline(const Image* src, Image* dst) {
    if (!src || !dst || !src->data || !dst->data) {
        return false;
    }

    float kernel[9] = {
        -1.0f, -1.0f, -1.0f,
        -1.0f,  8.0f, -1.0f,
        -1.0f, -1.0f, -1.0f
    };

    cpu_convolve_2d_raw(src, dst, kernel, 3);
    for (uint64_t i = 0; i < dst->data_size; ++i) {
        dst->data[i] = 255 - dst->data[i];
    }
    return true;
}

bool cpu_soften(const Image* src, Image* dst) {
    if (!src || !dst || !src->data || !dst->data) {
        return false;
    }

    Image blurred = allocate_image(src->width, src->height, src->channels, src->format);
    if (!blurred.data) {
        return false;
    }

    if (!cpu_gaussian_blur(src, &blurred, 5, 1.5f)) {
        free_image(&blurred);
        return false;
    }

    uint64_t total = src->data_size;
    for (uint64_t i = 0; i < total; ++i) {
        float blended = 0.65f * (float)src->data[i] + 0.35f * (float)blurred.data[i];
        dst->data[i] = manual_clamp_u8((int)std::round(blended));
    }

    free_image(&blurred);
    return true;
}

bool cpu_highpass(const Image* src, Image* dst, int ksize, float sigma) {
    if (!src || !dst || !src->data || !dst->data) {
        return false;
    }

    Image blurred = allocate_image(src->width, src->height, src->channels, src->format);
    if (!blurred.data) {
        return false;
    }

    if (!cpu_gaussian_blur(src, &blurred, ksize, sigma)) {
        free_image(&blurred);
        return false;
    }

    uint64_t total = src->data_size;
    for (uint64_t i = 0; i < total; ++i) {
        int hp = (int)src->data[i] - (int)blurred.data[i] + 128;
        dst->data[i] = manual_clamp_u8(hp);
    }

    free_image(&blurred);
    return true;
}

bool cpu_edge_enhance(const Image* src, Image* dst, float strength) {
    if (!src || !dst || !src->data || !dst->data) {
        return false;
    }

    Image edges = allocate_image(src->width, src->height, src->channels, src->format);
    if (!edges.data) {
        return false;
    }

    if (!cpu_sobel(src, &edges)) {
        free_image(&edges);
        return false;
    }

    uint64_t total = src->data_size;
    for (uint64_t i = 0; i < total; ++i) {
        float val = (float)src->data[i] + strength * (float)edges.data[i];
        dst->data[i] = manual_clamp_u8((int)std::round(val));
    }

    free_image(&edges);
    return true;
}

bool cpu_edge_art(const Image* src, Image* dst) {
    if (!src || !dst || !src->data || !dst->data) {
        return false;
    }

    Image edges = allocate_image(src->width, src->height, src->channels, src->format);
    if (!edges.data) {
        return false;
    }

    if (!cpu_sobel(src, &edges)) {
        free_image(&edges);
        return false;
    }

    uint64_t total = src->data_size;
    for (uint64_t i = 0; i < total; ++i) {
        uint8_t edge_val = edges.data[i];
        uint8_t orig_val = src->data[i];
        uint8_t posterized = (orig_val / 32) * 32;
        if (edge_val > 60) {
            dst->data[i] = 0;
        } else {
            dst->data[i] = posterized;
        }
    }

    free_image(&edges);
    return true;
}

bool cpu_pencil_sketch(const Image* src, Image* dst) {
    if (!src || !dst || !src->data || !dst->data) {
        return false;
    }

    Image gray = allocate_image(src->width, src->height, src->channels, src->format);
    Image inverted = allocate_image(src->width, src->height, src->channels, src->format);
    Image blurred = allocate_image(src->width, src->height, src->channels, src->format);

    if (!gray.data || !inverted.data || !blurred.data) {
        free_image(&gray);
        free_image(&inverted);
        free_image(&blurred);
        return false;
    }

    cpu_grayscale(src, &gray);
    cpu_negative(&gray, &inverted);
    cpu_gaussian_blur(&inverted, &blurred, 9, 3.0f);

    uint64_t total = src->data_size;
    for (uint64_t i = 0; i < total; ++i) {
        uint32_t g = (uint32_t)gray.data[i];
        uint32_t b = (uint32_t)blurred.data[i];
        if (b >= 255) {
            dst->data[i] = 255;
        } else {
            uint32_t val = (g * 256) / (256 - b);
            dst->data[i] = (val > 255) ? 255 : (uint8_t)val;
        }
    }

    free_image(&gray);
    free_image(&inverted);
    free_image(&blurred);
    return true;
}

bool cpu_creative_effect(const Image* src, Image* dst, FilterType effect) {
    switch (effect) {
        case FILTER_GRAYSCALE:
            return cpu_grayscale(src, dst);
        case FILTER_NEGATIVE:
            return cpu_negative(src, dst);
        case FILTER_BOX_BLUR:
            return cpu_box_blur(src, dst, 5);
        case FILTER_EMBOSS:
            return cpu_emboss(src, dst);
        case FILTER_OUTLINE:
            return cpu_outline(src, dst);
        case FILTER_SOFTEN:
            return cpu_soften(src, dst);
        case FILTER_HIGHPASS:
            return cpu_highpass(src, dst, 5, 1.5f);
        case FILTER_EDGE_ENHANCE:
            return cpu_edge_enhance(src, dst, 0.5f);
        case FILTER_EDGE_ART:
            return cpu_edge_art(src, dst);
        case FILTER_SKETCH:
            return cpu_pencil_sketch(src, dst);
        default:
            return cpu_grayscale(src, dst);
    }
}

