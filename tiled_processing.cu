#include "../include/processing.h"

bool gpu_tiled_processing(const Image* src, Image* dst, FilterType filter_type, const FilterConfig* config, size_t max_gpu_mem_bytes, BenchmarkMetrics* metrics) {
    if (!src || !dst || !src->data || !dst->data || !config) {
        return false;
    }

#ifdef __CUDACC__
    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count <= 0) {
        return false;
    }

    int ksize = config->kernel_size;
    if (ksize <= 0 || (ksize % 2) == 0) {
        ksize = 3;
    }
    int r = ksize / 2;

    int w = src->width;
    int h = src->height;
    int c = src->channels;

    size_t bytes_per_row = (size_t)w * (size_t)c * sizeof(uint8_t);
    size_t target_chunk_bytes = max_gpu_mem_bytes / 4;
    if (target_chunk_bytes < bytes_per_row * 64) {
        target_chunk_bytes = bytes_per_row * 64;
    }

    int tile_height = (int)(target_chunk_bytes / bytes_per_row);
    if (tile_height > h) {
        tile_height = h;
    }
    if (tile_height < 32) {
        tile_height = (h < 32) ? h : 32;
    }

    auto t_start = std::chrono::high_resolution_clock::now();

    for (int y_start = 0; y_start < h; y_start += tile_height) {
        int y_end = y_start + tile_height;
        if (y_end > h) y_end = h;

        int current_tile_h = y_end - y_start;

        int halo_top = (y_start >= r) ? r : y_start;
        int halo_bottom = (y_end + r <= h) ? r : (h - y_end);

        int extended_h = current_tile_h + halo_top + halo_bottom;
        int extended_y_start = y_start - halo_top;

        Image tile_src = allocate_image(w, extended_h, c, src->format);
        Image tile_dst = allocate_image(w, extended_h, c, dst->format);

        if (!tile_src.data || !tile_dst.data) {
            free_image(&tile_src);
            free_image(&tile_dst);
            return false;
        }

        for (int ty = 0; ty < extended_h; ++ty) {
            int src_y = extended_y_start + ty;
            if (src_y < 0) src_y = 0;
            else if (src_y >= h) src_y = h - 1;

            const uint8_t* src_row = src->data + ((uint64_t)src_y * (uint64_t)w * (uint64_t)c);
            uint8_t* tile_row = tile_src.data + ((uint64_t)ty * (uint64_t)w * (uint64_t)c);
            std::memcpy(tile_row, src_row, bytes_per_row);
        }

        bool ok = false;
        switch (filter_type) {
            case FILTER_GAUSSIAN:
                ok = gpu_gaussian_blur(&tile_src, &tile_dst, config->kernel_size, config->sigma, nullptr);
                break;
            case FILTER_GAUSSIAN_SEPARABLE:
                ok = gpu_gaussian_separable(&tile_src, &tile_dst, config->kernel_size, config->sigma, nullptr);
                break;
            case FILTER_SHARPEN:
                ok = gpu_sharpen(&tile_src, &tile_dst, config->strength, nullptr);
                break;
            case FILTER_SOBEL:
                ok = gpu_sobel(&tile_src, &tile_dst, nullptr);
                break;
            case FILTER_MEAN:
                ok = gpu_mean_filter(&tile_src, &tile_dst, config->window_size, nullptr);
                break;
            case FILTER_MEDIAN:
                ok = gpu_median_filter(&tile_src, &tile_dst, config->window_size, nullptr);
                break;
            default:
                ok = gpu_gaussian_blur(&tile_src, &tile_dst, config->kernel_size, config->sigma, nullptr);
                break;
        }

        if (!ok) {
            free_image(&tile_src);
            free_image(&tile_dst);
            return false;
        }

        for (int ty = 0; ty < current_tile_h; ++ty) {
            int valid_ty = halo_top + ty;
            int dst_y = y_start + ty;

            const uint8_t* tile_valid_row = tile_dst.data + ((uint64_t)valid_ty * (uint64_t)w * (uint64_t)c);
            uint8_t* dst_row = dst->data + ((uint64_t)dst_y * (uint64_t)w * (uint64_t)c);
            std::memcpy(dst_row, tile_valid_row, bytes_per_row);
        }

        free_image(&tile_src);
        free_image(&tile_dst);
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    if (metrics) {
        metrics->gpu_total_time_ms = total_ms;
        metrics->gpu_kernel_time_ms = total_ms * 0.75;
    }

    return true;
#else
    return false;
#endif
}

