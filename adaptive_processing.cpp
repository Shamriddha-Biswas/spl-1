#include "../include/processing.h"

bool adaptive_process(const Image* src, Image* dst, FilterType filter_type, const FilterConfig* config, BenchmarkMetrics* metrics, bool developer_mode) {
    if (!src || !dst || !src->data || !dst->data || !config) {
        return false;
    }

    HardwareProfile profile;
    bool has_gpu = query_gpu_hardware(&profile);

    if (!has_gpu || profile.device_count <= 0) {
        if (developer_mode) {
            std::cout << "[Adaptive Strategy]: CPU Sequential (No CUDA GPU Available)" << std::endl;
        }
        switch (filter_type) {
            case FILTER_GAUSSIAN:
                return cpu_gaussian_blur(src, dst, config->kernel_size, config->sigma);
            case FILTER_GAUSSIAN_SEPARABLE:
                return cpu_gaussian_separable(src, dst, config->kernel_size, config->sigma);
            case FILTER_SHARPEN:
                return cpu_sharpen(src, dst, config->strength);
            case FILTER_SOBEL:
                return cpu_sobel(src, dst);
            case FILTER_MEAN:
                return cpu_mean_filter(src, dst, config->window_size);
            case FILTER_MEDIAN:
                return cpu_median_filter(src, dst, config->window_size);
            case FILTER_ENHANCE_AUTO:
                return cpu_auto_enhance(src, dst);
            case FILTER_CONTRAST_STRETCH:
                return cpu_contrast_stretching(src, dst);
            case FILTER_HISTOGRAM_EQUALIZATION:
                return cpu_histogram_equalization(src, dst);
            default:
                return cpu_creative_effect(src, dst, filter_type);
        }
    }

    size_t required_bytes = src->data_size * 3;
    size_t free_mem = profile.free_memory;
    bool fits_in_gpu = validate_gpu_memory_headroom(required_bytes, free_mem, 0.25);

    if (!fits_in_gpu) {
        if (developer_mode) {
            std::cout << "[Adaptive Strategy]: GPU Tiled Processing (Image exceeds VRAM headroom)" << std::endl;
            std::cout << "  Required Memory: " << (required_bytes / (1024 * 1024)) << " MB, Free VRAM: " 
                      << (free_mem / (1024 * 1024)) << " MB" << std::endl;
        }
        return gpu_tiled_processing(src, dst, filter_type, config, free_mem / 2, metrics);
    }

    if (filter_type == FILTER_GAUSSIAN) {
        if (config->kernel_size <= 7 && config->quality == QUALITY_HIGH) {
            if (developer_mode) {
                std::cout << "[Adaptive Strategy]: GPU Shared Memory Optimized 2D Convolution" << std::endl;
            }
            return gpu_gaussian_shared_memory(src, dst, config->kernel_size, config->sigma, metrics);
        } else if (config->kernel_size > 7 || config->quality == QUALITY_FAST) {
            if (developer_mode) {
                std::cout << "[Adaptive Strategy]: GPU Separable 1D Horizontal/Vertical Convolution" << std::endl;
            }
            return gpu_gaussian_separable(src, dst, config->kernel_size, config->sigma, metrics);
        } else {
            if (developer_mode) {
                std::cout << "[Adaptive Strategy]: GPU Standard 2D Grid Convolution" << std::endl;
            }
            return gpu_gaussian_blur(src, dst, config->kernel_size, config->sigma, metrics);
        }
    }

    if (filter_type == FILTER_GAUSSIAN_SEPARABLE) {
        if (developer_mode) {
            std::cout << "[Adaptive Strategy]: GPU Separable 1D Horizontal/Vertical Convolution" << std::endl;
        }
        return gpu_gaussian_separable(src, dst, config->kernel_size, config->sigma, metrics);
    }

    if (filter_type == FILTER_SHARPEN) {
        if (developer_mode) {
            std::cout << "[Adaptive Strategy]: GPU Standard 2D Sharpening Convolution" << std::endl;
        }
        return gpu_sharpen(src, dst, config->strength, metrics);
    }

    if (filter_type == FILTER_SOBEL) {
        if (developer_mode) {
            std::cout << "[Adaptive Strategy]: GPU Conjoint Sobel Gradient Kernel" << std::endl;
        }
        return gpu_sobel(src, dst, metrics);
    }

    if (filter_type == FILTER_MEAN) {
        if (developer_mode) {
            std::cout << "[Adaptive Strategy]: GPU Box Neighborhood Filter" << std::endl;
        }
        return gpu_mean_filter(src, dst, config->window_size, metrics);
    }

    if (filter_type == FILTER_MEDIAN) {
        if (developer_mode) {
            std::cout << "[Adaptive Strategy]: GPU Register-Based Insertion Sort Median Filter" << std::endl;
        }
        return gpu_median_filter(src, dst, config->window_size, metrics);
    }

    if (filter_type == FILTER_ENHANCE_AUTO) {
        if (developer_mode) {
            std::cout << "[Adaptive Strategy]: GPU Auto Statistical Enhancement" << std::endl;
        }
        return gpu_auto_enhance(src, dst, metrics);
    }

    if (filter_type == FILTER_CONTRAST_STRETCH) {
        if (developer_mode) {
            std::cout << "[Adaptive Strategy]: GPU LUT Dynamic Contrast Stretching" << std::endl;
        }
        return gpu_contrast_stretching(src, dst, metrics);
    }

    if (filter_type == FILTER_HISTOGRAM_EQUALIZATION) {
        if (developer_mode) {
            std::cout << "[Adaptive Strategy]: GPU Atomic Histogram Equalization" << std::endl;
        }
        return gpu_histogram_equalization(src, dst, metrics);
    }

    if (developer_mode) {
        std::cout << "[Adaptive Strategy]: GPU Creative Artistic Pipeline" << std::endl;
    }
    return gpu_creative_effect(src, dst, filter_type, metrics);
}

