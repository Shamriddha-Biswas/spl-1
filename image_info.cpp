#include "../include/image.h"

void print_image_info(const Image* img, const char* filepath) {
    if (!img) {
        std::cout << "Error: Invalid image pointer." << std::endl;
        return;
    }

    uint64_t file_size = 0;
    if (filepath) {
        std::ifstream file(filepath, std::ios::binary | std::ios::ate);
        if (file.is_open()) {
            file_size = (uint64_t)file.tellg();
            file.close();
        }
    }

    const char* format_str = "Unknown";
    if (img->format == FORMAT_PGM) format_str = "Binary PGM (P5)";
    else if (img->format == FORMAT_PPM) format_str = "Binary PPM (P6)";
    else if (img->format == FORMAT_BMP) format_str = "Windows 24-bit Bitmap (BMP)";

    double memory_mb = (double)img->data_size / (1024.0 * 1024.0);
    double file_mb = (double)file_size / (1024.0 * 1024.0);
    double estimated_gpu_mb = (double)(img->data_size * 3) / (1024.0 * 1024.0);

    std::cout << "\n=================================================" << std::endl;
    std::cout << "               IMAGE INFORMATION                 " << std::endl;
    std::cout << "=================================================" << std::endl;
    std::cout << "Filename            : " << (filepath ? filepath : "In-memory") << std::endl;
    std::cout << "Format              : " << format_str << std::endl;
    std::cout << "Width               : " << img->width << " pixels" << std::endl;
    std::cout << "Height              : " << img->height << " pixels" << std::endl;
    std::cout << "Resolution          : " << img->width << " x " << img->height << std::endl;
    std::cout << "Channels            : " << img->channels << (img->channels == 1 ? " (Grayscale)" : " (RGB)") << std::endl;
    std::cout << "Bits Per Channel    : " << img->bits_per_channel << " bits" << std::endl;
    std::cout << "Total Pixel Count   : " << img->pixel_count << " pixels" << std::endl;
    std::cout << "Raw Buffer Size     : " << img->data_size << " bytes (" << memory_mb << " MB)" << std::endl;
    std::cout << "On-Disk File Size   : " << file_size << " bytes (" << file_mb << " MB)" << std::endl;
    std::cout << "Estimated GPU Memory: " << (img->data_size * 3) << " bytes (" << estimated_gpu_mb << " MB)" << std::endl;
    std::cout << "=================================================\n" << std::endl;
}

