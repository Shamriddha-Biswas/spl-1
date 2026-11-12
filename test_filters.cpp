#include "../include/processing.h"
#include <cassert>

int main() {
    std::cout << "[TEST] Running CPU Filters Algorithmic Verification Test..." << std::endl;

    Image step_img = allocate_image(8, 8, 1, FORMAT_PGM);
    for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) {
            set_pixel(&step_img, x, y, 0, (x < 4 ? 50 : 200));
        }
    }

    Image sobel_out = allocate_image(8, 8, 1, FORMAT_PGM);
    assert(cpu_sobel(&step_img, &sobel_out));
    assert(get_pixel_clamp(&sobel_out, 3, 4, 0) > 100);
    assert(get_pixel_clamp(&sobel_out, 0, 4, 0) == 0);
    free_image(&sobel_out);

    Image noisy_img = allocate_image(5, 5, 1, FORMAT_PGM);
    for (int i = 0; i < 25; ++i) noisy_img.data[i] = 40;
    set_pixel(&noisy_img, 2, 2, 0, 255);

    Image median_out = allocate_image(5, 5, 1, FORMAT_PGM);
    assert(cpu_median_filter(&noisy_img, &median_out, 3));
    assert(get_pixel_clamp(&median_out, 2, 2, 0) == 40);
    free_image(&noisy_img);
    free_image(&median_out);

    Image g2d_out = allocate_image(8, 8, 1, FORMAT_PGM);
    Image g1d_out = allocate_image(8, 8, 1, FORMAT_PGM);
    assert(cpu_gaussian_blur(&step_img, &g2d_out, 3, 1.0f));
    assert(cpu_gaussian_separable(&step_img, &g1d_out, 3, 1.0f));

    BenchmarkMetrics g_metrics;
    std::memset(&g_metrics, 0, sizeof(g_metrics));
    calculate_metrics(&g2d_out, &g1d_out, &g_metrics);
    assert(g_metrics.mae < 1.0);
    free_image(&g2d_out);
    free_image(&g1d_out);

    uint32_t hr[256], hg[256], hb[256];
    assert(cpu_calculate_histogram(&step_img, hr, hg, hb));
    uint32_t total_hist = 0;
    for (int i = 0; i < 256; ++i) total_hist += hr[i];
    assert(total_hist == 64);

    Image neg1 = allocate_image(8, 8, 1, FORMAT_PGM);
    Image neg2 = allocate_image(8, 8, 1, FORMAT_PGM);
    assert(cpu_negative(&step_img, &neg1));
    assert(cpu_negative(&neg1, &neg2));
    for (int i = 0; i < 64; ++i) {
        assert(neg2.data[i] == step_img.data[i]);
    }
    free_image(&neg1);
    free_image(&neg2);

    free_image(&step_img);
    std::cout << "[PASS] All CPU image processing filters passed algorithmic assertions!" << std::endl;
    return 0;
}

