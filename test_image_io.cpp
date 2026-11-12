#include "../include/image.h"
#include <cassert>

int main() {
    std::cout << "[TEST] Running Image I/O Verification Test..." << std::endl;

    Image pgm_out = allocate_image(10, 10, 1, FORMAT_PGM);
    for (int y = 0; y < 10; ++y) {
        for (int x = 0; x < 10; ++x) {
            set_pixel(&pgm_out, x, y, 0, (uint8_t)(x * 20 + y * 5));
        }
    }
    assert(write_pgm("test_temp.pgm", &pgm_out));

    Image pgm_in;
    assert(read_pgm("test_temp.pgm", &pgm_in));
    assert(pgm_in.width == 10 && pgm_in.height == 10 && pgm_in.channels == 1);
    for (uint64_t i = 0; i < pgm_out.data_size; ++i) {
        assert(pgm_in.data[i] == pgm_out.data[i]);
    }
    free_image(&pgm_out);
    free_image(&pgm_in);
    std::remove("test_temp.pgm");

    Image ppm_out = allocate_image(7, 5, 3, FORMAT_PPM);
    for (int y = 0; y < 5; ++y) {
        for (int x = 0; x < 7; ++x) {
            set_pixel(&ppm_out, x, y, 0, (uint8_t)(x * 30));
            set_pixel(&ppm_out, x, y, 1, (uint8_t)(y * 40));
            set_pixel(&ppm_out, x, y, 2, (uint8_t)(150));
        }
    }
    assert(write_ppm("test_temp.ppm", &ppm_out));

    Image ppm_in;
    assert(read_ppm("test_temp.ppm", &ppm_in));
    assert(ppm_in.width == 7 && ppm_in.height == 5 && ppm_in.channels == 3);
    for (uint64_t i = 0; i < ppm_out.data_size; ++i) {
        assert(ppm_in.data[i] == ppm_out.data[i]);
    }
    free_image(&ppm_out);
    free_image(&ppm_in);
    std::remove("test_temp.ppm");

    Image bmp_out = allocate_image(5, 7, 3, FORMAT_BMP);
    for (int y = 0; y < 7; ++y) {
        for (int x = 0; x < 5; ++x) {
            set_pixel(&bmp_out, x, y, 0, (uint8_t)(x * 40));
            set_pixel(&bmp_out, x, y, 1, (uint8_t)(y * 30));
            set_pixel(&bmp_out, x, y, 2, (uint8_t)(x + y * 10));
        }
    }
    assert(write_bmp("test_temp.bmp", &bmp_out));

    Image bmp_in;
    assert(read_bmp("test_temp.bmp", &bmp_in));
    assert(bmp_in.width == 5 && bmp_in.height == 7 && bmp_in.channels == 3);
    for (int y = 0; y < 7; ++y) {
        for (int x = 0; x < 5; ++x) {
            for (int c = 0; c < 3; ++c) {
                assert(get_pixel_clamp(&bmp_in, x, y, c) == get_pixel_clamp(&bmp_out, x, y, c));
            }
        }
    }
    free_image(&bmp_out);
    free_image(&bmp_in);
    std::remove("test_temp.bmp");

    Image fake;
    assert(!read_image("non_existent_file_xyz.ppm", &fake));

    std::cout << "[PASS] PGM, PPM, and BMP manual file I/O round-trip verified successfully!" << std::endl;
    return 0;
}

