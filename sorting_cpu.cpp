#include "../include/processing.h"

void cpu_sort_array_u8(uint8_t* arr, int n) {
    manual_sort_asc(arr, n);
}

uint8_t cpu_find_median_u8(uint8_t* arr, int n) {
    if (!arr || n <= 0) return 0;
    manual_sort_asc(arr, n);
    return arr[n / 2];
}

