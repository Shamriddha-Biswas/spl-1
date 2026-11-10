
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define WIDTH 5
#define HEIGHT 5
#define KSIZE 3

__global__ void convolution(const unsigned char* input,
                            unsigned char* output,
                            int width,
                            int height,
                            const float* kernel) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int radius = KSIZE / 2;
    float sum = 0.0f;

    for (int ky = -radius; ky <= radius; ky++) {
        for (int kx = -radius; kx <= radius; kx++) {

            int px = x + kx;
            int py = y + ky;

            if (px < 0) px = 0;
            if (px >= width) px = width - 1;

            if (py < 0) py = 0;
            if (py >= height) py = height - 1;

            int imageIndex = py * width + px;
            int kernelIndex = (ky + radius) * KSIZE + (kx + radius);

            sum += input[imageIndex] * kernel[kernelIndex];
        }
    }

    int result = (int)(sum + 0.5f);

    if (result > 255) result = 255;
    if (result < 0) result = 0;

    output[y * width + x] = (unsigned char)result;
}

void printImage(const char* title, unsigned char* image) {

    printf("\n%s\n", title);

    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            printf("%4d", image[y * WIDTH + x]);
        }
        printf("\n");
    }
}

int main() {

    unsigned char h_input[WIDTH * HEIGHT] = {
        10, 20, 30, 40, 50,
        60, 70, 80, 90, 100,
        110, 120, 130, 140, 150,
        160, 170, 180, 190, 200,
        210, 220, 230, 240, 250
    };

    unsigned char h_output[WIDTH * HEIGHT];

    float h_kernel[KSIZE * KSIZE] = {
        1.0f/9, 1.0f/9, 1.0f/9,
        1.0f/9, 1.0f/9, 1.0f/9,
        1.0f/9, 1.0f/9, 1.0f/9
    };

    unsigned char *d_input, *d_output;
    float *d_kernel;

    int imageBytes = WIDTH * HEIGHT * sizeof(unsigned char);
    int kernelBytes = KSIZE * KSIZE * sizeof(float);

    int deviceCount = 0;

    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
        printf("No CUDA GPU available!\n");
        return 1;
    }

    printf("CUDA Image Convolution Demo\n");
    printf("GPU detected: %d\n", deviceCount);

    cudaMalloc((void**)&d_input, imageBytes);
    cudaMalloc((void**)&d_output, imageBytes);
    cudaMalloc((void**)&d_kernel, kernelBytes);

    cudaMemcpy(d_input, h_input, imageBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernelBytes, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid(
        (WIDTH + block.x - 1) / block.x,
        (HEIGHT + block.y - 1) / block.y
    );

    convolution<<<grid, block>>>(
        d_input,
        d_output,
        WIDTH,
        HEIGHT,
        d_kernel
    );

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(error));
        return 1;
    }

    error = cudaDeviceSynchronize();

    if (error != cudaSuccess) {
        printf("Kernel execution error: %s\n", cudaGetErrorString(error));
        return 1;
    }

    cudaMemcpy(h_output, d_output, imageBytes, cudaMemcpyDeviceToHost);

    printImage("Original Image:", h_input);
    printImage("Convolved Image:", h_output);

    printf("\nConvolution completed successfully!\n");

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);

    return 0;
}
