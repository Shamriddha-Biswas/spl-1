#include "../include/cuda_utils.h"

bool check_cuda_success(cudaError_t err, const char* msg) {
#ifdef __CUDACC__
    if (err != cudaSuccess) {
        std::cout << "CUDA Error [" << (msg ? msg : "Operation") << "]: " 
                  << cudaGetErrorString(err) << std::endl;
        return false;
    }
#endif
    return true;
}

bool query_gpu_hardware(HardwareProfile* profile) {
    if (!profile) return false;

    std::memset(profile, 0, sizeof(HardwareProfile));

#ifdef __CUDACC__
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count <= 0) {
        profile->device_count = 0;
        return false;
    }

    profile->device_count = count;
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        return false;
    }

    std::strncpy(profile->device_name, prop.name, 255);
    profile->compute_major = prop.major;
    profile->compute_minor = prop.minor;
    profile->total_memory = prop.totalGlobalMem;

    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    profile->free_memory = free_mem;

    profile->max_threads_per_block = prop.maxThreadsPerBlock;
    profile->max_threads_dim[0] = prop.maxThreadsDim[0];
    profile->max_threads_dim[1] = prop.maxThreadsDim[1];
    profile->max_threads_dim[2] = prop.maxThreadsDim[2];
    profile->max_grid_dim[0] = prop.maxGridSize[0];
    profile->max_grid_dim[1] = prop.maxGridSize[1];
    profile->max_grid_dim[2] = prop.maxGridSize[2];
    profile->shared_mem_per_block = prop.sharedMemPerBlock;
    profile->warp_size = prop.warpSize;

    return true;
#else
    profile->device_count = 0;
    return false;
#endif
}

void print_hardware_profile(const HardwareProfile* profile) {
    if (!profile) return;

    std::cout << "\n=================================================" << std::endl;
    std::cout << "            GPU HARDWARE SPECIFICATIONS          " << std::endl;
    std::cout << "=================================================" << std::endl;
    if (profile->device_count <= 0) {
        std::cout << "Status                  : No CUDA Hardware Detected" << std::endl;
        std::cout << "=================================================\n" << std::endl;
        return;
    }

    double total_gb = (double)profile->total_memory / (1024.0 * 1024.0 * 1024.0);
    double free_gb = (double)profile->free_memory / (1024.0 * 1024.0 * 1024.0);
    double shared_kb = (double)profile->shared_mem_per_block / 1024.0;

    std::cout << "Device Name             : " << profile->device_name << std::endl;
    std::cout << "Compute Capability      : " << profile->compute_major << "." << profile->compute_minor << std::endl;
    std::cout << "Total Global Memory     : " << profile->total_memory << " bytes (" << total_gb << " GB)" << std::endl;
    std::cout << "Available Free Memory   : " << profile->free_memory << " bytes (" << free_gb << " GB)" << std::endl;
    std::cout << "Max Threads Per Block   : " << profile->max_threads_per_block << std::endl;
    std::cout << "Max Block Dimensions    : [" << profile->max_threads_dim[0] << ", " 
              << profile->max_threads_dim[1] << ", " << profile->max_threads_dim[2] << "]" << std::endl;
    std::cout << "Max Grid Dimensions     : [" << profile->max_grid_dim[0] << ", " 
              << profile->max_grid_dim[1] << ", " << profile->max_grid_dim[2] << "]" << std::endl;
    std::cout << "Shared Memory Per Block : " << profile->shared_mem_per_block << " bytes (" << shared_kb << " KB)" << std::endl;
    std::cout << "Warp Size               : " << profile->warp_size << " threads" << std::endl;
    std::cout << "=================================================\n" << std::endl;
}

void calculate_grid_block_2d(int width, int height, int block_x, int block_y, dim3* grid, dim3* block) {
    if (!grid || !block) return;
    if (block_x <= 0) block_x = 16;
    if (block_y <= 0) block_y = 16;

    block->x = block_x;
    block->y = block_y;
    block->z = 1;

    grid->x = (width + block_x - 1) / block_x;
    grid->y = (height + block_y - 1) / block_y;
    grid->z = 1;
}

bool allocate_device_image(int width, int height, int channels, uint8_t** d_ptr, size_t* allocated_bytes) {
    if (!d_ptr || width <= 0 || height <= 0 || channels <= 0) return false;

#ifdef __CUDACC__
    size_t bytes = (size_t)width * (size_t)height * (size_t)channels * sizeof(uint8_t);
    cudaError_t err = cudaMalloc((void**)d_ptr, bytes);
    if (!check_cuda_success(err, "allocate_device_image")) {
        *d_ptr = nullptr;
        return false;
    }
    if (allocated_bytes) *allocated_bytes = bytes;
    return true;
#else
    if (allocated_bytes) *allocated_bytes = 0;
    *d_ptr = nullptr;
    return false;
#endif
}

void free_device_buffer(void* d_ptr) {
    if (!d_ptr) return;
#ifdef __CUDACC__
    cudaFree(d_ptr);
#endif
}

bool copy_to_device(void* d_dst, const void* h_src, size_t bytes) {
    if (!d_dst || !h_src || bytes == 0) return false;
#ifdef __CUDACC__
    cudaError_t err = cudaMemcpy(d_dst, h_src, bytes, cudaMemcpyHostToDevice);
    return check_cuda_success(err, "copy_to_device");
#else
    return false;
#endif
}

bool copy_to_host(void* h_dst, const void* d_src, size_t bytes) {
    if (!h_dst || !d_src || bytes == 0) return false;
#ifdef __CUDACC__
    cudaError_t err = cudaMemcpy(h_dst, d_src, bytes, cudaMemcpyDeviceToHost);
    return check_cuda_success(err, "copy_to_host");
#else
    return false;
#endif
}
