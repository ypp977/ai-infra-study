#include <stdio.h>
#include <cuda_runtime.h>

// grid-stride loop写法
__global__ void vector_add_gridstride(const float *a, const float *b, float *c, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;i += blockDim.x * gridDim.x) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int n = 1 << 20; // 1M 元素
    size_t bytes = n * sizeof(float);

    float *host_a = (float *)malloc(bytes);
    float *host_b = (float *)malloc(bytes);
    float *host_c = (float *)malloc(bytes);

    for (int i = 0; i < n;i++) {
        host_a[i] = 1.0f;
        host_b[i] = 2.0f;
    }

    float *device_a, *device_b, *device_c;
    cudaMalloc(&device_a, bytes);
    cudaMalloc(&device_b, bytes);
    cudaMalloc(&device_c, bytes);

    cudaMemcpy(device_a, host_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, host_b, bytes, cudaMemcpyHostToDevice);

    // 比较两种写法
    int blockSize = 256;
    int gridSize_small = 32; // Grid-Stride Loop 用小 grid
    int gridSize_big = (n + blockSize - 1) / blockSize; // 传统大grid

    // 计时用时间
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Grid-stride loop
    cudaEventRecord(start);
    vector_add_gridstride<<<gridSize_small, blockSize>>>(device_a, device_b, device_c, n);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float ms1;
    cudaEventElapsedTime(&ms1, start, stop);

    // 单发大 grid
    cudaEventRecord(start);
    vector_add_gridstride<<<gridSize_big, blockSize>>>(device_a, device_b, device_c, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms2;
    cudaEventElapsedTime(&ms2, start, stop);

    printf("Grid-Stride Loop: %.3f ms\n", ms1);
    printf("One Big Grid:     %.3f ms\n", ms2);

    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);

    free(host_a);
    free(host_b);
    free(host_c);
    return 0;
}