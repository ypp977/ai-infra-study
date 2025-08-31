#include <cuda_runtime.h>
#include <stdio.h>

#define SIZE 1024 // 矩阵大小 N * N
// Global memory版本
__global__ void mat_add_global(const float* A, const float* B, float* C, int N)
{
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < N && c < N)
    {
        C[r * N + c] = A[r * N + c] + B[r * N + c];
    }
}

// Shared memory版本
__global__ void mat_add_shared(const float* A, const float* B, float* C, int N)
{
    __shared__ float A_shared[32][32];
    __shared__ float B_shared[32][32];

    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = threadIdx.y, tx = threadIdx.x;

    if (r < N && c < N)
    {
        // 把数据搬到shared memory
        A_shared[ty][tx] = A[r * N + c];
        B_shared[ty][tx] = B[r * N + c];

        // 确保所有线程写完
        __syncthreads();

        // 从 shared memory 读出再计算
        C[r * N + c] = A_shared[ty][tx] + B_shared[ty][tx];
    }
}

int main()
{
    size_t bytes = SIZE * SIZE * sizeof(float);

    // 分配 host 内存
    float* host_a = (float*)malloc(bytes);
    float* host_b = (float*)malloc(bytes);
    float* host_c = (float*)malloc(bytes);

    for (int i = 0; i < SIZE * SIZE; i++)
    {
        host_a[i] = 1.0f;
        host_b[i] = 2.0f;
    }

    // 分配 device 内存
    float *device_a, *device_b, *device_c;
    cudaMalloc(&device_a, bytes);
    cudaMalloc(&device_b, bytes);
    cudaMalloc(&device_c, bytes);

    cudaMemcpy(device_a, host_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, host_b, bytes, cudaMemcpyHostToDevice);

    // 每个 Block 32 * 32 个线程
    dim3 block(32, 32);
    dim3 grid((SIZE + 31) / 32, (SIZE + 31) / 32);

    // cuda 事件用于计时
    cudaEvent_t start, stop;
    float ms;

    // Global memory版本
    cudaMemset(device_c, 0, bytes);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    mat_add_global<<<grid, block>>>(device_a, device_b, device_c, SIZE);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&ms, start, stop);
    printf("mat_add_global: %.3f ms, 带宽≈ %.2f GB/s\n", ms,
           (3 * SIZE * SIZE * sizeof(float) / 1e9) / (ms / 1000));

    // Shared memory版本
    cudaMemset(device_c, 0, bytes);
    cudaEventRecord(start);
    mat_add_shared<<<grid, block>>>(device_a, device_b, device_c, SIZE);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);

    printf("mat_add_shared: %.3f ms, 带宽≈ %.2f GB/s\n", ms,
           (3 * SIZE * SIZE * sizeof(float) / 1e9) / (ms / 1000));

    // 清理
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);

    free(host_a);
    free(host_b);
    free(host_c);

    return 0;
}
