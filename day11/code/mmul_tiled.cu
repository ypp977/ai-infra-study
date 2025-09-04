#include <cuda_runtime.h>
#include <stdio.h>

#define TILE 32

// GPU 矩阵乘 (Tiled + shared memory)
__global__ void mmul_tiled(const float* a, const float* b, float* c, int N)
{
    __shared__ float a_shared[TILE][TILE];
    __shared__ float b_shared[TILE][TILE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (N + TILE - 1) / TILE; t++)
    {

        // 加载 A 的一个tile
        if (row < N && t * TILE + threadIdx.x < N)
        {
            a_shared[threadIdx.y][threadIdx.x] = a[row * N + t * TILE + threadIdx.x];
        }
        else
        {
            a_shared[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // 加载 B 的一个tile
        if (col < N && t * TILE + threadIdx.y < N)
        {
            b_shared[threadIdx.y][threadIdx.x] = b[(t * TILE + threadIdx.y) * N + col];
        }
        else
        {
            b_shared[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();
        // 当前 tile 的计算
        for (int k = 0; k < TILE; k++)
        {
            sum += a_shared[threadIdx.y][k] * b_shared[k][threadIdx.x];
        }

        __syncthreads();
    }

    // 写回结果
    if (row < N && col < N)
    {
        c[row * N + col] = sum;
    }
}

int main()
{
    int N = 1024;
    size_t size = N * N * sizeof(float);

    // 主机内存
    float* host_a = (float*)malloc(size);
    float* host_b = (float*)malloc(size);
    float* host_c = (float*)malloc(size);

    // 初始化A、B
    for (int i = 0; i < N * N; i++)
    {
        host_a[i] = 1.0f;
        host_b[i] = 1.0f;
    }

    // 设备内存
    float *device_a, *device_b, *device_c;
    cudaMalloc((void**)&device_a, size);
    cudaMalloc((void**)&device_b, size);
    cudaMalloc((void**)&device_c, size);

    // 拷贝数据到 GPU
    cudaMemcpy(device_a, host_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, host_b, size, cudaMemcpyHostToDevice);

    // 启动kernel
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    mmul_tiled<<<grid, block>>>(device_a, device_b, device_c, N);
    cudaEventRecord(stop);

    cudaMemcpy(host_c, device_c, size, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    printf("N = %d, GPU tiled kernel time = %.3f ms\n", N, ms);
    printf("host_c[0] = %.1f\n", host_c[0]); // 验证结果正确性

    // 释放资源
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);
    free(host_a);
    free(host_b);
    free(host_c);

    return 0;
}
