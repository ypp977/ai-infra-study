#include <cuda_runtime.h>
#include <stdio.h>

#define TILE 32

// 双缓冲 GEMM kernel
__global__ void mmul_tiled_double_buffer(const float* A, const float* B, float* C, int N)
{
    __shared__ float a_shared[2][TILE][TILE];
    __shared__ float b_shared[2][TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    int buf = 0;

    // 预加载第 0 快tile
    if (row < N && threadIdx.x < TILE)
    {
        a_shared[buf][threadIdx.y][threadIdx.x] = A[row * N + threadIdx.x];
    }
    else
    {
        a_shared[buf][threadIdx.y][threadIdx.x] = 0.0f;
    }
    if (col < N && threadIdx.y < TILE)
    {
        b_shared[buf][threadIdx.y][threadIdx.x] = B[threadIdx.y * N + col];
    }
    else
    {
        b_shared[buf][threadIdx.y][threadIdx.x] = 0.0f;
    }

    __syncthreads();

    // 遍历所有tile
    for (int t = 0; t < (N + TILE - 1) / TILE; t++)
    {
        int next = (buf + 1) % 2;

        // 提前预取下一个 tile
        if (t + 1 < (N + TILE - 1) / TILE)
        {
            if (row < N && (t + 1) * TILE + threadIdx.x < N)
            {
                a_shared[next][threadIdx.y][threadIdx.x] =
                    A[row * N + (t + 1) * TILE + threadIdx.x];
            }
            else
            {
                a_shared[next][threadIdx.y][threadIdx.x] = 0.0f;
            }

            if (col < N && (t + 1) * TILE + threadIdx.y < N)
            {
                b_shared[next][threadIdx.y][threadIdx.x] =
                    B[((t + 1) * TILE + threadIdx.y) * N + col];
            }
            else
            {
                b_shared[next][threadIdx.y][threadIdx.x] = 0.0f;
            }
        }
        for (int k = 0; k < TILE; k++)
        {
            sum += a_shared[buf][threadIdx.y][k] * b_shared[buf][k][threadIdx.x];
        }

        __syncthreads();
        buf = next;
    }
    if (row < N && col < N)
    {
        C[row * N + col] = sum;
    }
}

int main()
{
    int N = 1024; // 矩阵大小
    size_t size = N * N * sizeof(float);

    // 分配主机内存
    float* host_a = (float*)malloc(size);
    float* host_b = (float*)malloc(size);
    float* host_c = (float*)malloc(size);

    // 初始化
    for (int i = 0; i < N * N; i++)
    {
        host_a[i] = 1.0f;
        host_b[i] = 1.0f;
    }

    // 分配设备内存
    float *device_a, *device_b, *device_c;
    cudaMalloc((void**)&device_a, size);
    cudaMalloc((void**)&device_b, size);
    cudaMalloc((void**)&device_c, size);

    cudaMemcpy(device_a, host_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, host_b, size, cudaMemcpyHostToDevice);

    // 配置kernel
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    // 计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    mmul_tiled_double_buffer<<<grid, block>>>(device_a, device_b, device_c, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(host_c, device_c, size, cudaMemcpyDeviceToHost);

    // 输出结果
    printf("N = %d, GPU double buffer kernel time = %.3f ms\n", N, ms);
    printf("C[0] = %.1f\n", host_c[0]);

    // 清理
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);

    free(host_a);
    free(host_b);
    free(host_c);

    return 0;
}
