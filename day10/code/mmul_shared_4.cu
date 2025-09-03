#include <cuda_runtime.h>
#include <stdio.h>

#define TILE 16
// shared memory
__global__ void mmul_shared(const float* A, const float* B, float* C, int N)
{
    __shared__ float a_shared[TILE][TILE];
    __shared__ float b_shared[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0;
    for (int t = 0; t < (N + TILE - 1) / TILE; t++)
    {
        if (row < N && (t * TILE + threadIdx.x) < N)
        {
            a_shared[threadIdx.y][threadIdx.x] = A[row * N + t * TILE + threadIdx.x];
        }
        else
        {
            a_shared[threadIdx.y][threadIdx.x] = 0;
        }

        if (col < N && (t * TILE + threadIdx.y) < N)
        {
            b_shared[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
        }
        else
        {
            b_shared[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();
// 加上循环展开
#pragma unroll 4
        for (int k = 0; k < TILE; k++)
        {
            sum += a_shared[threadIdx.y][k] * b_shared[k][threadIdx.x];
        }

        __syncthreads();
    }
    if (row < N && col < N)
    {
        C[row * N + col] = sum;
    }
}

int main()
{
    const int N = 512;
    size_t bytes = N * N * sizeof(float);

    float* host_a = (float*)malloc(bytes);
    float* host_b = (float*)malloc(bytes);
    float* host_c = (float*)malloc(bytes);

    for (int i = 0; i < N * N; i++)
    {
        host_a[i] = 1.0f;
        host_b[i] = 2.0f;
    }

    float *device_a, *device_b, *device_c;
    cudaMalloc(&device_a, bytes);
    cudaMalloc(&device_b, bytes);
    cudaMalloc(&device_c, bytes);

    cudaMemcpy(device_a, host_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, host_b, bytes, cudaMemcpyHostToDevice);

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    mmul_shared<<<grid, block>>>(device_a, device_b, device_c, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(host_c, device_c, bytes, cudaMemcpyDeviceToHost);

    printf("Shared GEMM time = %.3f ms, result C[0]=%.1f\n", ms, host_c[0]);

    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);

    free(host_a);
    free(host_b);
    free(host_c);

    return 0;
}
