#include <cuda_runtime.h>
#include <stdio.h>

__global__ void touch(float* data, long N)
{
    long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        data[i] += 1.0f;
    }
}

int main(int argc, char* argv[])
{
    // N 太大可能导致系统直接OOM
    long N = (long)1e9; // 默认 1e9 (~4 GB)
    if (argc > 1)
    {
        N = atol(argv[1]); // 可以从命令行传 N
    }
    size_t bytes = N * sizeof(float);

    printf("Allocating %.2f GB Unified Memory...\n", bytes / 1e9);

    float* data;
    cudaMallocManaged(&data, bytes); // unified memory

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    touch<<<grid, block>>>(data, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    printf("Kernel with UM (N=%ld, %.2f GB): %.3f ms\n", N, bytes / 1e9, ms);

    cudaFree(data);

    return 0;
}
