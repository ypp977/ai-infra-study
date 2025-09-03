#include <cuda_runtime.h>
#include <stdio.h>

__global__ void kernel_if(int* out, const int* in, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N)
    {
        if (in[tid] % 2 == 0)
        {
            out[tid] = in[tid] * 2;
        }
        else
        {
            out[tid] = in[tid] * 3;
        }
    }
}

__global__ void kernel_selp(int* out, const int* in, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N)
    {
        int val = in[tid];
        // 三目运算避免分支发散
        out[tid] = (val % 2 == 0) ? (val * 2) : (val * 3);
    }
}

int main()
{
    const int N = 1 << 20;
    size_t bytes = N * sizeof(int);

    int* host_in = (int*)malloc(bytes);
    int* host_out = (int*)malloc(bytes);
    for (int i = 0; i < N; i++)
    {
        host_in[i] = i;
    }

    int *device_in, *device_out;
    cudaMalloc(&device_in, bytes);
    cudaMalloc(&device_out, bytes);
    cudaMemcpy(device_in, host_in, bytes, cudaMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    // 计时
    cudaEvent_t start, stop;
    float ms;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    kernel_if<<<grid, block>>>(device_out, device_in, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("if-else kernel: %.3f ms\n", ms);

    cudaEventRecord(start);
    kernel_selp<<<grid, block>>>(device_out, device_in, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("selp kernel: %.3f ms\n", ms);

    cudaFree(device_in);
    cudaFree(device_out);

    free(host_in);
    free(host_out);

    return 0;
}
