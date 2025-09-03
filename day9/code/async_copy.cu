#include <cuda_runtime.h>
#include <stdio.h>

// 简单的计算kernel (模拟耗时计算)
__global__ void computer(float* data, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        float x = data[i];
        for (int j = 0; j < 10000; j++)
        {
            x = x * 0.999f + 0.001f;
        }
        data[i] = x;
    }
}

int main()
{
    const int N = 1 << 24;
    size_t bytes = N * sizeof(float);

    float *host_data, *device_data;
    cudaMallocHost(&host_data, bytes); // 页锁定内存(必须)
    cudaMalloc(&device_data, bytes);

    for (int i = 0; i < N; i++)
    {
        host_data[i] = 1.0f;
    }

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    // 同步版本
    cudaEvent_t start, stop;
    float ms_sync, ms_async;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    // 同步拷贝 H2D
    cudaMemcpy(device_data, host_data, bytes, cudaMemcpyHostToDevice);
    // 计算
    computer<<<grid, block>>>(device_data, N);
    // 同步拷贝 D2H
    cudaMemcpy(host_data, device_data, bytes, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&ms_sync, start, stop);

    // 异步版本
    cudaStream_t s1, s2;
    cudaStreamCreate(&s1);
    cudaStreamCreate(&s2);

    cudaEventRecord(start);
    // H2D 异步拷贝
    cudaMemcpyAsync(device_data, host_data, bytes, cudaMemcpyHostToDevice, s1);
    // 计算放到另一个stream
    computer<<<grid, block, 0, s2>>>(device_data, N);
    // D2H 异步拷贝
    cudaMemcpyAsync(host_data, device_data, bytes, cudaMemcpyDeviceToHost, s1);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_async, start, stop);

    printf("Sync  version time: %.3f ms\n", ms_sync);
    printf("Async version time: %.3f ms\n", ms_async);

    cudaStreamDestroy(s1);
    cudaStreamDestroy(s2);
    cudaFree(device_data);
    cudaFreeHost(host_data);

    return 0;
}
