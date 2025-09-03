#include <cuda_runtime.h>
#include <stdio.h>

// 非合并访存（行错位）
__global__ void kernel_non_coalesce(float* out, const float* in, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N)
    {
        // 故意乱序
        out[tid] = in[(tid * 17) % N];
    }
}

// 合并访存
__global__ void kernel_coalesce(float* out, const float* in, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N)
    {
        // 顺序访问
        out[tid] = in[tid];
    }
}

int main()
{
    const int N = 1 << 20; // 1M元素
    size_t bytes = N * sizeof(float);

    float* host_in = (float*)malloc(bytes);
    float* host_out = (float*)malloc(bytes);
    for (int i = 0; i < N; i++)
    {
        host_in[i] = i;
    }

    float *device_in, *device_out;
    cudaMalloc(&device_in, bytes);
    cudaMalloc(&device_out, bytes);

    cudaMemcpy(device_in, host_in, bytes, cudaMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    float ms_1;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    kernel_non_coalesce<<<grid, block>>>(device_out, device_in, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&ms_1, start, stop);
    cudaMemcpy(host_out, device_out, bytes, cudaMemcpyDeviceToHost);
    printf("Non-coalesced: time=%.3f ms, out[0]=%.1f\n", ms_1, host_out[0]);

    cudaEventRecord(start);
    kernel_coalesce<<<grid, block>>>(device_out, device_in, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&ms_1, start, stop);
    cudaMemcpy(host_out, device_out, bytes, cudaMemcpyDeviceToHost);
    printf("Coalesced: time=%.3f ms, out[0]=%.1f\n", ms_1, host_out[0]);

    cudaFree(device_in);
    cudaFree(device_out);

    free(host_in);
    free(host_out);

    return 0;
}
