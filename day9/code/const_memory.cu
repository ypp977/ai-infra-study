#include <cuda_runtime.h>
#include <stdio.h>

// 常量内存
#define COEF_SIZE 1024
__constant__ float device_coef[COEF_SIZE]; // GPU常量内存

__global__ void kernel_const(const float* in, float* out, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        float val = in[i];
        for (int j = 0; j < 1000; j++)
        {
            val *= device_coef[j % COEF_SIZE];
        }
        out[i] = val;
    }
}

__global__ void kernel_global(const float* in, float* out, const float* coef, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        float val = in[i];
        for (int j = 0; j < 1000; j++)
        {
            val *= coef[j % COEF_SIZE];
        }
        out[i] = val;
    }
}

int main()
{
    const int N = 1 << 24; // 16M 元素
    size_t bytes = N * sizeof(float);

    float* host_in = (float*)malloc(bytes);
    float* host_out = (float*)malloc(bytes);
    for (int i = 0; i < N; i++)
    {
        host_in[i] = 1.0f;
    }

    float *device_in, *device_out, *device_coef_global;
    cudaMalloc(&device_in, bytes);
    cudaMalloc(&device_out, bytes);
    cudaMalloc(&device_coef_global, COEF_SIZE * sizeof(float));

    cudaMemcpy(device_in, host_in, bytes, cudaMemcpyHostToDevice);

    float host_coef[COEF_SIZE];
    for (int i = 0; i < COEF_SIZE; i++)
    {
        host_coef[i] = 1.0f;
    }
    // 把 coef 放到 constant memory
    cudaMemcpyToSymbol(device_coef, host_coef, COEF_SIZE * sizeof(float));
    // 把 coef 放到 global memory
    cudaMemcpy(device_coef_global, host_coef, COEF_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float ms;

    // constant memory
    cudaEventRecord(start);
    kernel_const<<<grid, block>>>(device_in, device_out, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("Const memory kernel: %.3f ms\n", ms);

    // global memory
    cudaEventRecord(start);
    kernel_global<<<grid, block>>>(device_in, device_out, device_coef_global, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("Global memory kernel: %.3f ms\n", ms);

    cudaFree(device_in);
    cudaFree(device_out);
    cudaFree(device_coef_global);

    free(host_in);
    free(host_out);

    return 0;
}
