#include <cuda_runtime.h>
#include <stdio.h>

// AoS
struct ParticleAos
{
    float x, y, z;
};

__global__ void kernel_aos(ParticleAos* arr, float* out, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N)
    {
        out[tid] = arr[tid].x + arr[tid].y + arr[tid].z;
    }
}

// SoA
__global__ void kernel_soa(float* x, float* y, float* z, float* out, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N)
    {
        out[tid] = x[tid] + y[tid] + z[tid];
    }
}

// 计时函数封装
template <typename Kernel_Func, typename... Args>
float runKernel(Kernel_Func Kernel, dim3 grid, dim3 block, Args... args)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    Kernel<<<grid, block>>>(args...);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}

int main()
{
    const int N = 1 << 20;
    size_t bytes = N * sizeof(float);

    float* host_x = (float*)malloc(bytes);
    float* host_y = (float*)malloc(bytes);
    float* host_z = (float*)malloc(bytes);
    for (int i = 0; i < N; i++)
    {
        host_x[i] = 1.0f;
        host_y[i] = 2.0f;
        host_z[i] = 3.0f;
    }

    ParticleAos* host_aos = (ParticleAos*)malloc(N * sizeof(ParticleAos));
    for (int i = 0; i < N; i++)
    {
        host_aos[i].x = 1.0f;
        host_aos[i].y = 2.0f;
        host_aos[i].z = 3.0f;
    }

    ParticleAos* device_aos;
    float *device_x, *device_y, *device_z, *device_out;
    cudaMalloc(&device_aos, N * sizeof(ParticleAos));
    cudaMalloc(&device_x, bytes);
    cudaMalloc(&device_y, bytes);
    cudaMalloc(&device_z, bytes);
    cudaMalloc(&device_out, bytes);

    cudaMemcpy(device_aos, host_aos, N * sizeof(ParticleAos), cudaMemcpyHostToDevice);
    cudaMemcpy(device_x, host_x, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_y, host_y, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_z, host_z, bytes, cudaMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    // AoS 内核计时
    float time_aos = runKernel(kernel_aos, grid, block, device_aos, device_out, N);
    // SoA 内核计时
    float time_soa =
        runKernel(kernel_soa, grid, block, device_x, device_y, device_z, device_out, N);

    printf("AoS kernel time = %.3f ms\n", time_aos);
    printf("SoA kernel time = %.3f ms\n", time_soa);

    cudaFree(device_aos);
    cudaFree(device_x);
    cudaFree(device_y);
    cudaFree(device_z);
    cudaFree(device_out);

    free(host_x);
    free(host_y);
    free(host_z);
    free(host_aos);

    return 0;
}
