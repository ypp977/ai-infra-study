#include <cuda_runtime.h>
#include <stdio.h>

// 冲突版本： stride = 17
__global__ void conflict(float* out)
{
    __shared__ float shared_data[32 * 17];
    int tid = threadIdx.x;
    // 多个线程映射到同一个bank
    shared_data[tid * 17] = tid;
    __syncthreads();
    out[tid] = shared_data[tid * 17];
}

// 无冲突版本： stride = 17 + padding
__global__ void no_conflict(float* out)
{
    __shared__ float shared_data[32 * 17 + 1]; // padding +1
    int tid = threadIdx.x;
    // padding 打散 bank 映射
    shared_data[tid * 17] = tid;
    __syncthreads();
    out[tid] = shared_data[tid * 17];
}

// 计时封装函数
float run_and_time(void (*kernel)(float*), float* device_out, int N)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    kernel<<<1, N>>>(device_out);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}

int main()
{
    // warp 内 32 线程
    const int N = 32;
    size_t bytes = N * sizeof(float);

    float host_out[N];
    float* device_out;
    cudaMalloc(&device_out, bytes);

    // 计时并运行冲突版本
    float time1 = run_and_time(conflict, device_out, N);
    cudaMemcpy(host_out, device_out, bytes, cudaMemcpyDeviceToHost);
    printf("Conflict kernel (%.6f ms)\n", time1);
    for (int i = 0; i < 5; i++)
    {
        printf("out[%d] = %.1f ", i, host_out[i]);
    }
    printf("\n");

    // 计时并运行无冲突版本
    float time2 = run_and_time(no_conflict, device_out, N);
    cudaMemcpy(host_out, device_out, bytes, cudaMemcpyDeviceToHost);
    printf("NO Conflict kernel (%.6f ms)\n", time2);
    for (int i = 0; i < 5; i++)
    {
        printf("out[%d] = %.1f ", i, host_out[i]);
    }
    printf("\n");

    cudaFree(device_out);
    return 0;
}
