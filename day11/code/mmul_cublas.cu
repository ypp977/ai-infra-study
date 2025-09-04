#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>

int main()
{
    int N = 1024;
    size_t size = N * N * sizeof(float);

    float* host_a = (float*)malloc(size);
    float* host_b = (float*)malloc(size);
    float* host_c = (float*)malloc(size);

    for (int i = 0; i < N * N; i++)
    {
        host_a[i] = 1.0f;
        host_b[i] = 1.0f;
    }

    float *device_a, *device_b, *device_c;
    cudaMalloc((void**)&device_a, size);
    cudaMalloc((void**)&device_b, size);
    cudaMalloc((void**)&device_c, size);

    cudaMemcpy(device_a, host_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, host_b, size, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate_v2(&handle);

    float alpha = 1.0f, beta = 0.0f;

    int repeat = 5; // 统计 5 次
    float total_ms = 0.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // 排除第一次调用overhead
    cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, device_b, N, device_a, N,
                   &beta, device_c, N);
    cudaDeviceSynchronize();

    for (int i = 0; i < repeat; i++)
    {
        cudaEventRecord(start);
        cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, device_b, N, device_a, N,
                       &beta, device_c, N);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;

        cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
    }
    float avg_ms = total_ms / repeat;
    cudaMemcpy(host_c, device_c, size, cudaMemcpyDeviceToHost);

    printf("cuBLAS N=%d, Time: %f ms\n", N, avg_ms);
    printf("host_c[0] = %f\n", host_c[0]);

    cublasDestroy_v2(handle);

    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);

    free(host_a);
    free(host_b);
    free(host_c);

    return 0;
}
