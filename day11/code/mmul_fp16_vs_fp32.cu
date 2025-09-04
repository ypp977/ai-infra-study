#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define TILE 16

// FP32 baseline kernel
__global__ void mmul_fp32(const float* A, const float* B, float* C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    if (row < N && col < N)
    {
        for (int k = 0; k < N; k++)
        {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// FP16 输入 + FP32 累加 kernel
__global__ void mmul_fp16_acc32(const __half* A, const __half* B, float* C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    if (row < N && col < N)
    {
        for (int k = 0; k < N; k++)
        {
            // 半精度加载，再转为 float 做乘加，避免累计精度损失
            float a_val = __half2float(A[row * N + k]);
            float b_val = __half2float(B[k * N + col]);
            sum += a_val * b_val;
        }
        C[row * N + col] = sum;
    }
}

int main()
{
    int N = 512;
    size_t size_f32 = N * N * sizeof(float);
    size_t size_f16 = N * N * sizeof(__half);

    // 主机内存
    float* host_a = (float*)malloc(size_f32);
    float* host_b = (float*)malloc(size_f32);
    float* host_c_fp32 = (float*)malloc(size_f32);
    float* host_c_fp16 = (float*)malloc(size_f32);

    // 初始化数据
    for (int i = 0; i < N * N; i++)
    {
        host_a[i] = (float)(i % 3 + 1);
        host_b[i] = (float)(i % 5 + 1);
    }

    // 设备内存
    float *device_a_f32, *device_b_f32, *device_c_f32;
    __half *device_a_f16, *device_b_f16;
    float* device_c_f16;

    cudaMalloc(&device_a_f32, size_f32);
    cudaMalloc(&device_b_f32, size_f32);
    cudaMalloc(&device_c_f32, size_f32);

    cudaMalloc(&device_a_f16, size_f16);
    cudaMalloc(&device_b_f16, size_f16);
    cudaMalloc(&device_c_f16, size_f32);

    // 拷贝 FP32 输入
    cudaMemcpy(device_a_f32, host_a, size_f32, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b_f32, host_b, size_f32, cudaMemcpyHostToDevice);

    // 将 FP32 转换为 FP16 并拷贝
    __half* host_a_f16 = (__half*)malloc(size_f16);
    __half* host_b_f16 = (__half*)malloc(size_f16);

    for (int i = 0; i < N * N; i++)
    {
        host_a_f16[i] = __float2half(host_a[i]);
        host_b_f16[i] = __float2half(host_b[i]);
    }
    cudaMemcpy(device_a_f16, host_a_f16, size_f16, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b_f16, host_b_f16, size_f16, cudaMemcpyHostToDevice);

    // kernel 配置
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // FP32 baseline
    cudaEventRecord(start);
    mmul_fp32<<<grid, block>>>(device_a_f32, device_b_f32, device_c_f32, N);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_fp32;
    cudaEventElapsedTime(&ms_fp32, start, stop);
    cudaMemcpy(host_c_fp32, device_c_f32, size_f32, cudaMemcpyDeviceToHost);

    // FP16 输入 + FP32 累加
    cudaEventRecord(start);
    mmul_fp16_acc32<<<grid, block>>>(device_a_f16, device_b_f16, device_c_f16, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_fp16;
    cudaEventElapsedTime(&ms_fp16, start, stop);
    cudaMemcpy(host_c_fp16, device_c_f16, size_f32, cudaMemcpyDeviceToHost);

    // 结果对比
    printf("Matrix N=%d\n", N);
    printf("FP32 kernel time = %.3f ms\n", ms_fp32);
    printf("FP16 input + FP32 accumulate kernel time = %.3f ms\n", ms_fp16);
    printf("C[0] FP32=%.2f, FP16+FP32=%.2f\n", host_c_fp32[0], host_c_fp16[0]);

    // 误差检查
    double diff = 0.0;
    for (int i = 0; i < N * N; i++)
    {
        diff += fabs(host_c_fp32[i] - host_c_fp16[i]);
    }
    printf("Total abs diff = %.3f\n", diff);

    // 释放
    free(host_a);
    free(host_b);
    free(host_c_fp32);
    free(host_c_fp16);
    free(host_a_f16);
    free(host_b_f16);
    cudaFree(device_a_f32);
    cudaFree(device_a_f32);
    cudaFree(device_b_f16);
    cudaFree(device_b_f32);
    cudaFree(device_c_f16);
    cudaFree(device_c_f32);

    return 0;
}
