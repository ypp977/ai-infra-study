#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vector_add(const float *a, const float *b, float *c,int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // 计算全局索引

    if(i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int n = 1 << 16; // 65536个元素
    size_t bytes = n * sizeof(float);

    // 分配 host 内存
    float *host_a = (float *)malloc(bytes);
    float *host_b = (float *)malloc(bytes);
    float *host_c = (float *)malloc(bytes);

    // 初始化数据
    for (int i = 0; i < n;i++) {
        host_a[i] = 1.0f;
        host_b[i] = 2.0f;
    }

    // 分配device 内存
    float *device_a, *device_b, *device_c;
    cudaMalloc(&device_a, bytes);
    cudaMalloc(&device_b, bytes);
    cudaMalloc(&device_c, bytes);

    // 复制数据 Host -> Device
    cudaMemcpy(device_a, host_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, host_b, bytes, cudaMemcpyHostToDevice);

    // 计算 grid/block 配置
    int blockSize = 64;
    int gridSize = (n + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    // 启动kernel
    vector_add<<<gridSize, blockSize>>>(device_a, device_b, device_c, n);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("blockSize=%d, Time=%.3f ms\n", blockSize, ms);

    cudaDeviceSynchronize();

    // 复制结果 device -> host;
    cudaMemcpy(host_c, device_c, bytes, cudaMemcpyDeviceToHost);

    // 验证结果
    for (int i = 0; i < 10;i++) {
        printf("c[%d] = %f\n", i, host_c[i]);
    }

    // 清理
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);
    
    free(host_a);
    free(host_b);
    free(host_c);

    return 0;
}