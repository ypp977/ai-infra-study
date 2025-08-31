#include <stdio.h>
#include <cuda_runtime.h>

// 1D Block 的矩阵加法
__global__ void mat_add_1D(const float *A, const float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N * N) {
        C[i] = A[i] + B[i];
    }
}

// 2D Block 的矩阵加法
__global__ void mat_add_2D(const float *A, const float *B, float *C, int N) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if(r < N && c < N) {
        int idx = r * N + c;// 行主序展开
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    const int N = 1024; // 矩阵大小N * N
    size_t bytes = N * N * sizeof(float);

    // 分配Host内存
    float *host_a = (float *)malloc(bytes);
    float *host_b = (float *)malloc(bytes);
    float *host_c1d = (float *)malloc(bytes);
    float *host_c2d = (float *)malloc(bytes);

    // 初始化矩阵数据
    for (int i = 0; i < N * N;i++) {
        host_a[i] = 1.0f;
        host_b[i] = 2.0f;
    }

    // 分配device内存
    float *device_a, *device_b, *device_c;
    cudaMalloc(&device_a, bytes);
    cudaMalloc(&device_b, bytes);
    cudaMalloc(&device_c, bytes);

    // 拷贝数据
    cudaMemcpy(device_a, host_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, host_b, bytes, cudaMemcpyHostToDevice);

    // 配置1D kernel 启动参数
    dim3 block1(256);
    dim3 grid1((N * N + block1.x - 1) / block1.x);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // 启动1D 内核
    mat_add_1D<<<grid1, block1>>>(device_a, device_b, device_c, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float c1d_ms;
    cudaEventElapsedTime(&c1d_ms, start, stop);

    cudaMemcpy(host_c1d, device_c, bytes, cudaMemcpyDeviceToHost);

    // 配置2D kernel 启动参数
    dim3 block2(16, 16);
    dim3 grid2((N + block2.x - 1) / block2.x, (N + block2.y - 1) / block2.y);

    cudaEventRecord(start);
    // 启动2D 内核
    mat_add_2D<<<grid2, block2>>>(device_a, device_b, device_c, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float c2d_ms;
    cudaEventElapsedTime(&c2d_ms, start, stop);
    cudaMemcpy(host_c2d, device_c, bytes, cudaMemcpyDeviceToHost);

    // 验证结果10 个元素
    printf("check results (first 10 elements):\n");
    for (int i = 0; i < 10; i++) {
        printf("C1D[%d]=%.1f C2D[%d]=%.1f\n", i, host_c1d[i], i, host_c2d[i]);
    }

    // 打印性能对比
    printf("\nPerformance comparison (Matrix %d x %d):\n", N, N);
    printf("1D Block: %.3f ms\n", c1d_ms);
    printf("2D Block: %.3f ms\n", c2d_ms);

    // 清理资源
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);

    free(host_a);
    free(host_b);
    free(host_c1d);
    free(host_c2d);

    return 0;
}