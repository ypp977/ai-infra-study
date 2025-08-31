#include <stdio.h>
#include <math.h>

// 简单数学运算kernel
__global__ void test_math(float *out) {
    int i = threadIdx.x;
    float x = i * 0.1f;

    // 调用sinf, cosf, sqrtf 这些数学函数
    out[i] = sinf(x) + cosf(x) + sqrt(x);
}

int main() {
    const int N = 128;
    float host_out[N], *device_out;

    // 分配device 内存
    cudaMalloc(&device_out, N * sizeof(float));

    // 启动kernel
    test_math<<<1, N>>>(device_out);

    // 拷回结果
    cudaMemcpy(host_out, device_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印前十个元素
    for (int i = 0; i < 10; i++) {
        printf("out[%d] = %.8f\n", i, host_out[i]);
    }

    cudaFree(device_out);
    return 0;
}