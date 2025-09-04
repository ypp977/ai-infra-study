#include <stdio.h>

__global__ void mmul_native(const float* A, const float* B, float* C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N)
    {
        float sum = 0.0f;
        for (int k = 0; k < N; k++)
        {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main()
{
    int N = 1024;
    size_t size = N * N * sizeof(float);

    float *host_a, *host_b, *host_c, *device_a, *device_b, *device_c;
    host_a = (float*)malloc(size);
    host_b = (float*)malloc(size);
    host_c = (float*)malloc(size);
    for (int i = 0; i < N * N; i++)
    {
        host_a[i] = 1.0f;
        host_b[i] = 1.0f;
    }

    cudaMalloc(&device_a, size);
    cudaMalloc(&device_b, size);
    cudaMalloc(&device_c, size);

    cudaMemcpy(device_a, host_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, host_b, size, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    mmul_native<<<grid, block>>>(device_a, device_b, device_c, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(host_c, device_c, size, cudaMemcpyDeviceToHost);

    printf("GPU naive done, host_c[0]=%.1f , time = %.4f\n", host_c[0], ms);

    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);

    free(host_a);
    free(host_b);
    free(host_c);

    return 0;
}
