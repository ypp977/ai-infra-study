#include <stdio.h>

__global__ void copy_shared(float* device_out, const float* device_in, int N)
{
    // 声明 Block 内的共享内存(固定 256 个float)
    __shared__ float share_data[256];

    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;

    if (global_tid < N)
    {
        // step 1:从 global memory 拷贝到 shared memory
        share_data[local_tid] = device_in[global_tid];

        // 等待所有线程完成拷贝
        __syncthreads();

        // step 2: 使用shared memory 的值
        device_out[global_tid] = share_data[local_tid] * 2.0f;
    }
}

int main()
{
    const int N = 256;
    size_t bytes = N * sizeof(float);
    float host_in[N], host_out[N];
    for (int i = 0; i < N; i++)
    {
        host_in[i] = i;
    }

    float *device_in, *device_out;
    cudaMalloc(&device_in, bytes);
    cudaMalloc(&device_out, bytes);

    cudaMemcpy(device_in, host_in, bytes, cudaMemcpyHostToDevice);

    copy_shared<<<1, 256>>>(device_out, device_in, N);
    cudaMemcpy(host_out, device_out, bytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
    {
        printf("host_out[%d] = %f\n", i, host_out[i]);
    }

    cudaFree(device_in);
    cudaFree(device_out);

    return 0;
}
