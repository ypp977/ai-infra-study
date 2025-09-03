#include <cuda_fp16.h>
#include <mma.h>
#include <stdio.h>

using namespace nvcuda;

#define M 16
#define N 16
#define K 16

// WMMA GEMM kernel
__global__ void wmma_geem(half* a, half* b, float* c)
{
    // 每个 warp 计算一个 16 * 16 tile
    wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, M, N, K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    // warp 内 id
    int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / 32;

    // 每个 warp 加载一个 tile
    wmma::load_matrix_sync(a_frag, a + warpId * M * N * K, K);
    wmma::load_matrix_sync(b_frag, b + warpId * M * N * K, K);

    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    wmma::store_matrix_sync(c + warpId * M * N, c_frag, N, wmma::mem_row_major);
}

int main()
{
    int numWarps = 1;
    int numThreads = 32 * numWarps;

    size_t bytes_a = M * K * sizeof(half);
    size_t bytes_b = K * N * sizeof(half);
    size_t bytes_c = M * N * sizeof(float);

    half* host_a = (half*)malloc(bytes_a);
    half* host_b = (half*)malloc(bytes_b);
    float* host_c = (float*)malloc(bytes_c);

    // 初始化
    for (int i = 0; i < M * K; i++)
    {
        host_a[i] = __float2half(1.0f);
    }

    for (int i = 0; i < K * N; i++)
    {
        host_b[i] = __float2half(2.0f);
    }

    half *device_a, *device_b;
    float* device_c;

    cudaMalloc(&device_a, bytes_a);
    cudaMalloc(&device_b, bytes_b);
    cudaMalloc(&device_c, bytes_c);

    cudaMemcpy(device_a, host_a, bytes_a, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, host_b, bytes_b, cudaMemcpyHostToDevice);

    // 启动kernel
    wmma_geem<<<1, numThreads>>>(device_a, device_b, device_c);
    cudaMemcpy(host_c, device_c, bytes_c, cudaMemcpyDeviceToHost);

    printf("C[0] = %.1f\n", host_c[0]);

    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);
    free(host_a);
    free(host_b);
    free(host_c);

    return 0;
}
