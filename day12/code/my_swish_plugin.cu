#include "my_swish_plugin.h"
#include <cuda_runtime.h>

// swish kernel
__global__ void swish_kernel(const float* input, float* output, int num)
{
    // 计算线程索引
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num)
    {
        // 计算 swish 值
        float x = input[i];
        output[i] = x / (1.0f + expf(-x));
    }
}

// enqueue
int SwishPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
                         const void* const* inputs, void* const* outputs, void* workspace,
                         cudaStream_t stream) noexcept
{
    // 计算输入和输出的大小
    int num = 1;

    for (int i = 0; i < inputDesc[0].dims.nbDims; i++)
    {
        num *= inputDesc[0].dims.d[i];
    }

    // 获取输入和输出
    const float* input = reinterpret_cast<const float*>(inputs[0]);
    float* output = reinterpret_cast<float*>(outputs[0]);

    // 计算 block 和 grid
    int block = 256;
    int grid = (num + block - 1) / block;

    // 启动 kernel
    swish_kernel<<<grid, block, 0, stream>>>(input, output, num);

    auto err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Swish kernel launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    return 0;
}
// 注册插件到 TensorRT
REGISTER_TENSORRT_PLUGIN(SwishPluginCreator);
