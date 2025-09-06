#include <NvInfer.h>
#include <cassert>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <iostream>
#include <vector>

using namespace nvinfer1;

// 日志器
class Logger : public ILogger
{
    // 记录日志
    void log(Severity severity, const char* msg) noexcept override
    {
        // 只记录 INFO 级别的日志
        if (severity <= Severity::kINFO)
        {
            std::cout << msg << std::endl;
        }
    }
};

int main()
{
    // 加载插件库
    void* handle = dlopen("./libswish.so", RTLD_LAZY);
    if (!handle)
    {
        std::cerr << "❌ Failed to load libswish.so: " << dlerror() << std::endl;
        return -1;
    }

    // 创建日志器
    Logger logger;
    // 创建构建器
    IBuilder* builder = createInferBuilder(logger);
    // 创建配置
    IBuilderConfig* config = builder->createBuilderConfig();
    // 创建网络
    auto network = builder->createNetworkV2(
        1U << static_cast<int>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
    // 创建输入
    ITensor* input = network->addInput("input", DataType::kFLOAT, Dims2{1, 16});
    // 获取插件注册表
    auto* registy = getPluginRegistry();
    // 获取插件创建者数量
    int32_t numCreators = 0;
    auto creatorList = registy->getPluginCreatorList(&numCreators);
    // 获取 Swish 插件创建者
    IPluginCreator* swish_creator = nullptr;
    for (int i = 0; i < numCreators; i++)
    {
        // 获取插件名称
        if (std::string(creatorList[i]->getPluginName()) == "SwishPlugin")
        {
            // 获取 Swish 插件创建者
            swish_creator = creatorList[i];
            break;
        }
    }
    // 断言 Swish 插件创建者不为空
    assert(swish_creator && "SwishPlugin not found!");

    // 创建插件字段集合
    PluginFieldCollection fc{};
    // 创建 Swish 插件
    IPluginV2* plugin = swish_creator->createPlugin("swish_layer", &fc);
    // 创建 Swish 层
    auto swish_layer = network->addPluginV2(&input, 1, *plugin);
    // 标记输出
    network->markOutput(*swish_layer->getOutput(0));

    // 序列化网络
    IHostMemory* serialized = builder->buildSerializedNetwork(*network, *config);
    // 创建运行时
    IRuntime* runtime = createInferRuntime(logger);
    // 反序列化网络
    ICudaEngine* engine = runtime->deserializeCudaEngine(serialized->data(), serialized->size());
    // 创建执行上下文
    IExecutionContext* context = engine->createExecutionContext();

    // 创建输入和输出
    std::vector<float> h_input(16, 1.0f), h_output(16, 0.0f);
    float *d_input, *d_output;
    // 分配输入 GPU 内存
    cudaMalloc(&d_input, h_input.size() * sizeof(float));
    // 分配输出 GPU 内存
    cudaMalloc(&d_output, h_output.size() * sizeof(float));

    for (int i = 0; i < 100; i++)
    {
        // Host → Device
        cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice);
        // 创建绑定
        void* bindings[] = {d_input, d_output};
        // 执行推理
        context->executeV2(bindings);
        // Device → Host
        cudaMemcpy(h_output.data(), d_output, h_output.size() * sizeof(float),
                   cudaMemcpyDeviceToHost);
    }
    std::cout << "✅ Done, first output: " << h_output[0] << std::endl;
    // 释放输入 GPU 内存
    cudaFree(d_input);
    cudaFree(d_output);
    // 释放执行上下文
    delete context;
    // 释放引擎
    delete engine;
    // 释放运行时
    delete runtime;
    // 释放序列化网络
    delete serialized;
    delete network;
    // 释放配置
    delete config;
    // 释放构建器
    delete builder;

    return 0;
}
