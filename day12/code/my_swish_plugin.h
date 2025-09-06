#ifndef MY_WISH_PLUGIN_H
#define MY_WISH_PLUGIN_H

#include "NvInfer.h"
#include <cassert>
#include <cmath>
#include <string>
#include <vector>

using namespace nvinfer1;

class SwishPlugin : public IPluginV2DynamicExt
{
  public:
    SwishPlugin() {}
    SwishPlugin(const void* data, size_t length) {}

    // 1. 获取插件类型
    const char* getPluginType() const noexcept override
    {
        return "SwishPlugin";
    }

    // 2. 获取插件版本
    const char* getPluginVersion() const noexcept override
    {
        return "1";
    }

    // 3. 获取输出数量
    int getNbOutputs() const noexcept override
    {
        return 1;
    }

    // 4. 获取输出维度
    DimsExprs getOutputDimensions(int outputIndex, const DimsExprs* inputs, int nbInputs,
                                  IExprBuilder& exprBulider) noexcept override
    {
        return inputs[0]; // 输入维度与输出相同
    }

    // 5. 支持的组合
    bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs,
                                   int nbOutputs) noexcept override
    {
        return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT;
    }

    // 6. 配置插件
    void configurePlugin(const DynamicPluginTensorDesc* inputs, int nbInputs,
                         const DynamicPluginTensorDesc* outputs, int nbOutputs) noexcept override
    {
    }

    // 7. 获取 workspace 大小
    size_t getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs,
                            const PluginTensorDesc* outputs, int nbOutputs) const noexcept override
    {
        return 0; // 无需额外 workspace
    }

    // 8. 执行插件
    int enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
                const void* const* inputs, void* const* outputs, void* workspace,
                cudaStream_t stream) noexcept override;

    // 9. 获取序列化大小
    size_t getSerializationSize() const noexcept override
    {
        return 0;
    }

    // 10. 序列化插件
    void serialize(void* buffer) const noexcept override {}

    // 11. 初始化插件
    int initialize() noexcept override
    {
        return 0;
    }

    // 12. 终止插件
    void terminate() noexcept override {}

    // 13. 克隆插件
    IPluginV2DynamicExt* clone() const noexcept override
    {
        return new SwishPlugin();
    }

    // 14. 销毁插件
    void destroy() noexcept override
    {
        delete this;
    }

    // 15. 设置插件命名空间
    void setPluginNamespace(const char* pluginNamespace) noexcept override {}

    // 16. 获取插件命名空间
    const char* getPluginNamespace() const noexcept override
    {
        return "";
    }

    // 17. 获取输出数据类型
    DataType getOutputDataType(int index, const DataType* intputTypes,
                               int nbInputs) const noexcept override
    {
        return intputTypes[0];
    }

    // 18. 绑定到上下文
    void attachToContext(cudnnContext*, cublasContext*, IGpuAllocator*) noexcept override {}

    // 19. 从上下文分离
    void detachFromContext() noexcept override {}
};

class SwishPluginCreator : public IPluginCreator
{
  public:
    // 1. 获取插件名称
    const char* getPluginName() const noexcept override
    {
        return "SwishPlugin";
    }

    // 2. 获取插件版本
    const char* getPluginVersion() const noexcept override
    {
        return "1";
    }

    // 3. 获取插件字段
    const PluginFieldCollection* getFieldNames() noexcept override
    {
        return nullptr;
    }

    // 4. 创建插件
    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override
    {
        return new SwishPlugin();
    }

    // 5. 反序列化插件
    IPluginV2* deserializePlugin(const char* name, const void* serialData,
                                 size_t serialLength) noexcept override
    {
        return new SwishPlugin(serialData, serialLength);
    }

    // 6. 设置插件命名空间
    void setPluginNamespace(const char* pluginNamespace) noexcept override {}

    // 7. 获取插件命名空间
    const char* getPluginNamespace() const noexcept override
    {
        return "";
    }
};
#endif
