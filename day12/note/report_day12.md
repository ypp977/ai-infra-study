# Day12 - TensorRT Plugin 入门

------

## 🎯 学习目标

- 理解为什么/何时需要自定义 TensorRT Plugin
- 学习 **IPluginV2 / IPluginV2DynamicExt** 的接口与生命周期
- 实现一个简单的 **激活函数 Plugin（如 ReLU/Swish）**
- 掌握 Plugin 的序列化/反序列化、输入输出 shape 推断、workspace 管理
- 将 Plugin 集成到 TensorRT Engine 并运行推理

------

## 1️⃣ 代码实验（强化学习）

### 思路讲解

在 TensorRT 中，大部分常见算子（卷积、GEMM、激活）都有内置支持。但在以下场景中需要自定义 **Plugin**：

- **框架中有而 TensorRT 没有的算子**（如 Swish、Mish、LayerNorm 的某些变体）
- **需要特殊优化**（融合算子、减少访存、避免冗余 kernel）
- **研究性/实验性算子**（快速验证新结构）

实验目标：

1. 编写一个 **Swish Plugin**（Swish(x) = x * sigmoid(x)）。
2. 支持 **动态 shape**（使用 `IPluginV2DynamicExt`）。
3. 集成到 TensorRT 构建流程中，并对比 TensorRT 内置 ReLU。

------

### Plugin 实现核心代码

#### my_swish_plugin.h

```c++
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
    size_t getSerializationSize() const noexcept override {}

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

```

#### my_swish_plugin.cu

```c++
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
    swish_kernel<<<grid, block>>>(input, output, num);

    return 0;
}

```

### 编译与运行步骤

1. **准备 TensorRT 环境**

   ```bash
   docker run --gpus all -it --rm -v $PWD:/workspace  my-ai-infer:trt bash
   ```

2. **编译 Plugin**

   ```bash
   nvcc -I/usr/include/x86_64-linux-gnu -I/usr/include -shared -Xcompiler -fPIC my_swish_plugin.cu -o libswish.so
   ```

   ![image-20250906031439534](./report_day12.assets/image-20250906031439534.png)

3. **构建网络时注册 Plugin**

   ```python
   import ctypes
   import numpy as np
   import tensorrt as trt
   import pycuda.driver as cuda
   import pycuda.autoinit  # 自动初始化 CUDA 上下文
   
   # 1. 加载插件库
   ctypes.CDLL("./libswish.so")
   
   # 2. 初始化 TensorRT
   logger = trt.Logger(trt.Logger.INFO)
   builder = trt.Builder(logger)
   network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
   config = builder.create_builder_config()
   trt.init_libnvinfer_plugins(logger, '')
   
   # 3. 获取 SwishPlugin
   creator_list = trt.get_plugin_registry().plugin_creator_list
   swish_creator = [c for c in creator_list if c.name == "SwishPlugin"][0]
   plugin = swish_creator.create_plugin("swish_layer", trt.PluginFieldCollection([]))
   
   # 4. 构建网络：Input → Swish → Output
   input_tensor = network.add_input("input", trt.DataType.FLOAT, (1, 6))
   swish_layer = network.add_plugin_v2([input_tensor], plugin)
   network.mark_output(swish_layer.get_output(0))
   
   # 5. 构建 Engine
   serialized_engine = builder.build_serialized_network(network, config)
   runtime = trt.Runtime(logger)
   engine = runtime.deserialize_cuda_engine(serialized_engine)
   context = engine.create_execution_context()
   print("✅ Engine 构建成功")
   
   # 6. 准备输入
   inp = np.array([[1, 2, 3, -1, -2, -3]], dtype=np.float32)
   out = np.empty_like(inp)
   
   # 分配 GPU 内存
   d_input = cuda.mem_alloc(inp.nbytes)
   d_output = cuda.mem_alloc(out.nbytes)
   
   # Host → Device
   cuda.memcpy_htod(d_input, inp)
   
   # 在 execute_v2 之前设置 shape
   context.set_input_shape("input", (1, 6))
   
   assert context.all_binding_shapes_specified, "输入 shape 未指定"
   # 运行推理
   context.execute_v2([int(d_input), int(d_output)])
   
   # Device → Host
   cuda.memcpy_dtoh(out, d_output)
   
   print("输入:", inp)
   print("Swish 输出:", out)
   
   ```

4. **运行推理**

   ```bash
   python test_swish_plugin.py
   ```

   运行结果

   ![image-20250906043056053](./report_day12.assets/image-20250906043056053.png)

------

### Nsight Compute/Systems 关注指标

- **Kernel Launch 时间**（Plugin 是否额外增加开销）
- **Global memory throughput**（Swish 算子访存是否高效）
- **Occupancy**（线程块调度效率）
- **Stream overlap**（是否能与其他算子并行执行）

------

## 2️⃣ 深度追问

### 1. `IPluginV2DynamicExt` vs `IPluginV2` 的选型标准？

#### 🔍 IPluginV2 vs IPluginV2DynamicExt 对比

| 特性                | **IPluginV2**                                           | **IPluginV2DynamicExt**                                      |
| ------------------- | ------------------------------------------------------- | ------------------------------------------------------------ |
| **引入版本**        | TensorRT 6 以前主要用                                   | TensorRT 6+ 推荐使用                                         |
| **输入输出 shape**  | **静态 shape**，构建 engine 时固定                      | **动态 shape**，支持显式 batch 和动态输入                    |
| **精度/数据类型**   | 只支持固定的数据类型（FP32 为主）                       | 支持 FP32/FP16/INT8 等，能灵活指定                           |
| **调用的关键函数**  | - `getOutputDimensions`（固定维度） - `enqueue`（执行） | - `getOutputDimensions`（动态 shape） - `getOutputDataType`（输出 dtype） - `enqueue`（执行） |
| **显式 batch 支持** | 不支持（只能隐式 batch 模式）                           | 完全支持显式 batch（`EXPLICIT_BATCH`）                       |
| **推荐程度**        | **过时**（仅在老版本 TensorRT 兼容场景下用）            | **推荐**（动态 shape、新项目都用这个）                       |

------

#### ✅ 什么时候用 IPluginV2？

- 你的 TensorRT engine 是 **老版本构建的（隐式 batch 模式）**。
- 输入输出的 shape 是固定的，不需要动态 batch。
- 只是写一个简单的小算子，环境里还在跑 TensorRT < 6。

👉 例如：老的 TensorRT 例子里写的 `IPluginV2`，只能用 `getOutputDimensions()` 固定返回输出 shape。

------

#### ✅ 什么时候用 IPluginV2DynamicExt？

- 你的网络是 **显式 batch (EXPLICIT_BATCH)** 构建的（TensorRT 6 之后默认推荐显式 batch）。
- 需要支持 **动态输入尺寸**（如 batch=1,2,4 或输入分辨率变化）。
- 想利用 **混合精度（FP16/INT8）**，而不仅仅是 FP32。

👉 例如：你写 Swish Plugin、GELU Plugin，或者想在 TensorRT 中自定义 LayerNorm，这些都要支持动态 shape → 必须用 `IPluginV2DynamicExt`。

------

#### 📌 总结

- **新项目 → 一律用 `IPluginV2DynamicExt`** ✅
- **旧代码/隐式 batch → 还能用 `IPluginV2`**
- TensorRT 官方文档也建议未来都迁移到 **DynamicExt**

### 2. 支持的数据格式与 layout 转换的代价？

#### 🔎 1. 支持的数据格式（TensorFormat）

在 TensorRT 里，每个算子或插件都要声明它能接受的数据格式：

- **最常见的**
  - `kLINEAR` → 普通连续内存 (NCHW)
  - `kCHW4` → 每 4 个 channel 打包 (对 INT8 常见)
  - `kCHW32` → 每 32 个 channel 打包 (Ampere Tensor Core 友好)
  - `kHWC8` / `kHWC16` → NHWC 排列，每个元素再按 8/16 对齐
- **浮点数精度**
  - `kFLOAT` (FP32)
  - `kHALF` (FP16)
  - `kINT8` (量化)

👉 插件要在 `supportsFormatCombination()` 里显式声明，比如：

```c++
bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut,
                               int nbInputs, int nbOutputs) noexcept override {
    const PluginTensorDesc& desc = inOut[pos];
    if (pos == 0) {
        // 输入 0：支持 FP32 + LINEAR
        return desc.format == TensorFormat::kLINEAR && desc.type == DataType::kFLOAT;
    } else {
        // 输出和输入保持一致
        return desc.format == inOut[0].format && desc.type == inOut[0].type;
    }
}
```

------

#### 🔎 2. Layout 转换的开销

如果你的插件只支持 **kLINEAR (NCHW)**，但上游算子输出的是 **kCHW32 (Tensor Core 优化格式)**，TensorRT 会在 **插件前后插入一个“reformat kernel”** 做 layout 转换。

##### 代价：

- **额外 kernel 启动**（launch overhead）
- **额外全量内存拷贝**（global memory bandwidth 消耗）
- **不能和前后算子融合**（pipeline 中断）

在 Nsight Systems 里就会看到 **额外的 memcpy-like kernel**，导致 timeline 出现 **gap**。

------

#### 🔎 3. 最佳实践

1. **尽量支持多种格式**
   - 对于卷积类/激活类算子，至少支持：
     - FP32 + kLINEAR
     - FP16 + kLINEAR
     - FP16 + kCHW32
   - 这样 TensorRT 就不需要插入额外的转换 kernel。
2. **输入输出保持一致**
   - 插件输出格式最好和输入相同，避免再做一次 reformat。
3. **按需对齐内存**
   - 如果写的是算子（比如 Swish、LayerNorm），内部计算其实和 layout 无关，可以写一个 kernel，直接支持 kLINEAR/kCHW32 两种。
   - 判断 `TensorFormat` 后选择不同 kernel。

------

#### 🔎 4. 举例

假设你实现了一个 Swish Plugin：

- 如果只写：

  ```c++
  return desc.format == kLINEAR && desc.type == kFLOAT;
  ```

  👉 那么 FP16 推理时，TensorRT 会自动插入 `linear->chw32` 和 `chw32->linear` 两个转换 kernel，性能掉 20–30%。

- 如果改写成：

  ```c++
  if (desc.type == kFLOAT && desc.format == kLINEAR) return true;
  if (desc.type == kHALF && (desc.format == kLINEAR || desc.format == kCHW32)) return true;
  ```

  👉 那么 FP16 Tensor Core 流水线可以 **直接过 plugin**，没有额外开销。

------

#### 🔎 5. 总结

- 插件必须声明支持的 **DataType + TensorFormat** 组合。
- 如果插件只支持一种格式，TensorRT 会自动插入 **layout 转换 kernel**，带来额外耗时和显存带宽开销。
- **最佳实践**：插件要尽可能支持常见格式（特别是 FP16 + CHW32），并保证输入输出格式一致，以避免额外 memcpy。

### 3. plugin 的序列化兼容性如何保障？

#### 🔎 1. TensorRT Plugin 序列化原理

- **Engine 序列化**：TensorRT 会把网络结构 + 权重 + 插件参数打包成一个 `ICudaEngine` 对象，存到磁盘。
- **Plugin 序列化**：Engine 里不会存完整的 Plugin C++ 实现代码，只会存你在 `serialize()` 写进去的**二进制参数**。
- **反序列化**：TensorRT runtime 加载 engine 文件后，会调用你注册的 `deserializePlugin()`，并把 `serialize()` 存的字节传回来，由插件重新构造对象。

👉 所以 **序列化/反序列化必须完全对称**。

------

#### 🔎 2. 序列化兼容性风险

1. **字段遗漏**
   - 如果你在 `serialize()` 里忘了写某个参数，反序列化时就会用默认值，可能导致结果错误。
2. **顺序不一致**
   - `serialize()` 写的字段必须和 `deserializePlugin()` 读取的顺序一模一样，否则会解读错数据。
3. **跨版本兼容性**
   - 你 plugin 第 1 版只存了 2 个参数，第 2 版又加了新参数。旧 engine 文件在新版本反序列化时就会出错。
4. **跨平台/跨编译器差异**
   - 直接用 `memcpy` 序列化 struct 可能因 **对齐方式 / endianness** 不同导致不兼容。

------

#### 🔎 3. 保障序列化兼容性的方法

✅ **1. 显式写入每个字段**
 不要 `memcpy` 整个 struct，而是逐个写入：

```c++
void serialize(void* buffer) const noexcept override {
    char* d = reinterpret_cast<char*>(buffer);
    write(d, mInputSize);
    write(d, mAlpha);
}

SwishPlugin(const void* data, size_t length) {
    const char* d = reinterpret_cast<const char*>(data);
    read(d, mInputSize);
    read(d, mAlpha);
}
```

✅ **2. 定义版本号**
 在 serialize 的开头存一个 plugin 内部版本号，反序列化时先读它：

```c++
static constexpr int PLUGIN_VERSION = 1;

void serialize(void* buffer) const noexcept override {
    char* d = reinterpret_cast<char*>(buffer);
    write(d, PLUGIN_VERSION);
    write(d, mAlpha);
}

SwishPlugin(const void* data, size_t length) {
    const char* d = reinterpret_cast<const char*>(data);
    int version;
    read(d, version);
    if (version != PLUGIN_VERSION) throw std::runtime_error("Plugin version mismatch!");
    read(d, mAlpha);
}
```

✅ **3. 保持字节对齐独立性**
 序列化时手动用 `memcpy` 写 `float`、`int`，不要写整个 struct，避免编译器填充。

✅ **4. 引入 `PluginFieldCollection` 参数机制**
 如果你要支持 **运行时动态配置**，建议在 `createPlugin()` 里解析 `PluginFieldCollection`，保持和 ONNX 转换端一致。

✅ **5. 回归测试**

- 保存 engine 文件，在不同 TensorRT 版本 / 不同机器反序列化测试。
- 确保结果一致。

------

#### 🔎 4. 最佳实践总结

1. 在 `serialize()` / `deserialize()` 里 **严格一一对应**字段。
2. 增加 **内部版本号**，避免老引擎在新插件里解析错误。
3. 不直接 dump struct，要逐字段写入。
4. 多环境测试（不同 GPU / TRT 版本），确保 engine 可移植性。

### 4. 多线程/多实例安全问题？

#### 🔎 1. 为什么要考虑线程安全？

- TensorRT 的 **Engine/ExecutionContext** 在运行时可以被多个线程同时调用。
- 插件的 `enqueue()` 可能被多个线程同时执行（不同 context 或 batch）。
- 如果插件内部用了 **全局变量/静态变量**，或者没有做好同步，就会产生数据竞争 → crash 或结果错误。

------

#### 🔎 2. 常见问题点

1. **全局/静态变量**

   ```c++
   static float buffer[1024]; // ❌ 多线程会互相覆盖
   ```

   → 每个线程写同一个内存区域，导致结果错乱。

2. **在插件内部 malloc/free**

   - 如果在 `enqueue()` 里频繁 `cudaMalloc/cudaFree`，不仅性能差，还可能线程间相互干扰。

3. **共享资源未加锁**

   - 例如多线程都访问某个 lookup table，而没加互斥保护。

4. **Plugin Creator 单例模式问题**

   - `IPluginCreator` 是全局注册的，如果里面保存了状态（比如参数），不同 plugin 实例就会互相干扰。

------

#### 🔎 3. 保证安全的原则

✅ **1. 插件对象不要用全局变量**

- 每个 `IPluginV2DynamicExt` 实例只保存自己的参数（如 alpha, beta）。
- 避免使用 `static` 数据。

✅ **2. 参数序列化到 Engine**

- 插件参数必须在 `serialize()` 里保存 → 反序列化时恢复。
- 这样每个 Engine 里都有独立副本。

✅ **3. 避免在 enqueue 里动态分配内存**

- 如果需要临时空间，使用 `getWorkspaceSize()` → TensorRT 会帮你分配每次执行的独立 workspace。

✅ **4. 确保 CUDA kernel 是无副作用的**

- kernel 只读输入、只写输出，不访问全局共享内存。
- 如果必须共享数据（比如 LUT 表），把它放到 `__constant__` 或 `__device__` 内存，并保证只读。

✅ **5. 多实例安全**

- 一个 Engine 里可能有多个 SwishPlugin 层，它们必须互不影响。
- 所以插件类成员变量只保存自己实例需要的东西。

✅ **6. Host 端多线程安全**

- TensorRT runtime 本身是线程安全的，但如果你在插件里做了 `std::cout`、写文件、操作全局资源，就要自己加锁。

------

#### 🔎 4. 最佳实践代码片段

```c++
class SwishPlugin : public IPluginV2DynamicExt {
public:
    SwishPlugin(float alpha = 1.0f) : mAlpha(alpha) {}

    // 每个实例有独立的参数
    float mAlpha;

    int enqueue(const PluginTensorDesc* inputDesc,
                const PluginTensorDesc* outputDesc,
                const void* const* inputs,
                void* const* outputs,
                void* workspace,
                cudaStream_t stream) noexcept override 
    {
        // ✅ 不使用全局/静态变量
        // ✅ kernel 只访问 inputs/outputs
        int num = volume(inputDesc[0].dims);
        swish_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const float*>(inputs[0]),
            reinterpret_cast<float*>(outputs[0]),
            num, mAlpha);
        return 0;
    }

    size_t getWorkspaceSize(...) const noexcept override {
        return 0; // ✅ 如果需要临时 buffer，用这里申请
    }
};
```

------

#### 🔎 5. 总结

1. **不要用全局变量** → 每个插件实例独立。
2. **不要在 enqueue 里 malloc/free** → 用 workspace。
3. **kernel 无副作用** → 输入输出完全独立。
4. **多线程调用安全** → 插件内状态不可共享/修改全局资源。
5. **PluginCreator 只做工厂**，不要存储插件参数。

### 5. shape 推断与动态维度边界检查？

#### 🔎 1. Shape 推断 (Shape Inference)

在 TensorRT 里，插件需要告诉框架 **输出张量的 shape**，否则 engine 构建不下去。

- **`IPluginV2`**（老接口）
  - 用 `Dims getOutputDimensions(int index, const Dims* inputs, int nbInputs)`
  - 输入输出 shape 必须是 **固定维度**，不支持动态。
- **`IPluginV2DynamicExt`**（推荐接口）
  - 用 `DimsExprs getOutputDimensions(int index, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder)`
  - 输入 shape 通过 **符号表达式** 表示，可以支持 `-1`（动态维度）。
  - 你可以在里面写规则，比如：
    - `output = inputs[0]`（保持一致）
    - `output = concat(inputs[0].d[1], inputs[1].d[1])`（拼接）
    - `output = exprBuilder.operation(DimensionOperation::kPROD, …)`（乘法/除法组合维度）

👉 作用：TensorRT engine build 时，会自动推断 shape 变化，把插件当成一个合法节点处理。

------

#### 🔎 2. 动态维度边界检查

动态 shape 带来的问题是：运行时用户可能传入 **不合法的 shape**（比如 batch 太大、channel 不对），如果插件没有检查，就可能 crash。

- **入口：`configurePlugin()`**
   在 engine build / runtime 初始化时，TensorRT 会调用 `configurePlugin()`，传入 **输入/输出的实际维度范围**。
   插件可以在这里做检查：

  ```c++
  void configurePlugin(const DynamicPluginTensorDesc* in, int nbInputs,
                       const DynamicPluginTensorDesc* out, int nbOutputs) noexcept override {
      int channels = in[0].desc.dims.d[1];
      if (channels <= 0 || channels > 1024) {
          throw std::runtime_error("Invalid input channel size");
      }
  }
  ```

- **边界检查的目的**

  - 确认输入维度满足算子逻辑（例如 Swish 允许任意 shape，但某些算子要求 channel 必须是 8 的倍数）。
  - 提前报错（engine build 阶段），避免 runtime 崩溃。

- **运行时检查**

  - 在 `enqueue()` 里也可以读取 `inputDesc[0].dims`，做 **最后一道保险**。
  - 如果发现不合法，返回 `-1`，TensorRT 会报错退出。

------

#### 🔎 3. 最佳实践总结

1. **Shape 推断**
   - 在 `getOutputDimensions()` 里写清楚规则，保证输出维度和输入一致或符合逻辑。
   - 动态 shape 用 `DimsExprs` + `IExprBuilder` 表达。
2. **边界检查**
   - 在 `configurePlugin()` 做静态检查（合法范围、对齐约束）。
   - 在 `enqueue()` 再做一次运行时校验（防止意外输入）。
3. **错误处理**
   - 如果 shape 不合法，要尽早报错，而不是 silent fail。
   - 推荐用 `assert` 或返回 `-1` 让 TensorRT 停止执行。

------

✅ 总结一句话：

- **Shape 推断** → 保证 TensorRT 能在 build 阶段正确知道输出尺寸。
- **动态维度边界检查** → 保证运行时输入 shape 合法，避免 kernel 崩溃。

### 6. plugin 错误处理策略？

#### 🔎 TensorRT Plugin 错误处理策略

1. **返回错误码**
   - `enqueue()` 返回 `-1` → TRT 会报错并中止执行。
2. **日志提示**
   - 用 `printf` / `std::cerr` 或自定义 `Logger` 打印错误，便于定位。
3. **断言/校验**
   - `assert()` 或手动检查 shape / dtype，不合法时立刻退出。
4. **序列化安全**
   - 在 `deserializePlugin()` 检查版本号/参数合法性，不对就报错。
5. **构建期检查**
   - 在 `configurePlugin()` 阶段验证输入输出范围，提早发现问题。

👉 总结：**构建期检查，运行时报错返回，必要时打印日志**。这样既能避免 silent fail，也方便排查。

------

## 3️⃣ 实验部分

### 🧪 实验 1：Swish Plugin vs 内置 ReLU

#### 1️⃣ 实验目标

- 比较 TensorRT 内置 ReLU 和 自定义 Swish 插件的运行延迟。
- 验证 Swish 插件功能是否正确。

------

#### 2️⃣ 实验方法

1. 输入固定大小 `(1, 1024)` 的随机数据。
2. 构建两个 TensorRT Engine：
   - **Engine A**：Input → ReLU → Output
   - **Engine B**：Input → SwishPlugin → Output
3. 使用 `time.perf_counter()` 多次运行，取平均延迟。
4. 对比结果。

------

#### 3️⃣ Python 实验代码

保存为 `experiment_relu_vs_swish.py`：

```python
import ctypes
import time
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# 加载插件库
ctypes.CDLL("./libswish.so")

# TRT Logger
logger = trt.Logger(trt.Logger.INFO)

# 通用函数：构建 engine
def build_engine(use_relu = True):
    # 创建 builder
    builder = trt.Builder(logger)
    # 创建网络
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    # 创建配置
    config = builder.create_builder_config()
    # 初始化插件
    trt.init_libnvinfer_plugins(logger,'')

    # 创建输入
    input_tensor = network.add_input("input",trt.DataType.FLOAT,(1,1024))

    if(use_relu):
        # 创建 ReLU 层
        relu_layer = network.add_activation(input_tensor,trt.ActivationType.RELU)
        # 标记输出
        network.mark_output(relu_layer.get_output(0))
    else:
        # 获取 Swish 插件
        creator_list = trt.get_plugin_registry().plugin_creator_list
        swish_creator = [c for c in creator_list if c.name == "SwishPlugin"][0]
        # 创建 Swish 层
        plugin = swish_creator.create_plugin("swish_layer", trt.PluginFieldCollection([]))
        swish_layer = network.add_plugin_v2([input_tensor], plugin)
        # 标记输出
        network.mark_output(swish_layer.get_output(0))

    # 序列化网络
    serialized_engine = builder.build_serialized_network(network, config)
    # 创建运行时
    runtime = trt.Runtime(logger)
    # 反序列化网络
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    return engine

# 构建 ReLU 引擎
relu_engine = build_engine(use_relu=True)
# 构建 Swish 引擎
swish_engine = build_engine(use_relu=False)

# 创建 ReLU 执行上下文
context_relu = relu_engine.create_execution_context()
# 创建 Swish 执行上下文
context_swish = swish_engine.create_execution_context()

# 设置 ReLU 输入形状
context_relu.set_input_shape("input", (1, 1024))
# 设置 Swish 输入形状
context_swish.set_input_shape("input", (1, 1024))

# 准备输入
inp = np.random.randn(1, 1024).astype(np.float32)
# 创建 ReLU 输出
out_relu = np.empty_like(inp)
# 创建 Swish 输出
out_swish = np.empty_like(inp)

# 分配 GPU 内存
d_input = cuda.mem_alloc(inp.nbytes)
# 分配 ReLU 输出 GPU 内存
d_output_relu = cuda.mem_alloc(out_relu.nbytes)
# 分配 Swish 输出 GPU 内存
d_output_swish = cuda.mem_alloc(out_swish.nbytes)

# Host → Device
cuda.memcpy_htod(d_input, inp)

# 执行函数
def run_infer(context, d_input, d_output,n_iters=50):
    # 记录开始时间
    start = time.perf_counter()
    # 执行推理
    for _ in range(n_iters):
        context.execute_v2([int(d_input), int(d_output)])
    cuda.Context.synchronize()
    # 记录结束时间
    end = time.perf_counter()
    return (end - start) / n_iters

# 测试 ReLU
lat_relu = run_infer(context_relu, d_input, d_output_relu)
# 复制 ReLU 输出到 Host
cuda.memcpy_dtoh(out_relu, d_output_relu)

# 测试 Swish
lat_swish = run_infer(context_swish, d_input, d_output_swish)
# 复制 Swish 输出到 Host
cuda.memcpy_dtoh(out_swish, d_output_swish)

# 打印结果
print("输入实例：", inp[0][:5])
print("ReLU 输出：", out_relu[0][:5])
print("Swish 输出：", out_swish[0][:5])
print(f"ReLU 平均延迟: {lat_relu*1000:.3f} ms")
print(f"Swish 平均延迟: {lat_swish*1000:.3f} ms")

```

------

#### 4️⃣ 运行步骤

```bash
python experiment_relu_vs_swish.py
```

------

#### 5️⃣ 结果

输出如图所示（不同 GPU 会有差异）：

![image-20250906175017761](./report_day12.assets/image-20250906175017761.png)

- **数值正确性**：ReLU 把负数变成 0，Swish 平滑抑制负数。
- **性能结果**：Swish 插件比内置 ReLU 略慢（因为是自定义 CUDA kernel，没有 cuDNN/TensorRT 优化）。

------

### 🧪 实验 2：动态 Shape 测试

#### 1️⃣ 实验目标

- 验证 `IPluginV2DynamicExt` 的功能：输入不同 batch size 时，输出 shape 是否自动匹配。
- 测试输入 `(1,16)`、`(8,16)`、`(32,16)`。

------

#### 2️⃣ 实验方法

1. 构建一个简单网络：`Input → SwishPlugin → Output`。
2. 使用 `context.set_binding_shape()` 设置不同的输入 shape。
3. 执行推理并打印输入/输出 shape，验证是否一致。

------

#### 3️⃣ Python 实验代码

保存为 `experiment_dynamic_shape.py`：

```
import ctypes
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# 加载插件库
ctypes.CDLL("./libswish.so")

logger = trt.Logger(trt.Logger.INFO)

# 构建 Swish engine
def build_swish_engine():
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    trt.init_libnvinfer_plugins(logger, '')

    # 输入定义为动态 batch
    input_tensor = network.add_input("input", trt.DataType.FLOAT, (-1, 16))

    # 插入 Swish 插件
    creator_list = trt.get_plugin_registry().plugin_creator_list
    swish_creator = [c for c in creator_list if c.name == "SwishPlugin"][0]
    plugin = swish_creator.create_plugin("swish_layer", trt.PluginFieldCollection([]))
    swish_layer = network.add_plugin_v2([input_tensor], plugin)

    network.mark_output(swish_layer.get_output(0))

    # 构建 engine
    serialized_engine = builder.build_serialized_network(network, config)
    runtime = trt.Runtime(logger)
    return runtime.deserialize_cuda_engine(serialized_engine)

# 构建 engine 和 context
engine = build_swish_engine()
context = engine.create_execution_context()

# 测试不同 shape
test_shapes = [(1, 16), (8, 16), (32, 16)]

for shape in test_shapes:
    print("\n===== 测试输入 shape:", shape, "=====")
    context.set_binding_shape(0, shape)
    assert context.all_binding_shapes_specified

    # 准备输入
    inp = np.random.randn(*shape).astype(np.float32)
    out = np.empty_like(inp)

    # 分配显存
    d_input = cuda.mem_alloc(inp.nbytes)
    d_output = cuda.mem_alloc(out.nbytes)

    cuda.memcpy_htod(d_input, inp)

    # 执行
    context.execute_v2([int(d_input), int(d_output)])

    cuda.memcpy_dtoh(out, d_output)

    print("输入 shape:", inp.shape)
    print("输出 shape:", out.shape)
    print("输入前5个值:", inp.flatten()[:5])
    print("输出前5个值:", out.flatten()[:5])
```

------

#### 4️⃣ 运行步骤

```
python experiment_dynamic_shape.py
```

------

#### 5️⃣ 预期结果

输出：

![image-20250906183739624](./report_day12.assets/image-20250906183739624.png)

------

### 🧪 实验 3：序列化/反序列化

#### 1️⃣ 实验目标

- 测试 TensorRT Engine 的 **持久化能力**。
- 验证 `SwishPlugin` 的序列化接口是否正确实现。
- 确认 **保存前后推理结果一致**。

------

#### 2️⃣ 实验方法

1. 构建带 `SwishPlugin` 的 Engine。
2. 序列化为二进制 `.engine` 文件并写入磁盘。
3. 重新加载 `.engine`，创建 ExecutionContext。
4. 在相同输入下运行推理，对比输出。

------

#### 3️⃣ Python 实验代码

保存为 `experiment_serialize.py`：

```python
import ctypes
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# 加载插件库
ctypes.CDLL("./libswish.so")

# 日志
logger = trt.Logger(trt.Logger.INFO)

def build_engine():
    # 创建 builder
    builder = trt.Builder(logger)
    # 创建网络
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    # 创建配置
    config = builder.create_builder_config()
    # 初始化插件
    trt.init_libnvinfer_plugins(logger,'')

    # 创建输入
    input_tensor = network.add_input("input",trt.DataType.FLOAT,(1,16))

    # 获取插件列表
    creator_list = trt.get_plugin_registry().plugin_creator_list
    # 获取 Swish 插件
    swish_creator = [c for c in creator_list if c.name == "SwishPlugin"][0]
    # 创建 Swish 插件
    plugin = swish_creator.create_plugin("swish_layer",trt.PluginFieldCollection([]))
    # 创建 Swish 层
    swish_layer = network.add_plugin_v2([input_tensor],plugin)
    # 标记输出
    network.mark_output(swish_layer.get_output(0))

    # 序列化网络
    serialized_engine = builder.build_serialized_network(network,config)
    # 创建运行时
    runtime = trt.Runtime(logger)
    # 反序列化网络
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    # 返回引擎和序列化后的网络
    return engine,serialized_engine

# 构建引擎
engine,serialized_engine = build_engine()

# 保存引擎
with open("swish.engine","wb") as f:
    f.write(serialized_engine)
print("✅ Engine 已保存到 swish.engine")

# 加载引擎
with open("swish.engine","rb") as f:
    engine_data = f.read()

# 创建运行时
runtime = trt.Runtime(logger)
# 反序列化网络
engine_loaded = runtime.deserialize_cuda_engine(engine_data)
print("✅ Engine 已从 swish.engine 加载成功")

# 运行推理
def run_infer(engine,inp):
    # 创建执行上下文
    context = engine.create_execution_context()
    # 设置输入形状
    context.set_binding_shape(0,inp.shape)
    # 创建输出
    out = np.empty_like(inp)
    # 创建输入 GPU 内存
    d_input = cuda.mem_alloc(inp.nbytes)
    # 创建输出 GPU 内存
    d_output = cuda.mem_alloc(out.nbytes)

    # Host → Device
    cuda.memcpy_htod(d_input,inp)
    # 执行推理
    context.execute_v2([int(d_input),int(d_output)])
    # Device → Host
    cuda.memcpy_dtoh(out,d_output)
    return out

# 创建输入
inp = np.random.randn(1,16).astype(np.float32)

# 运行推理
out_before = run_infer(engine,inp)
out_after = run_infer(engine_loaded,inp)

print("输入:", inp[0, :5])
print("保存前输出:", out_before[0, :5])
print("保存后输出:", out_after[0, :5])

# 计算最大差异
diff = np.max(np.abs(out_before - out_after))
print("最大差异:", diff)

```

------

#### 4️⃣ 运行步骤

```bash
python experiment_serialize.py
```

------

#### 5️⃣ 结果

输出：

![image-20250906200856540](./report_day12.assets/image-20250906200856540.png)

- ✅ 输出 shape 正确
- ✅ 保存前后输出完全一致
- ✅ 说明 `SwishPlugin` 的序列化/反序列化逻辑正确

------

### 🧪 实验 4：多线程并发

#### 1️⃣ 实验目标

- 检查 `SwishPlugin` 在多线程环境下是否安全。
- 多个线程同时运行推理，验证无 crash。
- 对比不同线程数下的推理性能，观察是否接近线性提升。

------

#### 2️⃣ 实验方法

1. 构建同一个 `SwishPlugin` engine。
2. 用 `threading.Thread` 启动多个推理线程。
3. 每个线程执行多次推理并计时。
4. 比较 **单线程 vs 多线程** 的平均耗时。

------

#### 3️⃣ Python 实验代码

保存为 `experiment_multithread.py`：

```python
import ctypes
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.tools as cuda_tools
import pycuda.autoinit
import threading,time

# 加载插件
ctypes.CDLL("./libswish.so")

# 创建日志器
logger = trt.Logger(trt.Logger.INFO)

def build_engine():
    # 创建构建器
    builder = trt.Builder(logger)
    # 创建网络
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    # 创建配置
    config = builder.create_builder_config()
    # 初始化插件
    trt.init_libnvinfer_plugins(logger,'')

    # 创建输入
    input_tensor = network.add_input("input",trt.DataType.FLOAT,(1,16))
    # 获取插件列表
    creator_list = trt.get_plugin_registry().plugin_creator_list
    # 获取 Swish 插件
    swish_creator = [c for c in creator_list if c.name == "SwishPlugin"][0]
    # 创建 Swish 插件
    plugin = swish_creator.create_plugin("swish_layer",trt.PluginFieldCollection([]))
    # 创建 Swish 层
    swish_layer = network.add_plugin_v2([input_tensor],plugin)
    # 标记输出
    network.mark_output(swish_layer.get_output(0))

    # 序列化网络
    serialized_engine = builder.build_serialized_network(network,config)
    # 创建运行时
    runtime = trt.Runtime(logger)
    # 反序列化网络
    return runtime.deserialize_cuda_engine(serialized_engine)

# 创建引擎
engine = build_engine()

def run_infer(thread_id,n_iters=50):
    # 创建 CUDA 上下文
    ctx = cuda.Device(0).make_context()
    try:
        # 创建 TensorRT 执行上下文
        trt_context = engine.create_execution_context()
        # 创建输入
        inp = np.random.randn(1,16).astype(np.float32)
        # 创建输出
        out = np.empty_like(inp)

        # 创建输入 GPU 内存
        d_input = cuda.mem_alloc(inp.nbytes)
        # 创建输出 GPU 内存
        d_output = cuda.mem_alloc(out.nbytes)

        # 记录开始时间
        start = time.perf_counter()

        for _ in range(n_iters):
            # Host → Device
            cuda.memcpy_htod(d_input,inp)
            # 执行推理
            trt_context.execute_v2([int(d_input),int(d_output)])
            # Device → Host
            cuda.memcpy_dtoh(out,d_output)
        # 同步 CUDA 上下文
        cuda.Context.synchronize()
        # 记录结束时间
        end = time.perf_counter()

        # 计算平均耗时
        avg_time = (end - start) / n_iters * 1000
        print(f"[线程 {thread_id}] 平均耗时: {avg_time:.3f} ms")
    finally:
        # 弹出 CUDA 上下文
        ctx.pop()

def test_multithread(n_threads=4):
    # 创建线程列表
    threads = []
    # 记录开始时间
    start = time.perf_counter()
    # 创建线程
    for i in range(n_threads):
        t = threading.Thread(target=run_infer,args=(i,))
        # 启动线程
        t.start()
        # 添加线程到列表
        threads.append(t)
    # 等待所有线程完成
    for t in threads:
        # 等待线程完成
        t.join()
    # 记录结束时间
    end = time.perf_counter()
    print(f"🔥 {n_threads} 线程总耗时: {(end - start)*1000:.2f} ms")

if __name__ == "__main__":
    # 单线程测试
    print("=====单线程====")
    test_multithread(1)

    # 双线程测试
    print("\n====双线程====")
    test_multithread(2)

    # 四线程测试
    print("\n====四线程====")
    test_multithread(4)

```

------

#### 4️⃣ 运行方式

```bash
python experiment_multithread.py
```

------

#### 5️⃣ 预期结果

输出（不同 GPU 会有差异）：

![image-20250906231814316](./report_day12.assets/image-20250906231814316.png)

------

#### 🎯 结论

- **功能正确性**：SwishPlugin 在多线程环境下运行稳定，无 crash。
- **性能表现**：单个线程内的延迟基本稳定。
- **吞吐提升**：多线程能提高整体吞吐，但由于 GPU 是共享资源，速度不会严格线性提升（受限于 SM 资源和上下文切换）。

------

### 🧪 实验 5：性能 Profiling

####  1️⃣ 实验目标

- 使用 **Nsight Systems (`nsys`)** 查看 **整体时间线**
- 使用 **Nsight Compute (`ncu`)** 查看 **单个 kernel 详情**
- 重点关注：
  - kernel 调度是否连续（有没有空洞）
  - 是否存在额外 **cudaMemcpy**
  - 是否和其他算子 overlap

------

#### 2️⃣  准备一个独立的可执行程序

之前我们用的是 Python 测试，现在我们要写一个 **C++ 程序 `plugin_test.cpp`**，它会：

1. 加载 `libswish.so`
2. 构建一个最简单的 TensorRT engine（输入 → Swish → 输出）
3. 运行几次推理

这样 `nsys` 和 `ncu` 才能 profile 出 kernel。

```c++
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

```

------

#### 3️⃣ 编译

```bash
g++ plugin_test.cpp -o plugin_test \
  -I/usr/include/x86_64-linux-gnu \
  -I/usr/local/cuda/include \
  -L/usr/lib/x86_64-linux-gnu \
  -L/usr/local/cuda/lib64 \
  -lnvinfer -lnvonnxparser -lcudart -ldl
```

> 注意：把 `/path/to/TensorRT/` 换成你容器里的实际路径，比如 `/usr/lib/x86_64-linux-gnu/`.

------

#### 4️⃣ 用 Nsight Systems 跑

```bash
nsys profile -o profile_report ./plugin_test
```

生成文件：`profile_report.qdrep`

然后你可以下载到本地用 **Nsight Systems GUI** 打开，查看：

- Timeline 上的 **kernel 调度**
- 是否有 **cudaMemcpy**
- 是否有空隙（GPU idle）

------

#### 5️⃣ 用 Nsight Compute 跑

```bash
ncu --set full --target-processes all ./plugin_test
```

这会输出：

- 每个 kernel 的耗时
- Occupancy、warp divergence
- Memory throughput

------

#### 6️⃣ 分析指标

重点关注：

- **Warp Divergence** < 5%（激活函数算子应该很低）
- **Memory Throughput** 接近理论带宽
- **Shared Memory Utilization**（如果 Swish 用到了）
- 是否有 **额外 memcpy**（插件内部不应再有）

------

## ✅ 总结

1. Plugin 是扩展 TensorRT 的关键机制，用于支持 **未内置算子** 或 **特殊优化**。
2. 本文实现了一个 **Swish Plugin**，完整覆盖了接口、序列化、动态 shape、推理。
3. 通过实验验证了 Plugin 的功能、性能与兼容性。
4. Nsight 分析能帮助定位 Plugin 的内存/算力瓶颈。
5. 最佳实践：**避免全局变量、保证序列化兼容、测试动态 shape、安全并发**。
6. 下一步可以尝试 **更复杂 Plugin（如 LayerNorm、Attention）**，并与内置算子性能对比。