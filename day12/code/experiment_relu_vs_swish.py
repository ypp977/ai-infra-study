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
