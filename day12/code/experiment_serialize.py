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
