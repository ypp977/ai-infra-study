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
