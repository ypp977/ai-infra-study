import ctypes
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# 加载插件库
ctypes.CDLL("./libswish.so")

# 日志
logger = trt.Logger(trt.Logger.INFO)

def build_swish_engine():
    # 创建 builder
    builder = trt.Builder(logger)
    # 创建网络
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    # 创建配置
    config = builder.create_builder_config()
    trt.init_libnvinfer_plugins(logger,'')

    input_tensor = network.add_input("input",trt.DataType.FLOAT,(-1,16))

    # 获取插件列表
    create_list = trt.get_plugin_registry().plugin_creator_list
    # 获取 Swish 插件
    swish_creator = [c for c in create_list if c.name == "SwishPlugin"][0]
    plugin = swish_creator.create_plugin("swish_layer",trt.PluginFieldCollection([]))
    swish_layer = network.add_plugin_v2([input_tensor],plugin)
    # 标记输出
    network.mark_output(swish_layer.get_output(0))

    # 创建优化 profile
    profile = builder.create_optimization_profile()
    # 设置输入形状
    profile.set_shape(input_tensor.name, (1,16),(8,16),(32,16))
    # 添加优化 profile
    config.add_optimization_profile(profile)

    # 序列化网络
    serialized_engine = builder.build_serialized_network(network,config)
    # 创建运行时
    runtime = trt.Runtime(logger)
    return runtime.deserialize_cuda_engine(serialized_engine)

engine = build_swish_engine()
# 创建执行上下文
context = engine.create_execution_context()

test_shapes = [(1,16),(8,16),(32,16)]

for shape in test_shapes:
    print("\n=====测试输入 shape:", shape, "=====")
    # 设置输入形状
    context.set_input_shape("input",shape)
    # 检查输入形状是否指定
    assert context.all_binding_shapes_specified

    # 生成随机输入
    inp = np.random.randn(*shape).astype(np.float32)
    # 创建输出
    out = np.empty_like(inp)

    # 分配输入 GPU 内存
    d_input = cuda.mem_alloc(inp.nbytes)
    d_output = cuda.mem_alloc(out.nbytes)

    # Host → Device
    cuda.memcpy_htod(d_input,inp)

    # 执行推理
    context.execute_v2([int(d_input),int(d_output)])

    # Device → Host
    cuda.memcpy_dtoh(out,d_output)

    print("输入 shape:", inp.shape)
    print("输出 shape:", out.shape)
    print("输入前5个值:", inp.flatten()[:5])
    print("输出前5个值:", out.flatten()[:5])

