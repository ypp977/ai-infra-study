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
