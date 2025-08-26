# tensorrt_infer.py  —— TensorRT 10.x / FP16 / 单次与基准测试
import time
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # 初始化 CUDA 上下文

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

class TrtRunner:
    def __init__(self, engine_path="mlp_fp16.engine", input_shape=(1,1,28,28)):
        self.input_shape = tuple(input_shape)
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError("Failed to deserialize engine")

        self.ctx = self.engine.create_execution_context()

        # 发现 1 输入 / 1 输出（TRT 10 API）
        inputs, outputs = [], []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                inputs.append(name)
            else:
                outputs.append(name)
        assert len(inputs) == 1 and len(outputs) == 1, f"IO not 1/1: {inputs}, {outputs}"
        self.in_name, self.out_name = inputs[0], outputs[0]

        # 设输入形状并查询输出形状/类型
        self.ctx.set_input_shape(self.in_name, self.input_shape)
        self.out_shape = tuple(self.ctx.get_tensor_shape(self.out_name))
        self.out_dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(self.out_name)))

        # 预分配显存 + 绑定地址（复用）
        self.stream = cuda.Stream()
        self.d_in  = cuda.mem_alloc(int(np.prod(self.input_shape)) * 4)  # float32
        self.d_out = cuda.mem_alloc(int(np.prod(self.out_shape)) * self.out_dtype.itemsize)
        self.ctx.set_tensor_address(self.in_name, int(self.d_in))
        self.ctx.set_tensor_address(self.out_name, int(self.d_out))

    def infer_once(self, x_np):
        # H2D
        cuda.memcpy_htod_async(self.d_in, x_np, self.stream)
        # 执行
        ok = self.ctx.execute_async_v3(self.stream.handle)
        if not ok:
            raise RuntimeError("TensorRT execution failed")
        # D2H
        y = np.empty(self.out_shape, dtype=self.out_dtype)
        cuda.memcpy_dtoh_async(y, self.d_out, self.stream)
        self.stream.synchronize()
        return y

if __name__ == "__main__":
    # 准备输入
    x = np.random.randn(1,1,28,28).astype(np.float32)

    # 初始化 & 单次推理
    runner = TrtRunner("mlp_fp16.engine", input_shape=x.shape)
    y = runner.infer_once(x)
    print("out shape:", y.shape)
    print("out[0][:10]:", y.reshape(-1)[:10])

    # 预热 + 基准测试
    WARMUP, ITERS = 100, 1000
    for _ in range(WARMUP):
        runner.infer_once(x)

    t0 = time.perf_counter()
    for _ in range(ITERS):
        runner.infer_once(x)
    cuda.Context.synchronize()  # 再保险同步一次
    t1 = time.perf_counter()

    print(f"平均延迟: {(t1 - t0) / ITERS * 1000:.3f} ms  (batch=1)")

