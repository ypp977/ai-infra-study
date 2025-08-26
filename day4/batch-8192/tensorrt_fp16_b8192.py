# tensorrt_fp16_b8192.py  —— TensorRT 10.x / FP16 / 批量=8192
import time, numpy as np, tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # 初始化 CUDA 上下文

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
BATCH = 8192
SHAPE = (BATCH, 1, 28, 28)
WARMUP, ITERS = 5, 20
np.random.seed(123)

class TrtRunner:
    def __init__(self, engine_path="mlp_fp16_b8192.engine", input_shape=SHAPE):
        self.input_shape = tuple(input_shape)
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError("Failed to deserialize engine")
        self.ctx = self.engine.create_execution_context()

        # 发现 IO（1 入 / 1 出）
        ins, outs = [], []
        for i in range(self.engine.num_io_tensors):
            n = self.engine.get_tensor_name(i)
            (ins if self.engine.get_tensor_mode(n)==trt.TensorIOMode.INPUT else outs).append(n)
        assert len(ins)==1 and len(outs)==1, f"IO not 1/1: {ins}, {outs}"
        self.in_name, self.out_name = ins[0], outs[0]

        # 设置输入形状，查询输出形状与类型
        self.ctx.set_input_shape(self.in_name, self.input_shape)
        self.out_shape = tuple(self.ctx.get_tensor_shape(self.out_name))
        self.out_dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(self.out_name)))

        # 预分配显存并绑定（复用）
        self.stream = cuda.Stream()
        self.d_in  = cuda.mem_alloc(int(np.prod(self.input_shape)) * 4)  # FP32 输入（端到端）
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

def main():
    runner = TrtRunner()
    x = np.random.randn(*SHAPE).astype(np.float32)

    # 预热
    for _ in range(WARMUP):
        runner.infer_once(x)

    # 计时
    t0 = time.perf_counter()
    for _ in range(ITERS):
        runner.infer_once(x)
    cuda.Context.synchronize()
    t1 = time.perf_counter()

    avg_s = (t1 - t0) / ITERS
    thr = BATCH / avg_s
    print(f"[TensorRT FP16] batch={BATCH} 平均延迟: {avg_s*1000:.3f} ms  吞吐: {thr:,.0f} samples/s")

if __name__ == "__main__":
    main()
