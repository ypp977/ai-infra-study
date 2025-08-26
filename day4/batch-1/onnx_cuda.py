# onnx_cuda.py
import time, numpy as np, onnxruntime as ort

WARMUP, ITERS = 100, 1000
np.random.seed(123)

def main():
    sess = ort.InferenceSession("mlp.onnx", providers=["CUDAExecutionProvider"])
    name = sess.get_inputs()[0].name
    x = np.random.randn(1, 1, 28, 28).astype(np.float32)

    # 预热
    for _ in range(WARMUP):
        sess.run(None, {name: x})

    # 计时
    t0 = time.perf_counter()
    for _ in range(ITERS):
        sess.run(None, {name: x})
    t1 = time.perf_counter()

    print(f"ONNX Runtime CUDA 平均延迟: {(t1 - t0)/ITERS*1000:.3f} ms")

if __name__ == "__main__":
    main()
