# onnx_cuda_b8192.py
import time, numpy as np, onnxruntime as ort

BATCH = 8192
SHAPE = (BATCH, 1, 28, 28)
WARMUP, ITERS = 5, 20
np.random.seed(123)

def main():
    # 需要 onnxruntime-gpu + 正确的 CUDA/cuDNN
    sess = ort.InferenceSession("mlp.onnx", providers=["CUDAExecutionProvider"])
    name = sess.get_inputs()[0].name
    x = np.random.randn(*SHAPE).astype(np.float32)

    # 预热
    for _ in range(WARMUP):
        _ = sess.run(None, {name: x})

    # 计时
    t0 = time.perf_counter()
    for _ in range(ITERS):
        _ = sess.run(None, {name: x})
    t1 = time.perf_counter()

    avg_s = (t1 - t0) / ITERS
    thr = BATCH / avg_s
    print(f"[ORT CUDA] batch={BATCH} 平均延迟: {avg_s*1000:.3f} ms  吞吐: {thr:,.0f} samples/s")

if __name__ == "__main__":
    main()
