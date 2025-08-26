# onnx_cpu_b8192.py
import os, time, numpy as np, onnxruntime as ort

# 为了避免抖动，固定线程数（按需调整更快）
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

BATCH = 8192
SHAPE = (BATCH, 1, 28, 28)
WARMUP, ITERS = 5, 20   # 大批量很重，次数适当减小
np.random.seed(123)

def main():
    sess = ort.InferenceSession("mlp.onnx", providers=["CPUExecutionProvider"])
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
    print(f"[ORT CPU] batch={BATCH} 平均延迟: {avg_s*1000:.3f} ms  吞吐: {thr:,.0f} samples/s")

if __name__ == "__main__":
    main()
