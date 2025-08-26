# onnx_infer.py
import onnxruntime as ort
import numpy as np

# 加载 ONNX
sess = ort.InferenceSession("mlp.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

# 随机输入
x = np.random.randn(1, 1, 28, 28).astype(np.float32)
inputs = {sess.get_inputs()[0].name: x}
outputs = sess.run(None, inputs)

print("推理结果:", outputs[0])
