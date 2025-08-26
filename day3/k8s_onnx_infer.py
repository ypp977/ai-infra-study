import onnxruntime as ort
import numpy as np

# 模型路径（通过 ConfigMap 挂载到 /workspace/models）
MODEL_PATH = "/workspace/models/mlp.onnx"

print(f"🚀 正在加载模型: {MODEL_PATH}")
sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

# 随机输入测试
x = np.random.randn(1, 1, 28, 28).astype(np.float32)
inputs = {sess.get_inputs()[0].name: x}
outputs = sess.run(None, inputs)

print("✅ 推理结果:", outputs[0])

