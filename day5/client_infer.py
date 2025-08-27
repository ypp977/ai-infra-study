import numpy as np
import tritonclient.http as http
import time

# 连接 Triton HTTP 服务
client = http.InferenceServerClient(url="localhost:8000")

# 构造输入：batch=8192，每个样本 1x28x28
batch_size = 8192
x = np.random.randn(batch_size, 1, 28, 28).astype(np.float32)

# 定义输入
inputs = [http.InferInput("input", x.shape, "FP32")]
inputs[0].set_data_from_numpy(x)

# 定义输出
outputs = [http.InferRequestedOutput("output")]

# 推理并计时
start = time.time()
result = client.infer(model_name="mnist_mlp", inputs=inputs, outputs=outputs)
end = time.time()

# 结果
output_data = result.as_numpy("output")
print(f"推理结果 shape: {output_data.shape}")   # 应该是 (8192, 10)
print(f"耗时: {end - start:.4f} 秒")
print("示例输出前2个样本:", output_data[:2])

