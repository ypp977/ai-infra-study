# export_onnx.py
import torch, torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = x.view(-1, 28*28)
        return self.fc2(self.relu(self.fc1(x)))

model = MLP()
model.load_state_dict(torch.load("mlp.pth"))
model.eval()

dummy_input = torch.randn(1, 1, 28, 28)
torch.onnx.export(
    model, dummy_input, "mlp.onnx",
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=12   # 强制导出 ONNX opset 12
)
print("ONNX 模型已导出: mlp.onnx")
