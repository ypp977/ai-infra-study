# NGC 官方 PyTorch 镜像（含 CUDA/cuDNN/NCCL + TRT SDK）
FROM nvcr.io/nvidia/pytorch:25.06-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=5

# 只补必要工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# 仅补 ONNX + ORT（TRT SDK 已在基镜像里）
RUN python -m pip install --extra-index-url https://pypi.org/simple -i https://mirrors.aliyun.com/pypi/simple \
    --only-binary=:all: --prefer-binary \
    onnx==1.16.1 onnxruntime-gpu==1.20.0

# 快速自检，构建时就能早发现环境问题（可注释）
RUN python - <<'PY'
import sys, torch, onnx, onnxruntime as ort
print("Python:", sys.version.split()[0])
print("Torch:", torch.__version__, "| CUDA:", torch.version.cuda, "| cuDNN:", torch.backends.cudnn.version())
print("ONNX:", onnx.__version__)
print("ORT:", ort.__version__, "| Providers:", ort.get_available_providers())
PY

# 拷贝推理脚本
COPY k8s_onnx_infer.py /workspace/k8s_onnx_infer.py

# 默认执行脚本（也可在 K8s 里覆盖）
CMD ["python", "/workspace/k8s_onnx_infer.py"]
