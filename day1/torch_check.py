import torch, onnxruntime as ort, platform
print("Torch:", torch.__version__, "| Python:", platform.python_version())
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"  [{i}] {torch.cuda.get_device_name(i)}")
print("ORT:", ort.__version__)
print("Providers:", ort.get_available_providers())
