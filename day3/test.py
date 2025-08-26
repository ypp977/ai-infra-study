import importlib

for pkg in ["onnx", "onnxruntime", "torchvision"]:
    try:
        m = importlib.import_module(pkg)
        print(f"✅ {pkg} 存在, 版本: {getattr(m, '__version__', '未知')}")
    except ImportError:
        print(f"❌ {pkg} 未安装")

