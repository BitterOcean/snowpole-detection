"""Quick check: verify GPU, CUDA, and required packages are available."""
import sys

print("── Python:", sys.version)

try:
    import torch
    print(f"── PyTorch: {torch.__version__}")
    print(f"── CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"── GPU: {torch.cuda.get_device_name(0)}")
        print(f"── VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
except ImportError:
    print("❌ PyTorch not installed")

for pkg in ["ultralytics", "transformers", "ensemble_boxes", "PIL", "tqdm"]:
    try:
        __import__(pkg if pkg != "PIL" else "PIL.Image")
        print(f"✅ {pkg}")
    except ImportError:
        print(f"❌ {pkg} — run: pip install {pkg}")
