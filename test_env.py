import sys
import torch
import transformers
import bitsandbytes as bnb
import accelerate
import safetensors
import sentencepiece

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"Transformers version: {transformers.__version__}")
print(f"Bitsandbytes version: {bnb.__version__}")
print(f"Accelerate version: {accelerate.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}") 