import sys
import torch
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("Python version:", sys.version)
print("PyTorch version:", torch.__version__)
print("PyTorch version type:", type(torch.__version__))
print("CUDA available:", torch.cuda.is_available())

# Patch the environment to help with version detection
os.environ["TORCH_VERSION"] = str(torch.__version__)

try:
    import transformers
    print("Transformers version:", transformers.__version__)
except Exception as e:
    print("Error importing transformers:", str(e))
    import traceback
    traceback.print_exc()

try:
    from packaging import version
    print("\nPackaging version test:")
    torch_version = torch.__version__
    print("- torch_version:", torch_version)
    print("- torch_version type:", type(torch_version))
    parsed_version = version.parse(torch_version)
    print("- Parsed PyTorch version:", parsed_version)
except Exception as e:
    print("Error parsing version:", str(e))
    import traceback
    traceback.print_exc() 