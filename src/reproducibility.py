import random
import sys
import numpy as np
import torch

def set_seed(seed: int = 42):
    """
    Sets random seeds for Python, NumPy, and PyTorch to enforce reproducibility.
    Also handles CUDA-specific deterministic flags if a GPU is available.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Configure PyTorch to use deterministic algorithms where possible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    print(f"[Reproducibility] Random seed set to: {seed}")

def get_environment_info():
    """
    Detects and returns details about Python, PyTorch, and CUDA execution environment.
    """
    info = {
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None (CPU Fallback)",
    }
    return info

def print_environment_report():
    """
    Prints a clean, concise environment report for the user or Colab interface.
    """
    env = get_environment_info()
    print("=" * 60)
    print("                    ENVIRONMENT REPORT                    ")
    print("=" * 60)
    print(f"Python Version    : {env['python_version']}")
    print(f"PyTorch Version   : {env['pytorch_version']}")
    print(f"CUDA Available    : {env['cuda_available']}")
    print(f"GPU Name          : {env['gpu_name']}")
    print("=" * 60)
    
    if not env["cuda_available"]:
        print("[Notice] CUDA is not available. System is running in CPU fallback mode.")
        print("         In GPU environments (like Colab GPU runtime), this will significantly slow training.")
        print("         Please check runtime settings if a GPU was expected.")
    else:
        print("[Notice] Mixed precision training (AMP) will be automatically enabled on GPU.")
    print("=" * 60)
    return env
