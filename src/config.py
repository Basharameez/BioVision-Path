import os
import torch

# Global Configuration Dictionary
CONFIG = {
    # General Setup
    "seed": 42,
    "demo_mode": True,  # Set to True for fast testing/verification, False for full experiments
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    
    # Directory Paths (Relative to project root)
    "data_dir": "data",
    "output_dir": "outputs",
    "checkpoint_dir": "checkpoints",
    
    # Task 1 - Classification (PathMNIST)
    "classification": {
        "dataset_name": "pathmnist",
        "image_size": 224,  # Size for ResNet-18 (MedMNIST+)
        "full": {
            "batch_size": 64,
            "epochs": 5,
            "lr": 1e-3,
            "weight_decay": 1e-4,
        },
        "demo": {
            "batch_size": 16,
            "epochs": 1,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "subset_size": 500,  # Number of samples to use in demo mode
        }
    },
    
    # Task 2 - Segmentation (TNBC Nuclei)
    "segmentation": {
        "dataset_url": "https://zenodo.org/record/1175282/files/TNBC_NucleiSegmentation.zip",
        "image_size": 256,
        "full": {
            "batch_size": 8,
            "epochs": 10,
            "lr": 1e-3,
        },
        "demo": {
            "batch_size": 4,
            "epochs": 1,
            "lr": 1e-3,
            "subset_size": 8,  # Number of slide folders to use
        }
    },
    
    # Task 3 - Detection (BCCD Smears)
    "detection": {
        "dataset_repo": "https://github.com/Shenggan/BCCD_Dataset.git",
        "image_size": (480, 640),  # Height, Width (Standard BCCD resolution)
        "classes": ["background", "WBC", "RBC", "Platelets"],  # 0 is always background
        "full": {
            "batch_size": 4,
            "epochs": 10,
            "lr": 5e-4,
            "weight_decay": 1e-5,
        },
        "demo": {
            "batch_size": 2,
            "epochs": 1,
            "lr": 5e-4,
            "weight_decay": 1e-5,
            "subset_size": 20,  # Number of images to use
        }
    }
}

# Ensure required directories exist
os.makedirs(CONFIG["data_dir"], exist_ok=True)
os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
for sub in ["classification", "segmentation", "detection", "features", "explainability", "reports"]:
    os.makedirs(os.path.join(CONFIG["output_dir"], sub), exist_ok=True)
