import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

# Note: medmnist is imported inside functions to prevent import errors before pip completes.

def load_pathmnist_dataset(split: str, size: int = 28, download: bool = True, transform=None, mmap_mode: str = 'r'):
    """
    Loads a specific split of the MedMNIST PathMNIST dataset at the given image resolution.
    """
    import medmnist
    from medmnist import PathMNIST
    
    # Force size=28 to load the lightweight version (~30MB) instead of MedMNIST+ 224x224 (~12.6GB).
    # Resizing to 224x224 is done in the preprocessing transforms.
    dataset = PathMNIST(
        split=split,
        download=download,
        size=28,
        transform=transform,
        mmap_mode=mmap_mode
    )
    return dataset

def get_pathmnist_metadata():
    """
    Retrieves metadata for PathMNIST from the medmnist info dictionary.
    """
    import medmnist
    info = medmnist.INFO['pathmnist']
    return {
        "name": info["label"],
        "description": info["description"],
        "classes": info["label"],
        "n_classes": len(info["label"]),
        "task": info["task"],
        "n_channels": info["n_channels"],
    }

def get_class_distribution(dataset):
    """
    Computes class counts and frequencies in the given dataset.
    """
    labels = np.array([int(y[0]) for y in dataset.labels])
    unique, counts = np.unique(labels, return_counts=True)
    dist = dict(zip(unique, counts))
    return dist

def get_pathmnist_dataloaders(batch_size: int, size: int = 28, train_transform=None, val_transform=None, test_transform=None, demo_mode: bool = False, subset_size: int = 500):
    """
    Creates PyTorch DataLoaders for train, val, and test splits.
    If demo_mode is True, subsets the datasets to subset_size to ensure fast execution.
    """
    train_dataset = load_pathmnist_dataset(split="train", size=size, transform=train_transform)
    val_dataset = load_pathmnist_dataset(split="val", size=size, transform=val_transform)
    test_dataset = load_pathmnist_dataset(split="test", size=size, transform=test_transform)
    
    if demo_mode:
        # Create small deterministic subsets for fast verification
        import torch
        g = torch.Generator().manual_seed(42)
        
        train_indices = torch.randperm(len(train_dataset), generator=g)[:subset_size].tolist()
        val_indices = torch.randperm(len(val_dataset), generator=g)[:min(subset_size // 2, len(val_dataset))].tolist()
        test_indices = torch.randperm(len(test_dataset), generator=g)[:min(subset_size // 2, len(test_dataset))].tolist()
        
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)
        test_dataset = Subset(test_dataset, test_indices)
        
        print(f"[Data] DEMO MODE: Subsetting datasets. Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    else:
        print(f"[Data] FULL MODE: Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader
