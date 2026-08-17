import os
import numpy as np
import urllib.request
import zipfile
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from src.preprocessing import apply_joint_segmentation_transforms

def download_and_extract_tnbc(url: str, data_dir: str):
    """
    Downloads and extracts the TNBC Nuclei Segmentation dataset from Zenodo.
    """
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "TNBC_NucleiSegmentation.zip")
    extract_path = os.path.join(data_dir, "TNBC_NucleiSegmentation")
    
    # Check if already extracted
    if os.path.exists(extract_path):
        print(f"[Dataset] TNBC dataset already exists at: {extract_path}")
        return extract_path
        
    # Download zip if not present
    if not os.path.exists(zip_path):
        print(f"[Dataset] Downloading TNBC dataset from Zenodo: {url}")
        # Custom User-Agent to avoid blocks
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req) as response, open(zip_path, 'wb') as out_file:
            out_file.write(response.read())
        print(f"[Dataset] Download completed: {zip_path}")
        
    # Extract zip
    print(f"[Dataset] Extracting TNBC dataset to: {extract_path}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    print(f"[Dataset] Extraction completed.")
    
    return extract_path


class TNBCDataset(Dataset):
    """
    PyTorch Dataset for loading the Triple Negative Breast Cancer (TNBC) nuclei images and masks.
    """
    def __init__(self, dataset_dir: str, image_size: int = 256, train: bool = True):
        self.dataset_dir = dataset_dir
        self.image_size = image_size
        self.train = train
        
        self.image_mask_pairs = []
        self._find_image_mask_pairs()
        
    def _find_image_mask_pairs(self):
        """
        Crawls the TNBC directory to map original images to their corresponding masks.
        """
        # Look for Slide_XX and GT_XX folders
        dirs = os.listdir(self.dataset_dir)
        slide_dirs = sorted([d for d in dirs if d.startswith("Slide_") and os.path.isdir(os.path.join(self.dataset_dir, d))])
        
        for slide_dir in slide_dirs:
            slide_idx = slide_dir.split("_")[1]
            gt_dir = f"GT_{slide_idx}"
            
            slide_path = os.path.join(self.dataset_dir, slide_dir)
            gt_path = os.path.join(self.dataset_dir, gt_dir)
            
            if not os.path.exists(gt_path):
                continue
                
            # List images in Slide_XX
            for filename in os.listdir(slide_path):
                if filename.lower().endswith(('.png', '.tif', '.jpg', '.jpeg')):
                    img_filepath = os.path.join(slide_path, filename)
                    # The mask is expected to have the same filename in the corresponding GT_XX directory
                    mask_filepath = os.path.join(gt_path, filename)
                    
                    if os.path.exists(mask_filepath):
                        self.image_mask_pairs.append((img_filepath, mask_filepath))
                        
        print(f"[Dataset] Found {len(self.image_mask_pairs)} image-mask pairs in TNBC dataset.")

    def __len__(self):
        return len(self.image_mask_pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.image_mask_pairs[idx]
        
        # Load as PIL images
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Grayscale mask
        
        # Apply joint transforms
        image_tensor, mask_tensor = apply_joint_segmentation_transforms(
            image, mask, image_size=self.image_size, train=self.train
        )
        
        return image_tensor, mask_tensor


def get_tnbc_dataloaders(dataset_dir: str, batch_size: int, image_size: int = 256, demo_mode: bool = False, subset_size: int = 8):
    """
    Splits the TNBC dataset into train, val, and test splits and creates PyTorch DataLoaders.
    """
    # Create dataset objects
    train_dataset = TNBCDataset(dataset_dir, image_size=image_size, train=True)
    eval_dataset = TNBCDataset(dataset_dir, image_size=image_size, train=False)
    
    total_samples = len(train_dataset)
    if total_samples == 0:
        raise ValueError(f"TNBC dataset at {dataset_dir} contains 0 samples. Verify extraction directory.")
        
    # Shuffle indices
    g = torch.Generator().manual_seed(42)
    indices = torch.randperm(total_samples, generator=g).tolist()
    
    # Splits: 70% Train, 15% Val, 15% Test
    val_split = int(np.floor(0.15 * total_samples))
    test_split = int(np.floor(0.15 * total_samples))
    train_split = total_samples - val_split - test_split
    
    train_idx = indices[:train_split]
    val_idx = indices[train_split:train_split+val_split]
    test_idx = indices[train_split+val_split:]
    
    if demo_mode:
        # Use very small subsets in demo mode
        train_idx = train_idx[:subset_size]
        val_idx = val_idx[:max(2, subset_size // 4)]
        test_idx = test_idx[:max(2, subset_size // 4)]
        
        print(f"[Dataset] DEMO MODE: Subsetting TNBC dataset. Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
        
    # Create Subset datasets
    train_sub = Subset(train_dataset, train_idx)
    val_sub = Subset(eval_dataset, val_idx)
    test_sub = Subset(eval_dataset, test_idx)
    
    # DataLoaders
    train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_sub, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader
