import os
import xml.etree.ElementTree as ET
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from src.preprocessing import preprocess_detection_sample

class BCCDDataset(Dataset):
    """
    PyTorch Dataset for parsing and loading the BCCD Blood Cell smear images and XML annotations.
    """
    def __init__(self, root_dir: str, target_size: tuple = (480, 640), label_map: dict = None):
        """
        root_dir: path to the directory containing BCCD folder (e.g. bccd_temp/BCCD)
        """
        self.root_dir = root_dir
        self.target_size = target_size  # (height, width)
        
        # Default label mapping (0 is background for Faster R-CNN)
        self.label_map = label_map if label_map is not None else {
            "WBC": 1,
            "RBC": 2,
            "Platelets": 3
        }
        
        self.annotations_dir = os.path.join(root_dir, "Annotations")
        self.images_dir = os.path.join(root_dir, "JPEGImages")
        
        self.xml_files = sorted([f for f in os.listdir(self.annotations_dir) if f.endswith(".xml")])
        
    def __len__(self):
        return len(self.xml_files)
        
    def _parse_xml(self, xml_path: str):
        """
        Parses Pascal VOC XML annotation file.
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        boxes = []
        labels = []
        
        for obj in root.findall("object"):
            label_name = obj.find("name").text
            if label_name not in self.label_map:
                continue
                
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
            
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.label_map[label_name])
            
        return boxes, labels

    def __getitem__(self, idx):
        xml_name = self.xml_files[idx]
        xml_path = os.path.join(self.annotations_dir, xml_name)
        
        # Determine image path (replaces .xml with .jpg)
        img_name = xml_name.replace(".xml", ".jpg")
        img_path = os.path.join(self.images_dir, img_name)
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        
        # Parse XML bounding boxes
        boxes, labels = self._parse_xml(xml_path)
        
        target = {
            "boxes": boxes,
            "labels": labels
        }
        
        # Preprocess and resize image & boxes
        img_tensor, target_preprocessed = preprocess_detection_sample(
            image, target, target_size=self.target_size
        )
        
        return img_tensor, target_preprocessed


def detection_collate_fn(batch):
    """
    Custom collate function for object detection batching.
    Faster R-CNN expects:
        images: list of tensors
        targets: list of dictionaries
    """
    return tuple(zip(*batch))


def get_bccd_dataloaders(root_dir: str, batch_size: int, target_size: tuple = (480, 640), demo_mode: bool = False, subset_size: int = 20):
    """
    Loads, splits, and creates DataLoaders for BCCD dataset.
    """
    dataset = BCCDDataset(root_dir, target_size=target_size)
    total_samples = len(dataset)
    
    if total_samples == 0:
        raise ValueError(f"BCCD dataset at {root_dir} contains 0 samples. Verify path.")
        
    # Split: 80% Train, 20% Val (and Test)
    g = torch.Generator().manual_seed(42)
    indices = torch.randperm(total_samples, generator=g).tolist()
    
    val_split = int(np.floor(0.20 * total_samples))
    train_split = total_samples - val_split
    
    train_idx = indices[:train_split]
    val_idx = indices[train_split:]
    
    if demo_mode:
        train_idx = train_idx[:subset_size]
        val_idx = val_idx[:max(2, subset_size // 4)]
        print(f"[Dataset] DEMO MODE: Subsetting BCCD dataset. Train: {len(train_idx)}, Val: {len(val_idx)}")
        
    train_sub = Subset(dataset, train_idx)
    val_sub = Subset(dataset, val_idx)
    
    # Custom collate_fn is passed to DataLoader
    train_loader = DataLoader(
        train_sub, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0, 
        collate_fn=detection_collate_fn
    )
    val_loader = DataLoader(
        val_sub, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0, 
        collate_fn=detection_collate_fn
    )
    
    return train_loader, val_loader
import numpy as np
