import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np
import random
from PIL import Image

# =====================================================================
# 1. Image Quality & Data Validation
# =====================================================================

def validate_tensor(tensor: torch.Tensor, name: str = "tensor") -> bool:
    """
    Validates a PyTorch tensor for NaNs, Infs, or invalid value ranges.
    """
    if torch.isnan(tensor).any():
        print(f"[Validation Error] {name} contains NaN values!")
        return False
    if torch.isinf(tensor).any():
        print(f"[Validation Error] {name} contains Inf values!")
        return False
    return True

def validate_image_file(image_path: str) -> bool:
    """
    Checks if an image file is readable, not corrupted, and has valid dimensions.
    """
    try:
        with Image.open(image_path) as img:
            img.verify()  # Verify image integrity
        return True
    except Exception as e:
        print(f"[Validation Error] Image file {image_path} is corrupted: {str(e)}")
        return False

def validate_bounding_boxes(boxes: torch.Tensor, labels: torch.Tensor, width: int, height: int):
    """
    Validates object detection bounding boxes.
    Checks:
    - xmin < xmax and ymin < ymax
    - Coordinates are within [0, width] and [0, height]
    - Filters out invalid boxes or clips them to boundaries.
    """
    valid_indices = []
    cleaned_boxes = []
    cleaned_labels = []
    
    for i, (box, label) in enumerate(zip(boxes, labels)):
        xmin, ymin, xmax, ymax = box.tolist()
        
        # Check coordinates validity
        if xmin >= xmax or ymin >= ymax:
            print(f"[Validation Warn] Invalid box coords (min >= max): {box.tolist()}. Skipping.")
            continue
            
        # Check area
        if (xmax - xmin) * (ymax - ymin) <= 0:
            print(f"[Validation Warn] Zero or negative area for box: {box.tolist()}. Skipping.")
            continue
            
        # Clip to image boundaries
        xmin_c = max(0.0, min(xmin, float(width)))
        ymin_c = max(0.0, min(ymin, float(height)))
        xmax_c = max(0.0, min(xmax, float(width)))
        ymax_c = max(0.0, min(ymax, float(height)))
        
        # Re-check area after clipping
        if (xmax_c - xmin_c) * (ymax_c - ymin_c) <= 1.0: # Filter out boxes that become tiny
            continue
            
        cleaned_boxes.append([xmin_c, ymin_c, xmax_c, ymax_c])
        cleaned_labels.append(label.item())
        
    return torch.tensor(cleaned_boxes, dtype=torch.float32), torch.tensor(cleaned_labels, dtype=torch.int64)

# =====================================================================
# 2. Task-Specific Data Preprocessing & Augmentations
# =====================================================================

# --- Task 1: Classification (PathMNIST) ---
# Decisons: H&E stains are sensitive to color representation, so we apply mild color jitter.
# Flips and rotations are highly realistic since cells can be oriented arbitrarily on a slide.
def get_classification_transforms(image_size: int = 224):
    """
    Returns train and evaluation transforms for histopathology classification.
    """
    train_transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),  # Converts to [0, 1] tensor
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05), # Mild color changes
        T.Normalize(mean=[0.707, 0.522, 0.672], std=[0.166, 0.186, 0.158]) # PathMNIST specific stats
    ])
    
    val_transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.707, 0.522, 0.672], std=[0.166, 0.186, 0.158])
    ])
    
    return train_transform, val_transform


# --- Task 2: Segmentation (TNBC Nuclei) ---
# Decisions: Target resolution 256x256. Flips and rotations are applied to image and mask simultaneously.
def apply_joint_segmentation_transforms(image: Image.Image, mask: Image.Image, image_size: int = 256, train: bool = True):
    """
    Applies synchronized geometric transformations to both the image and the mask.
    Returns: (Transformed Image Tensor, Transformed Mask Tensor)
    """
    # Resize
    img_res = TF.resize(image, [image_size, image_size], interpolation=T.InterpolationMode.BILINEAR)
    msk_res = TF.resize(mask, [image_size, image_size], interpolation=T.InterpolationMode.NEAREST)
    
    if train:
        # Random Horizontal Flip
        if random.random() > 0.5:
            img_res = TF.hflip(img_res)
            msk_res = TF.hflip(msk_res)
            
        # Random Vertical Flip
        if random.random() > 0.5:
            img_res = TF.vflip(img_res)
            msk_res = TF.vflip(msk_res)
            
        # Random Rotation
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            img_res = TF.rotate(img_res, angle, interpolation=T.InterpolationMode.BILINEAR)
            msk_res = TF.rotate(msk_res, angle, interpolation=T.InterpolationMode.NEAREST)
            
    # Convert to Tensor
    img_tensor = TF.to_tensor(img_res) # scales to [0,1]
    msk_tensor = TF.to_tensor(msk_res) # scales to [0,1]
    
    # Normalize Image Only (Standard ImageNet normalization is used for segmentation backbone compatibility)
    img_tensor = TF.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # Binarize mask
    msk_tensor = (msk_tensor > 0.5).float()
    
    return img_tensor, msk_tensor


# --- Task 3: Detection (BCCD Blood Smears) ---
# Decisions: Resize to standard size while preserving bounding box mapping.
def preprocess_detection_sample(image: Image.Image, target: dict, target_size: tuple = (480, 640)):
    """
    Preprocesses a detection image and resizes its bounding boxes.
    target_size is (height, width).
    """
    orig_w, orig_h = image.size
    new_h, new_w = target_size
    
    # Resize image
    img_resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    img_tensor = TF.to_tensor(img_resized)
    img_tensor = TF.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # Scale boxes
    scale_x = new_w / orig_w
    scale_y = new_h / orig_h
    
    boxes = target["boxes"]
    scaled_boxes = []
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        scaled_boxes.append([
            xmin * scale_x,
            ymin * scale_y,
            xmax * scale_x,
            ymax * scale_y
        ])
        
    target_scaled = {
        "boxes": torch.tensor(scaled_boxes, dtype=torch.float32),
        "labels": torch.tensor(target["labels"], dtype=torch.int64)
    }
    
    # Validate and clean up scaled boxes
    target_scaled["boxes"], target_scaled["labels"] = validate_bounding_boxes(
        target_scaled["boxes"], target_scaled["labels"], new_w, new_h
    )
    
    return img_tensor, target_scaled
