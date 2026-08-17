import os
import time
import torch
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

# =====================================================================
# 1. Epoch Level Training & Validation Loops
# =====================================================================

def train_epoch_standard(model, loader, optimizer, criterion, device, scaler=None):
    """
    Trains standard models (Classification/Segmentation) for one epoch.
    Supports Automatic Mixed Precision (AMP) when scaler is provided.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for x, y in tqdm(loader, desc="Training Batches", leave=False):
        # MedMNIST targets are often squeezed to shape (batch_size, 1), we need shape (batch_size,)
        if len(y.shape) == 2 and y.shape[1] == 1:
            y = y.squeeze(1)
            
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        
        if scaler is not None:
            with autocast():
                outputs = model(x)
                loss = criterion(outputs, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
        running_loss += loss.item() * x.size(0)
        
        # Classification metric collection
        if len(outputs.shape) > 1 and outputs.shape[1] > 1:
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == y.data).item()
            total += y.size(0)
            
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = (correct / total) if total > 0 else 0.0
    return epoch_loss, epoch_acc

@torch.no_grad()
def val_epoch_standard(model, loader, criterion, device):
    """
    Validates standard models (Classification/Segmentation) for one epoch.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for x, y in tqdm(loader, desc="Validation Batches", leave=False):
        if len(y.shape) == 2 and y.shape[1] == 1:
            y = y.squeeze(1)
            
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        loss = criterion(outputs, y)
        
        running_loss += loss.item() * x.size(0)
        
        if len(outputs.shape) > 1 and outputs.shape[1] > 1:
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == y.data).item()
            total += y.size(0)
            
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = (correct / total) if total > 0 else 0.0
    return epoch_loss, epoch_acc


def train_epoch_detector(model, loader, optimizer, device, scaler=None):
    """
    Trains Faster R-CNN detection model for one epoch.
    In PyTorch torchvision, Faster R-CNN outputs a loss dict when in training mode.
    """
    model.train()
    running_loss = 0.0
    
    for images, targets in tqdm(loader, desc="Detection Batches", leave=False):
        # Images: list of Tensors
        # Targets: list of dicts: {"boxes": Tensor, "labels": Tensor}
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with autocast():
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
            
        running_loss += losses.item() * len(images)
        
    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss, 0.0


# =====================================================================
# 2. Complete Training Runner with Checkpointing & Early Stopping
# =====================================================================

def fit_model(
    model, 
    train_loader, 
    val_loader, 
    optimizer, 
    criterion, 
    epochs, 
    device, 
    checkpoint_path, 
    scheduler=None, 
    early_stopping_patience=5,
    task_type="classification"
):
    """
    General runner function to train and validate a model.
    """
    best_val_loss = float("inf")
    patience_counter = 0
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "epoch_time": []
    }
    
    # Enable mixed precision scaler if running on CUDA
    scaler = GradScaler() if device == "cuda" else None
    
    print(f"[Training] Starting training on device: {device} (Task: {task_type})")
    
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        
        # 1. Run training epoch
        if task_type == "detection":
            train_loss, train_acc = train_epoch_detector(model, train_loader, optimizer, device, scaler)
            val_loss, val_acc = 0.0, 0.0  # Detection model doesn't compute validation loss this way natively
        else:
            train_loss, train_acc = train_epoch_standard(model, train_loader, optimizer, criterion, device, scaler)
            val_loss, val_acc = val_epoch_standard(model, val_loader, criterion, device)
            
        epoch_time = time.time() - start_time
        
        # Log metrics
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epoch_time"].append(epoch_time)
        
        # Print summary
        if task_type == "detection":
            print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Time: {epoch_time:.1f}s")
        else:
            print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Time: {epoch_time:.1f}s")
            
        # Update Scheduler
        if scheduler is not None:
            # Check if ReduceLROnPlateau or standard scheduler
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss if task_type != "detection" else train_loss)
            else:
                scheduler.step()
                
        # 2. Checkpoint Saving & Early Stopping
        if task_type == "detection":
            # For detection, we save based on train loss improvement
            metric_to_check = train_loss
        else:
            metric_to_check = val_loss
            
        if metric_to_check < best_val_loss:
            best_val_loss = metric_to_check
            patience_counter = 0
            # Save checkpoint
            torch.save(model.state_dict(), checkpoint_path)
            print(f" => State saved! New best loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"[Training] Early stopping triggered. Validation performance has not improved for {early_stopping_patience} epochs.")
                break
                
    # 3. Restore Best Checkpoint
    if os.path.exists(checkpoint_path):
        print(f"[Training] Restoring best checkpoint from: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print("[Training Warning] No checkpoint saved, returning current model state.")
        
    return history
