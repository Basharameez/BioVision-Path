import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw
from sklearn.metrics import confusion_matrix

def denormalize_image(tensor_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Denormalizes an image tensor back to range [0, 1] for visualization.
    """
    if isinstance(tensor_img, np.ndarray):
        img = tensor_img.copy()
        # Assume shape (3, H, W)
        for c in range(3):
            img[c] = img[c] * std[c] + mean[c]
        img = np.clip(img, 0, 1)
        return np.transpose(img, (1, 2, 0)) # (H, W, 3)
        
    img = tensor_img.clone()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    img = torch.clamp(img, 0, 1)
    return img.permute(1, 2, 0).cpu().numpy()

import torch

def plot_dataset_montage(dataset, class_names, num_samples=9, save_path=None):
    """
    Plots a grid of sample images with their ground-truth labels.
    """
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = axes.flatten()
    
    # Select random indices
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        img_tensor, label = dataset[idx]
        
        # If medmnist dataset, label is an array of shape (1,)
        if isinstance(label, np.ndarray):
            label_idx = int(label[0])
        elif isinstance(label, torch.Tensor):
            label_idx = label.item()
        else:
            label_idx = int(label)
            
        class_name = class_names[str(label_idx)] if isinstance(class_names, dict) else class_names[label_idx]
        
        # Check if normalized (if it is a tensor, denormalize, else convert PIL Image to numpy)
        if isinstance(img_tensor, torch.Tensor):
            img = denormalize_image(img_tensor, mean=[0.707, 0.522, 0.672], std=[0.166, 0.186, 0.158])
        else:
            img = np.array(img_tensor)
            
        axes[i].imshow(img)
        axes[i].set_title(class_name, fontsize=12)
        axes[i].axis("off")
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

def plot_class_distribution(dist_dict, class_names, title="Class Distribution", save_path=None):
    """
    Plots a bar chart showing class frequency.
    """
    plt.figure(figsize=(10, 5))
    
    # Sort keys
    keys = sorted(dist_dict.keys())
    counts = [dist_dict[k] for k in keys]
    labels = [class_names[str(k)] if isinstance(class_names, dict) else class_names[k] for k in keys]
    
    sns.barplot(x=labels, y=counts, palette="viridis")
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Class", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for i, count in enumerate(counts):
        plt.text(i, count + 0.01 * max(counts), str(count), ha='center', va='bottom', fontsize=9)
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

def plot_training_curves(history, save_path=None):
    """
    Plots train vs val loss and accuracy.
    """
    epochs = len(history["train_loss"])
    epochs_range = range(1, epochs + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history["train_loss"], label="Train Loss", marker='o')
    plt.plot(epochs_range, history["val_loss"], label="Val Loss", marker='s')
    plt.title("Training and Validation Loss", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    
    # Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history["train_acc"], label="Train Acc", marker='o')
    plt.plot(epochs_range, history["val_acc"], label="Val Acc", marker='s')
    plt.title("Training and Validation Accuracy", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

def plot_confusion_matrix_sns(y_true, y_pred, class_names, title="Confusion Matrix", save_path=None):
    """
    Plots a confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    
    labels = [class_names[str(i)] if isinstance(class_names, dict) else class_names[i] for i in range(len(class_names))]
    
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.ylabel("Ground Truth", fontsize=12)
    plt.xlabel("Prediction", fontsize=12)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

def plot_segmentation_predictions(image_tensor, gt_mask_tensor, pred_mask_tensor, save_path=None):
    """
    Displays a row comparing: Original Image, Ground Truth Mask, Predicted Mask, Overlay.
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Denormalize image
    img = denormalize_image(image_tensor)
    
    # Convert masks to 2D numpy arrays
    gt = gt_mask_tensor.squeeze().cpu().numpy()
    pred = pred_mask_tensor.squeeze().cpu().numpy()
    
    # Create overlay (green for true positive, red for false positive/negative)
    overlay = img.copy()
    # Apply ground-truth overlay as blue, prediction as green
    overlay_mask = np.zeros_like(img)
    overlay_mask[:, :, 1] = pred * 0.5  # Green for predictions
    overlay_mask[:, :, 2] = gt * 0.5    # Blue for ground truth
    
    # Blended overlay
    blended = np.clip(img * 0.7 + overlay_mask, 0, 1)
    
    axes[0].imshow(img)
    axes[0].set_title("Original Image", fontsize=12, fontweight="bold")
    axes[0].axis("off")
    
    axes[1].imshow(gt, cmap="gray")
    axes[1].set_title("Ground-Truth Mask", fontsize=12, fontweight="bold")
    axes[1].axis("off")
    
    axes[2].imshow(pred, cmap="gray")
    axes[2].set_title("Predicted Mask", fontsize=12, fontweight="bold")
    axes[2].axis("off")
    
    axes[3].imshow(blended)
    axes[3].set_title("Overlay (Pred=G, GT=B)", fontsize=12, fontweight="bold")
    axes[3].axis("off")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

def plot_detection_predictions(image_tensor, target, prediction=None, class_names=None, save_path=None):
    """
    Plots an image with bounding boxes. Shows ground truth boxes (blue) and optionally predicted boxes (red).
    """
    # Denormalize image
    img_np = denormalize_image(image_tensor)
    img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
    draw = ImageDraw.Draw(img_pil)
    
    # Draw Ground Truth (Blue)
    gt_boxes = target["boxes"]
    gt_labels = target["labels"]
    for box, lbl in zip(gt_boxes, gt_labels):
        xmin, ymin, xmax, ymax = box.tolist()
        draw.rectangle([xmin, ymin, xmax, ymax], outline="blue", width=3)
        label_text = class_names[lbl.item()] if class_names else f"GT:{lbl.item()}"
        draw.text((xmin + 2, ymin + 2), label_text, fill="blue")
        
    # Draw Predictions (Red)
    if prediction is not None:
        pred_boxes = prediction["boxes"]
        pred_labels = prediction["labels"]
        pred_scores = prediction["scores"]
        
        for box, lbl, score in zip(pred_boxes, pred_labels, pred_scores):
            if score < 0.5:
                continue
            xmin, ymin, xmax, ymax = box.tolist()
            draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)
            label_text = f"{class_names[lbl.item()] if class_names else lbl.item()}:{score:.2f}"
            draw.text((xmin + 2, ymax - 12), label_text, fill="red")
            
    plt.figure(figsize=(8, 6))
    plt.imshow(img_pil)
    plt.title("Object Detection Smear (Blue=GT, Red=Pred)", fontsize=12, fontweight="bold")
    plt.axis("off")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

def plot_embeddings(reduced_embeddings, labels, class_names, method_name="PCA", save_path=None):
    """
    Scatter plot of class-colored features.
    """
    plt.figure(figsize=(10, 8))
    
    unique_labels = np.unique(labels)
    cmap = plt.get_cmap("tab10")
    
    for idx, lbl in enumerate(unique_labels):
        mask = labels == lbl
        lbl_name = class_names[str(lbl)] if isinstance(class_names, dict) else class_names[lbl]
        plt.scatter(
            reduced_embeddings[mask, 0],
            reduced_embeddings[mask, 1],
            label=lbl_name,
            alpha=0.7,
            color=cmap(idx)
        )
        
    plt.title(f"Embedding Visualization via {method_name}", fontsize=14, fontweight="bold")
    plt.xlabel(f"{method_name} Component 1", fontsize=12)
    plt.ylabel(f"{method_name} Component 2", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle=":", alpha=0.5)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

def plot_nearest_neighbors(query_img, neighbor_imgs, distances, query_class=None, neighbor_classes=None, class_names=None, save_path=None):
    """
    Plots the query image and its top-K nearest neighbors.
    """
    fig, axes = plt.subplots(1, len(neighbor_imgs) + 1, figsize=(18, 4))
    
    # Plot Query
    q_img = denormalize_image(query_img, mean=[0.707, 0.522, 0.672], std=[0.166, 0.186, 0.158])
    axes[0].imshow(q_img)
    q_class_name = class_names[str(query_class)] if isinstance(class_names, dict) else class_names[query_class] if query_class is not None else ""
    axes[0].set_title(f"Query\nClass: {q_class_name}", fontsize=12, fontweight="bold", color="blue")
    axes[0].axis("off")
    
    # Plot Neighbors
    for idx, (nb_img, dist) in enumerate(zip(neighbor_imgs, distances)):
        n_img = denormalize_image(nb_img, mean=[0.707, 0.522, 0.672], std=[0.166, 0.186, 0.158])
        axes[idx + 1].imshow(n_img)
        
        n_lbl = neighbor_classes[idx]
        n_class_name = class_names[str(n_lbl)] if isinstance(class_names, dict) else class_names[n_lbl] if n_lbl is not None else ""
        
        axes[idx + 1].set_title(f"NN {idx + 1} (d={dist:.2f})\nClass: {n_class_name}", fontsize=11)
        axes[idx + 1].axis("off")
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()
