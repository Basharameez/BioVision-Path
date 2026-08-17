import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

# =====================================================================
# 1. Classification Evaluation
# =====================================================================

@torch.no_grad()
def evaluate_classifier(model, loader, device):
    """
    Evaluates classification models on the test set.
    Returns:
        metrics: dict of aggregate metrics
        per_class: dict of per-class precision, recall, and f1
        y_true: array of ground truth labels
        y_pred: array of predicted labels
        y_prob: array of predicted class probabilities
    """
    model.eval()
    all_preds = []
    all_trues = []
    all_probs = []
    
    for x, y in loader:
        if len(y.shape) > 1 and y.shape[1] == 1:
            y = y.squeeze(1)
            
        x = x.to(device)
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_trues.extend(y.numpy())
        all_probs.extend(probs.cpu().numpy())
        
    y_true = np.array(all_trues)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    
    # Calculate global metrics
    acc = accuracy_score(y_true, y_pred)
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    weighted_prec, weighted_rec, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    
    # Calculate per-class metrics
    class_prec, class_rec, class_f1, class_support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    
    metrics = {
        "accuracy": acc,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_prec,
        "weighted_recall": weighted_rec,
        "weighted_f1": weighted_f1
    }
    
    per_class = {
        "precision": class_prec,
        "recall": class_rec,
        "f1": class_f1,
        "support": class_support
    }
    
    return metrics, per_class, y_true, y_pred, y_prob


# =====================================================================
# 2. Segmentation Evaluation
# =====================================================================

def calculate_dice_coefficient(pred_mask: torch.Tensor, gt_mask: torch.Tensor, smooth: float = 1e-6) -> float:
    """
    Calculates the Dice Coefficient for a binary segmentation mask.
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    """
    pred_flat = pred_mask.view(-1)
    gt_flat = gt_mask.view(-1)
    intersection = (pred_flat * gt_flat).sum()
    dice = (2. * intersection + smooth) / (pred_flat.sum() + gt_flat.sum() + smooth)
    return dice.item()

def calculate_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor, smooth: float = 1e-6) -> float:
    """
    Calculates the Intersection over Union (IoU) / Jaccard index.
    IoU = |A ∩ B| / |A ∪ B|
    """
    pred_flat = pred_mask.view(-1)
    gt_flat = gt_mask.view(-1)
    intersection = (pred_flat * gt_flat).sum()
    union = pred_flat.sum() + gt_flat.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()

@torch.no_grad()
def evaluate_segmentation(model, loader, device):
    """
    Evaluates U-Net segmentation on test set.
    Returns mean Dice, mean IoU, and pixel accuracy.
    """
    model.eval()
    dice_scores = []
    iou_scores = []
    pixel_accs = []
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        
        # Calculate per-batch/per-sample metrics
        for p, g in zip(preds, y):
            dice = calculate_dice_coefficient(p, g)
            iou = calculate_iou(p, g)
            
            # Pixel accuracy
            correct = (p == g).float().sum()
            total = g.numel()
            pix_acc = (correct / total).item()
            
            dice_scores.append(dice)
            iou_scores.append(iou)
            pixel_accs.append(pix_acc)
            
    return {
        "mean_dice": np.mean(dice_scores),
        "mean_iou": np.mean(iou_scores),
        "pixel_accuracy": np.mean(pixel_accs)
    }


# =====================================================================
# 3. Detection Evaluation (IoU Box Matching)
# =====================================================================

def calculate_box_iou(box1, box2):
    """
    Calculates IoU between two bounding boxes: [xmin, ymin, xmax, ymax]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    return intersection / union

@torch.no_grad()
def evaluate_detector(model, loader, device, iou_threshold=0.5, score_threshold=0.5):
    """
    Evaluates object detection model on the validation/test set.
    Matches predicted boxes to ground truth boxes using IoU and calculates Precision, Recall, and mIoU.
    """
    model.eval()
    total_gts = 0
    total_preds = 0
    true_positives = 0
    matched_ious = []
    
    for images, targets in loader:
        images_dev = [img.to(device) for img in images]
        predictions = model(images_dev)
        
        # predictions is a list of dicts: {"boxes": Tensor, "labels": Tensor, "scores": Tensor}
        for pred, target in zip(predictions, targets):
            p_boxes = pred["boxes"].cpu()
            p_labels = pred["labels"].cpu()
            p_scores = pred["scores"].cpu()
            
            t_boxes = target["boxes"]
            t_labels = target["labels"]
            
            # Filter predictions by confidence score threshold
            keep = p_scores > score_threshold
            p_boxes = p_boxes[keep]
            p_labels = p_labels[keep]
            
            total_gts += len(t_boxes)
            total_preds += len(p_boxes)
            
            # Keep track of matched ground truth boxes
            gt_matched = np.zeros(len(t_boxes))
            
            for p_box, p_lbl in zip(p_boxes, p_labels):
                best_iou = -1.0
                best_idx = -1
                
                for idx, (t_box, t_lbl) in enumerate(zip(t_boxes, t_labels)):
                    if gt_matched[idx] == 1 or p_lbl != t_lbl:
                        continue
                    iou = calculate_box_iou(p_box, t_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = idx
                        
                if best_iou >= iou_threshold and best_idx != -1:
                    true_positives += 1
                    gt_matched[best_idx] = 1
                    matched_ious.append(best_iou)
                    
    precision = true_positives / total_preds if total_preds > 0 else 0.0
    recall = true_positives / total_gts if total_gts > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou = np.mean(matched_ious) if len(matched_ious) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": mean_iou,
        "total_ground_truths": total_gts,
        "total_predictions": total_preds
    }
