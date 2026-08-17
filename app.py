# Import spaces dynamically for Hugging Face ZeroGPU compatibility
try:
    import spaces
    has_spaces = True
except ImportError:
    has_spaces = False

def gpu_decorator(func):
    if has_spaces:
        return spaces.GPU(func)
    return func

import os
import torch
import numpy as np
import gradio as gr
from PIL import Image, ImageDraw
import torchvision.transforms.functional as TF
import torchvision.transforms as T

# Import project modules
from src.config import CONFIG
from src.models import get_resnet18_model, UNet, get_faster_rcnn_mobilenet_v3
from src.data import get_pathmnist_metadata
from src.preprocessing import get_classification_transforms
from src.explainability import GradCAM, overlay_gradcam_on_image
from src.visualization import denormalize_image

# Dynamic execution device (GPU if available, fallback to CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
meta = get_pathmnist_metadata()

# Load models and weights
print("[Gradio] Loading checkpoints...")

# 1. Classification model (ResNet-18)
cls_model = get_resnet18_model(num_classes=meta["n_classes"], pretrained=False).to(device)
c_path = "checkpoints/resnet18_classification.pth"
if os.path.exists(c_path):
    cls_model.load_state_dict(torch.load(c_path, map_location=device))
cls_model.eval()

# 2. Segmentation model (U-Net)
seg_model = UNet().to(device)
s_path = "checkpoints/unet_segmentation.pth"
if os.path.exists(s_path):
    seg_model.load_state_dict(torch.load(s_path, map_location=device))
seg_model.eval()

# 3. Detection model (Faster R-CNN)
det_model = get_faster_rcnn_mobilenet_v3(num_classes=4, pretrained=False).to(device)
d_path = "checkpoints/faster_rcnn_detection.pth"
if os.path.exists(d_path):
    det_model.load_state_dict(torch.load(d_path, map_location=device))
det_model.eval()

# Preprocessing transforms
_, val_transform = get_classification_transforms(224)

# =====================================================================
# Inference functions
# =====================================================================

@gpu_decorator
def predict_classification(input_img):
    """
    Classifies a colorectal histopathology patch and generates Grad-CAM.
    """
    if input_img is None:
        return "Please upload an image.", None
        
    # Resize and preprocess PIL Image
    img_resized = input_img.resize((224, 224), Image.Resampling.BILINEAR)
    img_tensor = val_transform(img_resized).unsqueeze(0).to(device)
    
    # Forward pass
    with torch.no_grad():
        outputs = cls_model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        max_idx = torch.argmax(probs).item()
        
    # Format labels
    label_dict = {}
    for i in range(meta["n_classes"]):
        class_name = meta["classes"][str(i)]
        label_dict[class_name] = float(probs[i].cpu().numpy())
        
    # Run Grad-CAM
    gradcam = GradCAM(cls_model, cls_model.layer4)
    heatmap, _ = gradcam.generate_heatmap(img_tensor)
    gradcam.close()
    
    # Overlay Grad-CAM
    original_np = denormalize_image(img_tensor[0], mean=[0.707, 0.522, 0.672], std=[0.166, 0.186, 0.158])
    blended, _ = overlay_gradcam_on_image(original_np, heatmap, alpha=0.5)
    
    # Convert blended back to PIL Image
    blended_img = Image.fromarray((blended * 255).astype(np.uint8))
    
    return label_dict, blended_img


@gpu_decorator
def predict_segmentation(input_img):
    """
    Segments nuclei from H&E breast cancer slide images.
    """
    if input_img is None:
        return None, None
        
    # Preprocess
    orig_w, orig_h = input_img.size
    img_res = input_img.resize((256, 256), Image.Resampling.BILINEAR)
    
    img_tensor = TF.to_tensor(img_res)
    img_tensor = TF.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        preds = torch.sigmoid(seg_model(img_tensor))[0, 0]
        mask_pred = (preds > 0.5).cpu().numpy().astype(np.uint8)
        
    # Upscale mask back to original size
    mask_pil = Image.fromarray(mask_pred * 255).resize((orig_w, orig_h), Image.Resampling.NEAREST)
    
    # Create overlay (green highlight on original image)
    overlay_img = input_img.convert("RGBA")
    mask_rgba = Image.new("RGBA", (orig_w, orig_h), (0, 255, 0, 0))
    
    # Put mask values into alpha channel of green overlay
    mask_np = np.array(mask_pil)
    for y in range(orig_h):
        for x in range(orig_w):
            if mask_np[y, x] > 0:
                mask_rgba.putpixel((x, y), (0, 255, 0, 80)) # Semi-transparent green
                
    final_overlay = Image.alpha_composite(overlay_img, mask_rgba).convert("RGB")
    
    return mask_pil, final_overlay


@gpu_decorator
def predict_detection(input_img, score_threshold=0.5):
    """
    Detects blood smear cells (WBC, RBC, Platelets).
    """
    if input_img is None:
        return None
        
    # Resize to standard size for prediction
    orig_w, orig_h = input_img.size
    target_w, target_h = 640, 480
    img_res = input_img.resize((target_w, target_h), Image.Resampling.BILINEAR)
    
    img_tensor = TF.to_tensor(img_res)
    img_tensor = TF.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        predictions = det_model(img_tensor)[0]
        
    # Filter predictions by confidence
    boxes = predictions["boxes"].cpu().numpy()
    labels = predictions["labels"].cpu().numpy()
    scores = predictions["scores"].cpu().numpy()
    
    keep = scores >= score_threshold
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]
    
    # Scale boxes back to original size
    scale_x = orig_w / target_w
    scale_y = orig_h / target_h
    
    # Draw boxes
    draw_img = input_img.copy()
    draw = ImageDraw.Draw(draw_img)
    
    class_map = CONFIG["detection"]["classes"]
    colors = {1: "red", 2: "blue", 3: "green"} # Red for WBC, Blue for RBC, Green for Platelets
    
    for box, label, score in zip(boxes, labels, scores):
        xmin, ymin, xmax, ymax = box
        xmin_s = xmin * scale_x
        ymin_s = ymin * scale_y
        xmax_s = xmax * scale_x
        ymax_s = ymax * scale_y
        
        class_name = class_map[label]
        color = colors.get(label, "yellow")
        
        # Draw box outline
        draw.rectangle([xmin_s, ymin_s, xmax_s, ymax_s], outline=color, width=3)
        # Draw label text
        draw.text((xmin_s + 5, ymin_s + 5), f"{class_name}: {score:.2f}", fill=color)
        
    return draw_img

# =====================================================================
# Gradio Dashboard Design
# =====================================================================

description_text = """
# BioVision-Path: Multi-Task Biomedical Image Analysis
This interactive dashboard demonstrates the power of modular deep learning pipelines on three major biomedical vision tasks:
1. **Colorectal Pathology Classification**: Uses fine-tuned ResNet-18. Grad-CAM visualizes attributions (the epithelial tissues driving the classifier).
2. **Nuclei Segmentation**: Uses a custom U-Net trained from scratch to isolate breast cancer nuclei.
3. **Microscopy Cell Detection**: Uses a Faster R-CNN MobileNet-V3 FPN detector to locate and classify White Blood Cells, Red Blood Cells, and Platelets.

*Strictly for portfolio and research demonstration.*
"""

# Tab 1: Classification
tab1 = gr.Interface(
    fn=predict_classification,
    inputs=gr.Image(type="pil", label="Upload Histopathology Patch (H&E)"),
    outputs=[
        gr.Label(num_top_classes=3, label="Predicted Tissue Category"),
        gr.Image(type="pil", label="Grad-CAM Interpretability Overlay")
    ],
    title="1. Histopathology Classification & Grad-CAM",
    description="Upload a H&E tissue patch from the PathMNIST dataset. The system predicts the tissue type and visualizes the attributions using Grad-CAM."
)

# Tab 2: Segmentation
tab2 = gr.Interface(
    fn=predict_segmentation,
    inputs=gr.Image(type="pil", label="Upload Nuclei Tissue Slide"),
    outputs=[
        gr.Image(type="pil", label="Predicted Binary Nuclei Mask"),
        gr.Image(type="pil", label="Segmentation Contour Overlay")
    ],
    title="2. Biomedical Nuclei Segmentation (U-Net)",
    description="Upload a breast cancer tissue slide patch. The U-Net will segment the boundaries of individual tumor nuclei."
)

# Tab 3: Detection
tab3 = gr.Interface(
    fn=predict_detection,
    inputs=[
        gr.Image(type="pil", label="Upload Blood Smear Microscopy Image"),
        gr.Slider(minimum=0.1, maximum=0.9, value=0.5, step=0.05, label="Detector Confidence Threshold")
    ],
    outputs=gr.Image(type="pil", label="Detected Cells Overlay"),
    title="3. Blood Smear Cell Detection (Faster R-CNN)",
    description="Upload a microscopy slide smear. Red circles indicate WBCs, blue circles indicate RBCs, and green circles indicate Platelets."
)

# Combine into Tabbed Interface
demo = gr.TabbedInterface(
    [tab1, tab2, tab3],
    tab_names=["Histopathology Classification", "Breast Cancer Segmentation", "Smear Cell Detection"]
)

if __name__ == "__main__":
    # Launch Gradio dashboard.
    print("[Gradio] Launching visual web dashboard...")
    demo.launch(server_name="0.0.0.0", server_port=7860)
    
    # Keep main thread alive persistently
    import time
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[Gradio] Shutting down...")
