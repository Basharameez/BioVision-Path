import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image

class GradCAM:
    """
    Custom hook-based Grad-CAM implementation for PyTorch CNNs (e.g., ResNet-18).
    Attributes predictions to image regions by weighting convolutional activations with backpropagated gradients.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        
        # Register hooks
        self.forward_hook_handle = target_layer.register_forward_hook(self._forward_hook)
        # register_full_backward_hook is the modern PyTorch standard replacing register_backward_hook
        self.backward_hook_handle = target_layer.register_full_backward_hook(self._backward_hook)
        
    def _forward_hook(self, module, input, output):
        self.activations = output.detach()
        
    def _backward_hook(self, module, grad_input, grad_output):
        # grad_output is a tuple containing gradients with respect to output feature map
        self.gradients = grad_output[0].detach()
        
    def generate_heatmap(self, input_tensor: torch.Tensor, target_class: int = None):
        """
        Generates the 2D Grad-CAM heatmap for a given input tensor and target class.
        input_tensor: shape (1, C, H, W)
        """
        self.model.eval()
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # If target class is not provided, use the class with the highest probability
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()
            
        score = output[0, target_class]
        
        # Backward pass to calculate gradients
        score.backward()
        
        # Extract activations and gradients
        # Shape of activations & gradients: (1, channels, h_feat, w_feat)
        acts = self.activations
        grads = self.gradients
        
        if acts is None or grads is None:
            raise RuntimeError("Hook activations or gradients are empty. Verify forward/backward pass execution.")
            
        # Global Average Pooling of gradients to compute channel weights
        weights = torch.mean(grads, dim=(2, 3), keepdim=True)  # Shape: (1, channels, 1, 1)
        
        # Weighted sum of feature map activations
        cam = torch.sum(weights * acts, dim=1, keepdim=True)  # Shape: (1, 1, h_feat, w_feat)
        
        # Apply ReLU to retain only positive influences on target class
        cam = torch.clamp(cam, min=0)
        
        # Normalize heatmap to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)
            
        # Remove batch and channel dims, convert to numpy
        heatmap = cam.squeeze().cpu().numpy()
        return heatmap, target_class

    def close(self):
        """Removes the hooks to prevent memory leaks."""
        self.forward_hook_handle.remove()
        self.backward_hook_handle.remove()


def overlay_gradcam_on_image(original_img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5):
    """
    Overlays a Grad-CAM heatmap onto a raw image using matplotlib colormap.
    original_img: numpy array of shape (H, W, 3), values in range [0, 1] or [0, 255]
    heatmap: 2D numpy array of shape (H, W), values in range [0, 1]
    """
    # Ensure original_img is in range [0, 255] and uint8
    if original_img.dtype != np.uint8:
        if original_img.max() <= 1.0:
            original_img = (original_img * 255).astype(np.uint8)
        else:
            original_img = original_img.astype(np.uint8)
            
    # Resize heatmap to match image dimensions
    pil_heat = Image.fromarray((heatmap * 255).astype(np.uint8))
    pil_heat = pil_heat.resize((original_img.shape[1], original_img.shape[0]), Image.Resampling.BILINEAR)
    heatmap_resized = np.array(pil_heat) / 255.0
    
    # Get colormap jet
    cmap = plt.get_cmap('jet')
    color_heatmap = cmap(heatmap_resized)[:, :, :3]  # Drop alpha channel
    color_heatmap = (color_heatmap * 255).astype(np.uint8)
    
    # Blend images
    blended = (alpha * color_heatmap + (1 - alpha) * original_img).astype(np.uint8)
    return blended, color_heatmap
