import torch
import torch.nn as nn
import torchvision.models as models

# =====================================================================
# 1. Task 1: Classification Models
# =====================================================================

class BaselineCNN(nn.Module):
    """
    A lightweight, custom 3-stage CNN to establish baseline classification performance.
    Architecture: 3 Conv stages -> Global Average Pooling -> Linear head
    """
    def __init__(self, num_classes: int = 9):
        super().__init__()
        self.features = nn.Sequential(
            # Stage 1: Conv -> BN -> ReLU -> Pool
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Stage 2: Conv -> BN -> ReLU -> Pool
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Stage 3: Conv -> BN -> ReLU -> Pool
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def get_resnet18_model(num_classes: int = 9, pretrained: bool = True):
    """
    Loads a ResNet-18 model, replaces the classification head, and returns it.
    If pretrained is True, loads weights from ImageNet.
    """
    if pretrained:
        # Modern torchvision API for loading pretrained weights
        from torchvision.models import ResNet18_Weights
        weights = ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    else:
        model = models.resnet18(weights=None)
        
    # Replace the classification head
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


# =====================================================================
# 2. Task 2: Segmentation Model (Lightweight U-Net)
# =====================================================================

class DoubleConv(nn.Module):
    """(conv -> BN -> ReLU) * 2"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """
    A lightweight, configurable U-Net implementation with skip connections.
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 1, init_features: int = 32):
        super().__init__()
        
        # Encoder (Downsampling path)
        self.enc1 = DoubleConv(in_channels, init_features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc2 = DoubleConv(init_features, init_features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc3 = DoubleConv(init_features * 2, init_features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(init_features * 4, init_features * 8)
        
        # Decoder (Upsampling path)
        self.up3 = nn.ConvTranspose2d(init_features * 8, init_features * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(init_features * 8, init_features * 4)
        
        self.up2 = nn.ConvTranspose2d(init_features * 4, init_features * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(init_features * 4, init_features * 2)
        
        self.up1 = nn.ConvTranspose2d(init_features * 2, init_features, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(init_features * 2, init_features)
        
        # Final Convolution
        self.conv = nn.Conv2d(init_features, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool3(enc3))
        
        # Decoder with skip connections
        up3 = self.up3(bottleneck)
        dec3 = self.dec3(torch.cat([up3, enc3], dim=1))
        
        up2 = self.up2(dec3)
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))
        
        up1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))
        
        # Final outputs
        return self.conv(dec1)


# =====================================================================
# 3. Task 3: Detection Model (Faster R-CNN)
# =====================================================================

def get_faster_rcnn_mobilenet_v3(num_classes: int = 4, pretrained: bool = True):
    """
    Loads a Faster R-CNN model with a MobileNet-V3-Large FPN backbone.
    Replacing the final predictor to support the requested number of classes (including background).
    """
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
    
    if pretrained:
        weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
    else:
        model = fasterrcnn_mobilenet_v3_large_fpn(weights=None)
        
    # Get number of input features for the classifier head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the box predictor head
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model
