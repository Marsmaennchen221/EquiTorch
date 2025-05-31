import torch
import torch.nn as nn

class NestedCustomCNN(nn.Module):
    """A CNN module that uses multiple types of modules to test recursive transpilation.
    
    This module combines:
    - Manual converters: Conv2d, BatchNorm2d, Linear, Dropout
    - Auto-transpiled modules: GroupNorm, LayerNorm, AdaptiveAvgPool2d
    """
    
    def __init__(self, input_channels, num_classes):
        super().__init__()
        
        # First conv block with manual converters
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Second conv block with auto-transpiled GroupNorm
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(8, 64)  # Auto-transpiled
        
        # Third conv block with manual converters
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Auto-transpiled adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))  # Auto-transpiled
        
        # Classifier with mixed modules
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),  # Manual converter
            nn.LayerNorm(256),            # Auto-transpiled
            nn.Dropout(0.5),              # Manual converter
            nn.Linear(256, num_classes)   # Manual converter
        )
        
    def forward(self, x):
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2)
        
        # Second conv block with GroupNorm
        x = self.conv2(x)
        x = self.gn2(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2)
        
        # Third conv block
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        
        # Adaptive pooling
        x = self.adaptive_pool(x)
        
        # Flatten for classifier
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.classifier(x)
        
        return x


class CustomResidualBlock(nn.Module):
    """A custom residual block that uses auto-transpiled modules."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Main path with manual converters
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Use auto-transpiled GroupNorm instead of BatchNorm for second layer
        self.gn2 = nn.GroupNorm(8, out_channels)  # Auto-transpiled
        
        # Skip connection (if needed)
        self.skip_conv = None
        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            
    def forward(self, x):
        identity = x
        
        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        
        out = self.conv2(out)
        out = self.gn2(out)
        
        # Skip connection
        if self.skip_conv is not None:
            identity = self.skip_conv(x)
            
        out += identity
        out = torch.relu(out)
        
        return out


class DeepNestedCNN(nn.Module):
    """A deep CNN that nests the custom residual blocks to test deep recursion."""
    
    def __init__(self, input_channels, num_classes):
        super().__init__()
        
        # Initial conv with manual converter
        self.initial_conv = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.initial_bn = nn.BatchNorm2d(64)
        
        # Nested custom blocks (these will trigger recursive transpilation)
        self.block1 = CustomResidualBlock(64, 64)
        self.block2 = CustomResidualBlock(64, 128)
        self.block3 = CustomResidualBlock(128, 256)
        
        # Auto-transpiled pooling and normalization
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Auto-transpiled
        
        # Final classifier with auto-transpiled LayerNorm
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),     # Manual converter
            nn.LayerNorm(128),       # Auto-transpiled
            nn.Dropout(0.3),         # Manual converter
            nn.Linear(128, num_classes)  # Manual converter
        )
        
    def forward(self, x):
        # Initial processing
        x = self.initial_conv(x)
        x = self.initial_bn(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        
        # Nested blocks (recursive transpilation)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.classifier(x)
        
        return x
