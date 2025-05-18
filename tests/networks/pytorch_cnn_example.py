import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, num_classes)
        
    def forward(self, x):
        # Input shape: [batch_size, in_channels, 28, 28]
        x = F.relu(self.conv1(x))  # -> [batch_size, 16, 28, 28]
        x = F.max_pool2d(x, 2)     # -> [batch_size, 16, 14, 14]
        x = F.relu(self.conv2(x))  # -> [batch_size, 32, 14, 14]
        x = F.max_pool2d(x, 2)     # -> [batch_size, 32, 7, 7]
        x = x.view(x.size(0), -1)  # -> [batch_size, 32 * 7 * 7]
        x = self.fc(x)             # -> [batch_size, num_classes]
        return x
