# pytorch_impl/models/mlp.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        # PyTorch handles weight initialization by default for nn.Linear
        # (Kaiming uniform for weights, zero for biases if present)

    def forward(self, x, apply_sigmoid: bool = False):
        """
        Forward pass.

        Args:
            x: Input data.
            apply_sigmoid: If True, apply sigmoid to the output (for binary classification probability).
                           The loss function BCEWithLogitsLoss handles this internally, so usually False.
        Returns:
            Model output (logits by default).
        """
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x