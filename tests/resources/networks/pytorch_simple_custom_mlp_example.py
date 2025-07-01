import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.bias = None
    
    def forward(self, x):
        """Forward pass using functional linear."""
        return F.linear(x, self.weight, self.bias)


class SimpleMLP(nn.Module):
    """A simple MLP that uses custom linear layers to test recursive transpilation."""
    
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # Use custom linear layers instead of nn.Linear
        self.layer1 = CustomLinear(input_size, hidden_size)
        self.layer2 = CustomLinear(hidden_size, output_size)
    
    def forward(self, x):
        """Forward pass through the MLP."""
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x
