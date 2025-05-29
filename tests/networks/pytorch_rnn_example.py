import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hidden=None):
        # x shape: [batch_size, seq_len, input_size]
        if hidden is None:
            # Initialize hidden state
            hidden = torch.zeros(1, x.size(0), self.hidden_size, device=x.device)
            
        # Run RNN
        output, hidden = self.rnn(x, hidden)
        
        # Use output of last time step for classification
        output = self.fc(output[:, -1, :])
        return output, hidden
    
    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)
