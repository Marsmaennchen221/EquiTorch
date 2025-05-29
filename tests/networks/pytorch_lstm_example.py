import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        
        # Attention layer
        self.attention = nn.Linear(hidden_size, hidden_size)
        self.context_vector = nn.Parameter(torch.randn(hidden_size))
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
        
    def attention_net(self, lstm_output):
        # lstm_output shape: [batch_size, seq_len, hidden_size]
        
        # Calculate attention scores
        attention_hidden = torch.tanh(self.attention(lstm_output))
        # [batch_size, seq_len, hidden_size] @ [hidden_size] -> [batch_size, seq_len]
        attention_score = torch.matmul(attention_hidden, self.context_vector)
        attention_weights = F.softmax(attention_score, dim=1).unsqueeze(2)
        
        # Apply attention weights to LSTM outputs
        # [batch_size, seq_len, 1] * [batch_size, seq_len, hidden_size]
        context = torch.sum(attention_weights * lstm_output, dim=1)
        return context, attention_weights
        
    def forward(self, x, hidden=None):
        # x shape: [batch_size, seq_len, input_size]
        batch_size = x.size(0)
        
        if hidden is None:
            h0 = torch.zeros(1, batch_size, self.hidden_size, device=x.device)
            c0 = torch.zeros(1, batch_size, self.hidden_size, device=x.device)
            hidden = (h0, c0)
        
        # Run LSTM
        lstm_output, hidden = self.lstm(x, hidden)
        
        # Apply attention
        attn_output, attention_weights = self.attention_net(lstm_output)
        
        # Final prediction
        output = self.fc(attn_output)
        
        return output, hidden, attention_weights
