import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, vocab_size=1000, hidden_size=768, num_layers=6, num_heads=8):
        super().__init__()
        # 这里简化，实际应构建Transformer
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(hidden_size, num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.classifier = nn.Linear(hidden_size, 10)

    def forward(self, x):
        # x: (batch_size, seq_len)
        x = self.embedding(x)  # (batch_size, seq_len, hidden_size)
        x = x.transpose(0, 1)  # (seq_len, batch_size, hidden_size)
        x = self.transformer(x)
        x = x.mean(dim=0)  # (batch_size, hidden_size)
        return self.classifier(x)