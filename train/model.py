import torch
from torch import nn

class CharTransformer(nn.Module):
    def __init__(self, vocab_size, seq_len=512, dim=256, n_layers=4, n_heads=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=n_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc_out = nn.Linear(dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x) * (x.size(1) ** 0.5)
        x = self.transformer(x)
        return self.fc_out(x)
