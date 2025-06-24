import torch
import torch.nn as nn
class GPT_Embedding(nn.Module):
    def __init__(self, maxlen,d_model,vocab_size):
        super(GPT_Embedding, self).__init__()
        self.positionEmbedding = nn.Embedding(maxlen, d_model)
        self.wordEmbedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        position=torch.arange(0, x.size(1), dtype=torch.long, device=x.device)
        position=self.positionEmbedding(position)
        x=self.wordEmbedding(x)
        return x+position