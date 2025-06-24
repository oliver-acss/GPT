import torch
from torch import nn
from torch.cuda import device


class GPT_Attention(nn.Module):
    def __init__(self, d_model,n_head,d_ff,dropout):
        super(GPT_Attention, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_ff = d_ff
        self.dropout = dropout
        self.wq=nn.Linear(d_model,d_model)
        self.wk=nn.Linear(d_model,d_model)
        self.wv=nn.Linear(d_model,d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout,batch_first=True)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.attn_weights=None
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.Dropout(dropout),
            nn.Tanh(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        Q=self.wq(x)
        K=self.wk(x)
        V=self.wv(x)
        attn_mask=torch.triu(torch.ones(x.shape[1],x.shape[1]),diagonal=1).bool()
        attn_mask=attn_mask.to(x.device)
        attn_output,attn_weights=self.attn(Q,K,V,attn_mask=attn_mask,need_weights=True)
        self.attn_weights=attn_weights
        attn_output=attn_output+x#残差连接
        attn_output=self.layer_norm1(attn_output)
        attn_output=attn_output+self.feed_forward(attn_output)
        attn_output=self.layer_norm2(attn_output)
        return attn_output
