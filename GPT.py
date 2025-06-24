import copy

from torch import nn

from GPT_Attention import GPT_Attention
from GPT_Embedding import GPT_Embedding
from Generator import Generator

def getGPT_Attention(d_model,n_head,d_ff,dropout):
    return GPT_Attention(d_model,n_head,d_ff,dropout)
class GPT(nn.Module):
    def __init__(self, d_model, vocab_size,n_head,d_ff,max_len,dropout):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_head = n_head
        self.d_ff = d_ff
        self.dropout = dropout
        self.max_len = max_len
        self.generator=Generator(d_model,vocab_size)
        self.embedding = GPT_Embedding(max_len,d_model,vocab_size)
        def gpt_attetion_block(n):
            model_list=list()
            for i in range(n):
                model_list.append(getGPT_Attention(d_model,n_head,d_ff,dropout))
            return model_list
        self.GPT_block=nn.Sequential(*gpt_attetion_block(1))
    def forward(self,x):
        x = self.embedding(x)
        x = self.GPT_block(x)
        x = self.generator(x)
        return x