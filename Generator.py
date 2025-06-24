from torch import nn


class Generator(nn.Module):
    def __init__(self,d_model,vocab_size):
        super(Generator,self).__init__()
        self.d_model=d_model
        self.vocab_size=vocab_size
        self.linear=nn.Linear(d_model,vocab_size)
    def forward(self,x):
        return self.linear(x)