import torch
import torch.nn as nn

"""
    Applies positional encoding and passes to multihead attention layer
    Helps to abstract the query, key, value computation
"""
class SoftmaxAttention(nn.Module):
    def __init__(self, nb_in: int, dim_head: int = 64, heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim_head*heads, heads, dropout=dropout)

        self.to_qkv = nn.Linear(nb_in, dim_head*heads*3, bias=False)
        self.to_out = nn.Linear(dim_head*heads, nb_in)
        nn.init.kaiming_normal_(self.to_qkv.weight, nonlinearity='linear')
        nn.init.kaiming_normal_(self.to_out.weight, nonlinearity='linear')

    def forward(self, x: torch.FloatTensor, mask: torch.BoolTensor = None):
        xq, xk, xv = self.to_qkv(x).chunk(3, dim=-1)
        x, _ = self.attn(xq.transpose(0, 1), xk.transpose(0, 1), xv.transpose(0, 1), key_padding_mask = ~mask)
        x = self.to_out(x.transpose(0, 1))
        return x
