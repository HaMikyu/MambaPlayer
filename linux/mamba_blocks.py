from mamba_ssm import Mamba
from torch import nn
import torch
from torch.nn import Module,Dropout,LayerNorm
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mamba_simple import Mamba

class MambaBlock(Module):
    def __init__(self, embed_dim, dropout_level=0,dtype=torch.bfloat16,mamba2=False):
        super().__init__()
        if mamba2:
            self.mamba =  Mamba2(d_model=embed_dim, d_state=16, d_conv=4, expand=2,dtype=dtype)
        else:
            self.mamba =  Mamba(d_model=embed_dim, d_state=16, d_conv=4, expand=2,dtype=dtype)
        self.norm = LayerNorm(embed_dim,dtype=dtype)
        self.dropout = Dropout(dropout_level)

    def forward(self, x):
        x = self.norm(self.mamba(x) + x)
        return self.dropout(x)

class MambaTower(nn.Module):
    def __init__(self, embed_dim, n_layers, global_pool=False, dropout=0,dtype=None,mamba2=True):
        super().__init__()
        self.blocks = nn.Sequential(*[MambaBlock(embed_dim, dropout_level=dropout,dtype=dtype,mamba2=mamba2) for _ in range(n_layers)])
        self.global_pool = global_pool

    def forward(self, x):
        out = self.blocks(x) if not self.global_pool else torch.mean(self.blocks(x),1)
        return out

class ThresholdLayer(nn.Module):
    def __init__(self, s):
        super(ThresholdLayer, self).__init__()
        self.s=s
    def forward(self, x):
        return torch.where(x > self.s['threshold'], torch.tensor(1, dtype=self.s['dtype']), torch.tensor(0, dtype=self.s['dtype']))
