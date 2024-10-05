from utils import get_device
import torch
from torch import nn
import torch.functional as F

import math
from typing import Union, Callable

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int = 300, dropout: float = 0.1, max_len: int = 200):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0), :]
        # return self.dropout(x)
        return x
    

class TransformerEncoder(nn.Module):
    def __init__(
            self, 
            d_model: int, 
            nhead: int, 
            dim_ffn: int, 
            num_layers: int,
            dropout: float,
            activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = nn.ReLU()
        ) -> None:
        super().__init__()

        self.pos_enc = PositionalEncoding(d_model=d_model, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ffn,
            dropout=dropout,
            activation=activation
        )
        self.main = nn.TransformerEncoder(
            encoder_layer=enc_layer,
            num_layers=num_layers
        )
    
    def forward(self, x):
        # x_pe = self.pos_enc(x)
        return self.main(x)[:,0,:].reshape(x.size(0), -1)
