import torch
from torch import nn
import torch.functional as F

from typing import Union, Callable


class TransformerEncoder(nn.Module):
    def __init__(
            self, 
            d_model: int, 
            nhead: int, 
            dim_ffn: int, 
            num_layers: int,
            dropout: float,
            activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.relu,
        ) -> None:
        super().__init__()

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
        return self.main(x)[:,-1,:].view(x.size(0), -1)