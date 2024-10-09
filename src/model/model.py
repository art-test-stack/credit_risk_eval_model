from src.model.te import TransformerEncoder

import torch
from torch import nn

from typing import Callable, Union


class CREModel(nn.Module):
    """
    Credit risk evaluation model with textual features from loan descriptions for P2P lending
    """
    def __init__(
            self, 
            d_model: int = 200,
            nhead: int = 8,
            dim_ffn: int = 50,
            num_layers: int = 1,
            dropout: float = .1,
            dim_ft: int = 47,
            te_act: Union[str, Callable[[torch.Tensor], torch.Tensor]] = nn.ReLU(),
        ) -> None:
        super().__init__()
        self.dropout_rate = dropout
        self.te = TransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            dim_ffn=dim_ffn,
            num_layers=num_layers,
            dropout=dropout,
            activation=te_act,
        )
        self.dff = nn.Sequential(
            nn.Linear(d_model + dim_ft - 1, 10),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(10, 2),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Softmax(dim=-1)
        )
        

    def forward(self, x: torch.Tensor, desc: torch.Tensor) -> torch.Tensor:
        te_out = self.te(desc)

        x = torch.concat((te_out, x), dim=1)
        
        return self.dff(x)