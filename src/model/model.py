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
            dropout: float = .0,
            dim_ft: int = 47,
            te_act: Union[str, Callable[[torch.Tensor], torch.Tensor]] = nn.ReLU(),
        ) -> None:
        super().__init__()
        self.te = TransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            dim_ffn=dim_ffn,
            num_layers=num_layers,
            dropout=dropout,
            activation=te_act,
        )
        self.dff = nn.Linear(d_model + dim_ft - 1, 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, X: torch.Tensor, desc: torch.Tensor) -> torch.Tensor:
        res_te = self.te(desc)

        x_dff = torch.concat((res_te, X), dim=1)
        dff_out = self.dff(x_dff)
        return self.softmax(dff_out)