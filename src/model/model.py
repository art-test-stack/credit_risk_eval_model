from src.model.te import TransformerEncoder

import torch
from torch import nn
import torch.functional as F

from typing import Callable, Union


class CREModel(nn.Module):
    """
    Credit risk evaluation model with textual features from loan descriptions for P2P lending
    """
    def __init__(
            self, 
            d_model: int = 300,
            nhead: int = 8,
            dim_ffn: int = 50,
            num_layers: int = 1,
            dropout: float = .0,
            te_act: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.relu,
        ) -> None:
        super().__init__()
        self.te = TransformerEncoder(
            d_model=d_model,
            n_head=nhead,
            dim_ffn=dim_ffn,
            num_layers=num_layers,
            dropout=dropout,
            activation=te_act,
        )
        self.dff = nn.Linear(dim_ffn, 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, emb: torch.Tensor, hard_ft: torch.Tensor) -> torch.Tensor:
        # desc_text, hard_features = loans

        # seg_text = self.corenlp(desc_text)
        # emb = self.glove(seg_text, to_tensor=True)

        res_te = self.te(emb)

        x_dff = torch.concat(res_te, hard_ft, dim=1)
        dff_out = self.dff(x_dff)

        return self.softmax(dff_out)