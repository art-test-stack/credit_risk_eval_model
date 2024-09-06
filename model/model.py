from te import TransformerEncoder

import torch
from torch import nn
import torch.functional as F

from typing import Callable, Union

class CoreNLP:
    pass

class GloVe:
    pass

class Model(nn.Module):
    def __init__(
            self, 
            d_model: int, 
            nhead: int, 
            dim_ffn: int, 
            num_layers: int,
            dropout: float,
            # dim_dff: int,
            te_act: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.relu,
        ) -> None:
        super().__init__()
        self.corenlp = CoreNLP()
        self.glove = GloVe()
        self.te = TransformerEncoder(
            d_model=d_model,
            n_head=nhead,
            dim_ffn=dim_ffn,
            num_layers=num_layers,
            dropout=dropout,
            activation=te_act,
        )
        self.dff = nn.Linear(dim_ffn, 4)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, loans):
        corpora, text, hard_ft = loans

        word_emb = self.glove(self.corenlp(corpora))
        seg_text = self.corenlp(text)

        emb = torch.concat(word_emb, seg_text, dim=1)
        res_te = self.te(emb)

        x_dff = torch.concat(res_te, hard_ft, dim=1)
        dff_out = self.dff(x_dff)

        return self.softmax(dff_out)