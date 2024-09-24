from typing import Any, Union, List

import spacy

import numpy as np
import torch


class GloVe:
    def __init__(self, model: str = "en_core_web_lg") -> None:
    
        self.model = spacy.load(model)
        pass

    def __call__(self, tokens: List[str], to_tensor: bool = True) -> Union[np.ndarray, torch.Tensor]:
        return self.forward(tokens)

    def forward(self, tokens: List[str], to_tensor: bool = True) -> Union[np.ndarray, torch.Tensor]:
        emb = np.array([ self.model.vocab.get_vector(tok) for tok in tokens ])

        if to_tensor:
            emb = torch.Tensor(emb)
            
        return emb