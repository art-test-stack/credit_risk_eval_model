from utils import get_device
from src. eval.metric import GMean, ROCAUC

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from typing import Callable


def evaluate_model(
        model: nn.Module,
        X_test: torch.Tensor,
        desc: torch.Tensor,
        y_test: torch.Tensor,
        batch_size: int = 32,
        eval_metric: Callable = GMean(),
        device: str | torch.device = get_device()
    ) -> None:
    model = model.to(device)

    data_loader = DataLoader(
        TensorDataset(X_test, desc, y_test), 
        batch_size=batch_size, 
        shuffle=False
    )
    model.eval()
    eval_metric.reset()

    for X, desc, y in data_loader:
        with torch.no_grad():
            y_pred = model(X, desc)
            eval_metric(y_pred, y)
    return eval_metric.compute()
