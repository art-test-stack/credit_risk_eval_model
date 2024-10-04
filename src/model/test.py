from utils import get_device
from src. eval.metric import GMean, ROCAUC

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from typing import Callable


def _eval_by_batch(
        model: nn.Module,
        X: torch.Tensor,
        desc: torch.Tensor,
        y: torch.Tensor,
        eval_metric: Callable,
        device: str | torch.device,
    ) -> torch.Tensor:
    
    with torch.no_grad():
        X, desc, y = X.to(device), desc.to(device), y.to(device)
        y_pred = model(X, desc)
        metric = eval_metric(y, y_pred) * len(X)
    return metric

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
        TensorDataset(X_test, y_test), 
        batch_size=batch_size, 
        shuffle=False
    )
    model.eval()

    metric = 0

    for idx, (X, y) in enumerate(data_loader):
        with torch.no_grad():
            # X, y = X.to(device), y.to(device)
            # y_pred = model(X)
            # metric += eval_metric(y, y_pred) * len(X)
            metric += _eval_by_batch(model, X, desc, y, eval_metric, device)
    metric /= len(data_loader.dataset)
    return metric
