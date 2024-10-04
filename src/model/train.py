from src.eval.metric import AUC, GMean
from utils import get_device
# from src.model.test import _eval_by_batch

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple

from tqdm import tqdm

def fit(
        model: nn.Module,
        X_train: torch.Tensor,
        desc_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: torch.Tensor | None = None,
        desc_test: torch.Tensor | None = None,
        y_test: torch.Tensor | None = None,
        epochs: int = 10,
        batch_size: int = 32,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        device: str | torch.device = get_device(),
        verbose: bool = True
    ) -> Tuple[nn.Module, dict]:
    model = model.to(device)
    optimizer = optim.Adam(
        model.parameters(), 
        lr=lr, 
        weight_decay=weight_decay
    )
    criterion = nn.BCELoss()
    # criterion = nn.BCEWithLogitsLoss()
    auc = AUC()
    gmean = GMean()
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    history = {"train_loss": [], "test_loss": []}

    for epoch in tqdm(range(epochs)):

        with tqdm(train_loader, unit="batch", disable=not verbose) as tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            model.train()
            train_loss = 0
            for idx, (X_batch, y_batch) in enumerate(train_loader):
                # X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                y_pred = model(X_batch, desc_train[idx])
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            history["train_loss"].append(train_loss.cpu().numpy() / len(train_loader))

            if X_test is not None and y_test is not None:
                model.eval()
                test_loss = 0
                auc_v = 0
                gmean_v = 0
                with torch.no_grad():
                    for idx, (X_batch, y_batch) in enumerate(dev_loader):
                        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                        y_pred = model(X_batch, desc_test[idx])
                        loss = criterion(y_pred, y_batch)
                        test_loss += loss.item()
                        auc_v = auc(y_batch, y_pred) * len(X_batch)
                        gmean_v = gmean(y_batch, y_pred) * len(X_batch)

                auc_v /= len(dev_loader.dataset)
                gmean_v /= len(dev_loader.dataset)
                history["test_loss"].append(test_loss.cpu().numpy()  / len(dev_loader))

            tepoch.set_postfix(
                loss = history["train_loss"][-1],
                test_loss = history["test_loss"][-1], 
                auc = 100. * auc_v,
                gmean = 100. * gmean_v
            )

    return model, history
