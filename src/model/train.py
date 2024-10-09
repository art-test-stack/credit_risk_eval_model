from src.eval.metric import ROCAUC, GMean
from src.model.model import CREModel
from utils import get_device, MODEL_DIR

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple

from tqdm import tqdm


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.05):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.save_model = True

    def __call__(self, test_loss):
        self.save_model = False
        if self.best_loss is None:
            self.best_loss = test_loss
            self.save_model = True
        elif test_loss < self.best_loss - self.min_delta:
            self.best_loss = test_loss
            self.counter = 0
            self.save_model = True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

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
        patience: int = 100,
        min_delta: float = 1e-4,
        device: str | torch.device = get_device(),
        verbose: bool = True
    ) -> Tuple[nn.Module, dict]:
    model = model.to(device)
    optimizer = optim.Adam(
        model.parameters(), 
        lr=lr, 
        weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss(reduction="sum")
    auc = ROCAUC()
    gmean = GMean()
    train_loader = DataLoader(TensorDataset(X_train, desc_train, y_train), batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(TensorDataset(X_test, desc_test, y_test), batch_size=batch_size, shuffle=False)

    history = {"model_name": model.name, "train_loss": [], "test_loss": []}

    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

    with tqdm(range(epochs), unit="epoch", disable=not verbose) as tepoch:
        
        for epoch in tqdm(range(epochs)):    
            model.train()
            train_loss = 0
            for X_batch, desc_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = model(X_batch, desc_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            history["train_loss"].append(train_loss / len(train_loader.dataset))

            if X_test is not None and y_test is not None:
                model.eval()
                auc.reset()
                gmean.reset()
                test_loss = 0
                with torch.no_grad():
                    for X_batch, desc_batch, y_batch in dev_loader:
                        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                        y_pred = model(X_batch, desc_batch)
                        loss = criterion(y_pred, y_batch)
                        test_loss += loss.item()
                        auc(y_pred, y_batch)
                        gmean(y_pred, y_batch)

                history["test_loss"].append(test_loss / len(dev_loader.dataset))

            auc_value = auc.compute().cpu().numpy()
            gmean_value = gmean.compute().cpu().numpy()

            tepoch.set_postfix(
                loss = history["train_loss"][-1],
                test_loss = history["test_loss"][-1], 
                auc = 100. * auc_value,
                gmean = 100. * gmean_value,
            )

            # early_stopping(history["test_loss"][-1])

            # early_stopping(-(max(auc_value, gmean_value)))
            # early_stopping(-(auc_value + gmean_value)))
            early_stopping(-(auc_value * gmean_value))

            if early_stopping.save_model:
                best_model_state = model.state_dict()
                torch.save(best_model_state, MODEL_DIR.joinpath(model.name))

            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch}")
                break

            
            tepoch.update(1)

    best_model = CREModel(model.name)
    best_model.load_state_dict(torch.load(MODEL_DIR.joinpath(model.name)))
    return best_model, history
   