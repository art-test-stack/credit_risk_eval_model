import torch
from torch.nn import functional as F
import pandas as pd
from typing import Any, Callable
from pathlib import Path

from src.data.preprocess import preprocess_data

from src.data.stanfordnlp import StanfordNLP
from src.model.model import CREModel
from src.model.train import fit
from src.model.test import evaluate_model

from src.eval.metric import GMean, ROCAUC
from src.eval.visualize import plot_losses
from utils import get_device, PREPROCESSED_FILE, LOANS_FILE



def pipeline(
        model: CREModel,
        nlp_model: Callable,
        loans_file: Path | str = LOANS_FILE,
        do_preprocessing: bool = True,
        epochs: int = 100,
        batch_size: int = 32,
        early_stopping_patience: int = 100,
        early_stopping_min_delta: float = 1e-4,
        device: str | torch.device = get_device(),
        preprocessed_data_file: Path = PREPROCESSED_FILE,
        verbose: bool = True
    ) -> Any:
    
    print("Start preprocessing data...")
    if do_preprocessing:
        train_set, test_set, _ = preprocess_data(
            nlp_model, preprocessed_data_file, loans_file, concat_train_dev_sets=True, verbose=verbose)

        X_train, train_desc, y_train = train_set
        X_test, test_desc, y_test = test_set

        if verbose:
            print(X_train.shape, train_desc.shape, y_train.shape)

    if not do_preprocessing:
        X_train = torch.load(preprocessed_data_file.joinpath("X_train.pt"))
        train_desc = torch.load(preprocessed_data_file.joinpath("train_desc.pt"))
        y_train = torch.load(preprocessed_data_file.joinpath("y_train.pt"))

        X_test = torch.load(preprocessed_data_file.joinpath("X_test.pt"))
        test_desc = torch.load(preprocessed_data_file.joinpath("test_desc.pt"))
        y_test = torch.load(preprocessed_data_file.joinpath("y_test.pt"))

    X_train = X_train.to(device)
    train_desc = train_desc.to(device)
    y_train = y_train.to(device)

    X_test = X_test.to(device)
    test_desc = test_desc.to(device)
    y_test = y_test.to(device)
    
    y_train = y_train.to(torch.int64).reshape(-1)
    y_test = y_test.to(torch.int64).reshape(-1)
    # y_train = F.one_hot(y_train.to(torch.int64)).to(torch.float).reshape(-1,2)
    # y_test = F.one_hot(y_test.to(torch.int64)).to(torch.float).reshape(-1,2)
    
    model.to(device)
    # MODEL TRAINING
    model, history = fit(
        model, 
        X_train, 
        train_desc,
        y_train, 
        X_test,
        test_desc, 
        y_test, 
        epochs=epochs, 
        patience=early_stopping_patience,
        min_delta=early_stopping_min_delta,
        batch_size=batch_size, 
        lr=1e-4, 
        weight_decay=1e-4, 
        device=device, 
        verbose=verbose
    )

    try:
        plot_losses(history)
    except:
        if verbose:
            print("Could not plot loss")
        else:
            pass

    # EVALUATION
    auc = evaluate_model(
        model, 
        X_test, 
        test_desc,
        y_test, 
        eval_metric=ROCAUC(), 
        device=device
    )
    gmean = evaluate_model(
        model, 
        X_test,
        test_desc,
        y_test,
        eval_metric=GMean(),
        device=device
    )

    print(f"AUC: {100 * auc:.2f}, GM: {100 * gmean:.2f}")

