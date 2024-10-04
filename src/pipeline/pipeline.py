from src.data.preprocess import preprocess_data

from src.data.stanfordnlp import StanfordNLP
from src.model.model import CREModel
from src.model.train import fit
from src.model.test import evaluate_model

from src.eval.metric import GMean, AUC
from src.eval.visualize import plot_losses
from utils import get_device

import torch
import pandas as pd
from typing import Any, Callable
from pathlib import Path


def pipeline(
        loans_file: Path | str = Path("data/accepted_2007_to_2018Q4.csv"),
        model: CREModel = CREModel(),
        nlp_model: Callable = StanfordNLP(),
        do_preprocessing: bool = True,
        device: str | torch.device = get_device()
    ) -> Any:
    
    print("Start preprocessing data...")
    if do_preprocessing:
        train_set, test_set, dev_set = preprocess_data(
            loans_file, nlp_model, concat_train_dev_sets=True)

        X_train, train_desc, y_train = train_set
        X_test, test_desc, y_test = test_set
        X_dev, dev_desc, y_dev = dev_set

        print(X_train.shape, train_desc.shape, y_train.shape)

        print("Saving preprocessed data...")
        torch.save(X_train, Path("data/preprocessed/").joinpath("X_train.pt"))
        torch.save(train_desc, Path("data/preprocessed/").joinpath("train_desc.pt"))
        torch.save(y_train, Path("data/preprocessed/").joinpath("y_train.pt"))

        torch.save(X_test, Path("data/preprocessed/").joinpath("X_test.pt"))
        torch.save(test_desc, Path("data/preprocessed/").joinpath("test_desc.pt"))
        torch.save(y_test, Path("data/preprocessed/").joinpath("y_test.pt"))

    if not do_preprocessing:
        X_train = torch.load(Path("data/preprocessed/").joinpath("X_train.pt"))
        train_desc = torch.load(Path("data/preprocessed/").joinpath("train_desc.pt"))
        y_train = torch.load(Path("data/preprocessed/").joinpath("y_train.pt"))

        X_test = torch.load(Path("data/preprocessed/").joinpath("X_test.pt"))
        test_desc = torch.load(Path("data/preprocessed/").joinpath("test_desc.pt"))
        y_test = torch.load(Path("data/preprocessed/").joinpath("y_test.pt"))

        X_train.to(device)
        train_desc.to(device)
        y_train.to(device)

        X_test.to(device)
        test_desc.to(device)
        y_test.to(device)
    
    # MODEL TRAINING
    model, history = fit(
        model, 
        X_train, 
        train_desc,
        y_train, 
        X_test,
        test_desc, 
        y_test, 
        epochs=100, 
        batch_size=1024, 
        lr=1e-4, 
        weight_decay=1e-4, 
        device=device, 
        verbose=True
    )

    try:
        plot_losses(history)
    except:
        print("Could not plot loss")

    # EVALUATION
    auc = evaluate_model(
        model, 
        X_test, 
        test_desc,
        y_test, 
        eval_metric=AUC(), 
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

    print(f"AUC: {auc:.4f}, GM: {gmean:.4f}")

