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
        loans_file: Path | str = LOANS_FILE,
        model: CREModel = CREModel(),
        nlp_model: Callable = StanfordNLP(),
        do_preprocessing: bool = True,
        epochs: int = 100,
        device: str | torch.device = get_device(),
        preprocessed_data_file: Path = PREPROCESSED_FILE
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
        torch.save(X_train, preprocessed_data_file.joinpath("X_train.pt"))
        torch.save(train_desc, preprocessed_data_file.joinpath("train_desc.pt"))
        torch.save(y_train, preprocessed_data_file.joinpath("y_train.pt"))

        torch.save(X_test, preprocessed_data_file.joinpath("X_test.pt"))
        torch.save(test_desc, preprocessed_data_file.joinpath("test_desc.pt"))
        torch.save(y_test, preprocessed_data_file.joinpath("y_test.pt"))

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
    
    y_train = F.one_hot(y_train.to(torch.int64)).to(torch.float).reshape(-1,2)
    y_test = F.one_hot(y_test.to(torch.int64)).to(torch.float).reshape(-1,2)
    print(y_train.shape)
    print(y_test.shape)
    print("X_train.device", X_train.device)
    print("train_desc.device", train_desc.device)
    print("y_train.device", y_train.device)

    print("X_test.device", X_test.device)
    print("test_desc.device", test_desc.device)
    print("y_test.device", y_test.device)
    
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

