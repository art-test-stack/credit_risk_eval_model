from src.data.preprocess import preprocess_data

from src.data.stanfordnlp import StanfordNLP
from src.model.model import CREModel

from src.eval.metric import GMean, AUC

import torch
import pandas as pd
from typing import Any, Callable
from pathlib import Path


def pipeline(
        loans_file: Path | str = Path("data/accepted_2007_to_2018Q4.csv"),
        model: CREModel = CREModel(),
        nlp_model: Callable = StanfordNLP(),
    ) -> Any:
    
    print("Start preprocessing data...")
    train_set, test_set, dev_set = preprocess_data(loans_file, nlp_model)

    X_train, train_desc, y_train = train_set
    X_test, test_desc, y_test = test_set
    X_dev, dev_desc, y_dev = dev_set

    print(X_train.shape, train_desc.shape, y_train.shape)

    torch.save(X_train, Path("data/preprocessed/").joinpath("X_train.pt"))
    torch.save(train_desc, Path("data/preprocessed/").joinpath("train_desc.pt"))
    torch.save(y_train, Path("data/preprocessed/").joinpath("y_train.pt"))

    torch.save(X_test, Path("data/preprocessed/").joinpath("X_test.pt"))
    torch.save(test_desc, Path("data/preprocessed/").joinpath("test_desc.pt"))
    torch.save(y_test, Path("data/preprocessed/").joinpath("y_test.pt"))
    # CONCAT training sets


    # TODO:
    # MODEL TRAINING
    # TO device

    # EVALUATION


