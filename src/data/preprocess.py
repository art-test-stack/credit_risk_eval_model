from src.data.features import select_features, DataType, features
from src.data.corenlp import CoreNLP
from src.data.glove import GloVe

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import math

import torch
from pathlib import Path
from typing import Union, Tuple



def print_stats(df: pd.DataFrame) -> None:
    df = df.copy()
    df["target"] = df["loan_status"].apply(lambda x: "Fully Paid" in x)

    print("Paid description:", df[df["target"]]["desc_len"].describe())
    print("Default description:", df[df["target"]==False]["desc_len"].describe())
    print("All description:", df["desc_len"].describe())
    

def one_hot(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    cat_features = [ ft for ft, type in features.items() if type in [DataType.CATEGORICAL] and not ft == "loan_status"]
    df = pd.get_dummies(df, columns=cat_features, drop_first=True)

    return df


def preprocess_hard_features(df: pd.DataFrame) -> pd.DataFrame:
    raw = df.copy()

    hard_features = [ft for ft in df.columns if not features[ft] == DataType.TEXTUAL]
    raw_hard = raw[hard_features]

    # PREPROCESS HERE
    df = one_hot(raw_hard)

    raw[raw_hard] = raw_hard
    return raw

 
def preprocess_textual_feature(
        df: pd.DataFrame, 
        seg_model: CoreNLP = CoreNLP(), 
        emb_model: GloVe = GloVe()
    ) -> torch.Tensor:

    desc = df["desc"].values
    df.drop(columns="desc", inplace=True)

    # PREPROCESS HERE
    segmented_text = seg_model(desc)
    embeddings = emb_model(segmented_text, to_tensor=True) # B, S, D
    # pad all loan textual descriptions to include 200 terms for loan descriptions from LendingClub

    return embeddings


def split_data(df: pd.DataFrame, get_dev_set: bool = False) -> pd.DataFrame:
    df = df.copy()
    ftrs = list(features.copy().keys())
    ftrs.remove("loan_status")
    X, y = df[ftrs], df[["loan_status"]]

    if get_dev_set:
        X_train, X_devtest, y_train, y_devtest = train_test_split(X, y, test_size=.2, shuffle=False)
        X_dev, X_test, y_dev, y_test = train_test_split(X_devtest, y_devtest, test_size=.5, shuffle=False)

    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, shuffle=False)

    return X_train, X_dev, X_test, y_train, y_dev, y_test


def normalize(
        X_train: pd.DataFrame, 
        X_test: pd.DataFrame, 
        X_dev: pd.DataFrame | None = None,
        scaler: StandardScaler = StandardScaler(),
        return_scaler: bool = False
    ) -> None:

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    if X_dev:
        X_dev = scaler.transform(X_dev)

    if return_scaler:
        X_train, X_test, X_dev, scaler
    return X_train, X_test, X_dev


def preprocess_data(
        file_path: Union[str, Path] = Path("data/accepted_2007_to_2018Q4.csv"), 
        get_dev_set: bool = False,
        seg_model: CoreNLP = CoreNLP(), 
        emb_model: GloVe = GloVe()
    ) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    df = select_features(df)

    hard = preprocess_hard_features(df)

    X_train, X_dev, X_test, y_train, y_dev, y_test = split_data(df, get_dev_set)

    train_desc = preprocess_textual_feature(X_train)
    if get_dev_set:
        dev_desc = preprocess_textual_feature(X_dev)
    test_desc = preprocess_textual_feature(X_train)

    X_train, X_test, X_dev = normalize(X_train, X_test, X_dev)
    # TO TENSOR
    return (X_train, train_desc, y_train), (X_test, test_desc, y_test), (X_dev, dev_desc, y_dev)

