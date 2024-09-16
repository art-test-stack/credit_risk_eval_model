from src.data.features import select_features, DataType, features

import pandas as pd

from sklearn.model_selection import train_test_split

import math
from pathlib import Path
from typing import Union



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

 
def preprocess_textual_feature(df: pd.DataFrame) -> pd.DataFrame:
    raw = df.copy()
    desc = raw["desc"]

    # PREPROCESS HERE

    raw["desc"] = desc
    return raw


def split_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ftrs = list(features.copy().keys())
    ftrs.remove("loan_status")
    X, y = df[ftrs], df[["loan_status"]]
    X_train, X_devtest, y_train, y_devtest = train_test_split(X, y, test_size=.2)
    X_dev, X_test, y_dev, y_test = train_test_split(X_devtest, y_devtest, test_size=.5)

    return X_train, X_dev, X_test, y_train, y_dev, y_test


def preprocess_data(file_path: Union[str, Path] = "data/accepted_2007_to_2018Q4.csv") -> pd.DataFrame:
    df = pd.read_csv(file_path)

    df = select_features(df)

    X_train, X_dev, X_test, y_train, y_dev, y_test = split_data(df)

    df = preprocess_hard_features(df)
    df = preprocess_textual_feature(df)

