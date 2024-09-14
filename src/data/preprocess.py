from src.data.features import select_features

import pandas as pd

import math
from pathlib import Path
from typing import Union



def print_stats(df: pd.DataFrame) -> None:
    df = df.copy()
    df["target"] = df["loan_status"].apply(lambda x: "Fully Paid" in x)

    print("Paid description:", df[df["target"]]["desc_len"].describe())
    print("Default description:", df[df["target"]==False]["desc_len"].describe())
    print("All description:", df["desc_len"].describe())
    

def preprocess_hard_features(df: pd.DataFrame) -> pd.DataFrame:
    pass

 
def preprocess_textual_feature(df: pd.DataFrame) -> pd.DataFrame:
    pass


def preprocess_data(file_path: Union[str, Path] = "data/accepted_2007_to_2018Q4.csv") -> pd.DataFrame:
    df = pd.read_csv(file_path)

    df = select_features(df)
