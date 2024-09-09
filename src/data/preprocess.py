from src.data.features import features

import pandas as pd

from pathlib import Path
from typing import Union


def print_stats(df: pd.DataFrame) -> None:
    df = df.copy()
    df["target"] = df["loan_status"].apply(lambda x: "Fully Paid" in x)

    print("Paid description:", df[df["target"]]["desc_len"].describe())
    print("Default description:", df[df["target"]==False]["desc_len"].describe())
    print("All description:", df["desc_len"].describe())
    
    
def preprocess_data(file_path: Union[str, Path] = "data/accepted_2007_to_2018Q4.csv") -> pd.DataFrame:
    raw = pd.read_csv(file_path)

    raw["desc_len"] = raw.dropna(subset="desc")["desc"].apply(lambda x: len(str(x).split()))

    res = list(map(lambda x: str(x).split("-"), raw["issue_d"].values.tolist()))
    not_nan_dates = [ date for date in res if 'nan' not in date ]
    args = [ idx for idx, [_, y] in enumerate(not_nan_dates) if int(y) <= 2014 and int(y) >= 2007]

    res = raw.iloc[args]
    res = res[res["desc_len"] > 20]
    _res = res[features.keys()].dropna()

    print("With total_il_high_credit_limit and revol_util features")
    print_stats(_res)

    features.pop("total_il_high_credit_limit")
    features.pop("revol_util") 
    res = res[features.keys()].dropna()

    print("\nWithout")
    print_stats(res)