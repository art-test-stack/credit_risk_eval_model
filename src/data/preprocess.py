from src.data.features import features, features_pre_processing

import pandas as pd

import math
from pathlib import Path
from typing import Union


convert_months = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}


def print_stats(df: pd.DataFrame) -> None:
    df = df.copy()
    df["target"] = df["loan_status"].apply(lambda x: "Fully Paid" in x)

    print("Paid description:", df[df["target"]]["desc_len"].describe())
    print("Default description:", df[df["target"]==False]["desc_len"].describe())
    print("All description:", df["desc_len"].describe())
    
    
def preprocess_data(file_path: Union[str, Path] = "data/accepted_2007_to_2018Q4.csv") -> pd.DataFrame:
    raw = pd.read_csv(file_path)

    raw["desc_len"] = raw.dropna(subset="desc")["desc"].apply(lambda x: len(str(x).split()))
    raw = raw.dropna(subset=list(features_pre_processing.keys()))

    raw = raw[raw["desc_len"] > 20]

    # COMPUTE fico
    raw["fico"] = (raw["fico_range_high"] - raw["fico_range_low"]) / 2


    # COMPUTE "revol_inc_rat"
    # revol_inc_rat = revolving credit balance /  borrower’s monthly income
    raw["revol_inc_rat"] = (raw["revol_bal"] / (raw["annual_inc"] / 12)) # .apply(lambda x: math.log(x))

    # COMPUTE "credit_age"
    get_month = lambda x: int(convert_months[str(x).split("-")[0]])
    get_year = lambda x: int(str(x).split("-")[1])

    raw["issue_d_m"] = raw["issue_d"].apply(get_month)
    raw["issue_d_y"] = raw["issue_d"].apply(get_year)
    raw = raw[(raw["issue_d_y"] >= 2007) & (raw["issue_d_y"] < 2014)]

    raw["earliest_cr_line_m"] = raw["earliest_cr_line"].apply(get_month)
    raw["earliest_cr_line_y"] = raw["earliest_cr_line"].apply(get_year)
    raw["credit_age"] = (raw["issue_d_y"] - raw["earliest_cr_line_y"]) * 12 + raw["issue_d_m"] - raw["earliest_cr_line_m"]

    # COMPUTE LOG VALUES
    raw["log_loan_amnt"] = raw["loan_amnt"].apply(lambda x: math.log(x))
    raw["log_annual_inc"] = raw["annual_inc"].apply(lambda x: math.log(x))

    # Select only interesting features
    res = raw[list(features.keys())]

    print_stats(res)

    return res