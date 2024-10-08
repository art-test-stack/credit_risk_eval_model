from src.data.features import select_features, DataType, features, features_numerical
from src.data.stanfordnlp import StanfordNLP

from utils import RANDOM_STATE

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

import math

import torch
from pathlib import Path
from typing import Union, Tuple, Callable


features_post_preprocessing = [
    'loan_status', 'log_loan_amnt', 'int_rate', 'fico', 'inq_last_6mths',
    'revol_util', 'delinq_2yrs', 'pub_rec', 'open_acc', 'revol_inc_rat',
    'total_acc', 'credit_age', 'log_annual_inc', 'emp_length', 'dti',
    'desc', 'desc_len', 'term_ 36 months', 'term_ 60 months', 'purpose_car',
    'purpose_credit_card', 'purpose_debt_consolidation',
    'purpose_educational', 'purpose_home_improvement', 'purpose_house',
    'purpose_major_purchase', 'purpose_medical', 'purpose_moving',
    'purpose_other', 'purpose_renewable_energy', 'purpose_small_business',
    'purpose_vacation', 'purpose_wedding', 'grade_A', 'grade_B', 'grade_C',
    'grade_D', 'grade_E', 'grade_F', 'grade_G', 'home_ownership_MORTGAGE',
    'home_ownership_NONE', 'home_ownership_OTHER', 'home_ownership_OWN',
    'home_ownership_RENT', 'verification_status_Not Verified',
    'verification_status_Source Verified', 'verification_status_Verified'
]

def print_stats(df: pd.DataFrame) -> None:
    df = df.copy()
    df["target"] = df["loan_status"].apply(lambda x: "Fully Paid" in x)

    print("Paid description:", df[df["target"]]["desc_len"].describe())
    print("Default description:", df[df["target"]==False]["desc_len"].describe())
    print("All description:", df["desc_len"].describe())
    

def one_hot(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    cat_features = [ ft for ft, type in features.items() if type in [DataType.CATEGORICAL] and not ft == "loan_status"]
    df = pd.get_dummies(df, columns=cat_features)

    return df


def preprocess_hard_features(df: pd.DataFrame) -> pd.DataFrame:
    raw = df.copy()

    hard_features = [ft for ft in df.columns if not features[ft] == DataType.TEXTUAL]
    
    raw_hard = raw[hard_features]

    # PREPROCESS HERE
    df = one_hot(raw_hard)

    df['desc'] = raw["desc"]
    return df

 
def preprocess_textual_feature(
        df: pd.DataFrame, 
        nlp_model: Callable = StanfordNLP(),
    ) -> torch.Tensor:

    embeddings = nlp_model.process_batch(df["desc"].values)
    df.drop(columns="desc", inplace=True)
    return torch.Tensor(embeddings)


def split_data(df: pd.DataFrame, get_dev_set: bool = True) -> pd.DataFrame:
    df = df.copy()
    ftrs = features_post_preprocessing.copy()
    ftrs.remove("loan_status")
    X, y = df[ftrs], df[["loan_status"]]

    if get_dev_set:
        X_train, X_devtest, y_train, y_devtest = train_test_split(X, y, test_size=.2, shuffle=True, random_state=RANDOM_STATE)
        X_dev, X_test, y_dev, y_test = train_test_split(X_devtest, y_devtest, test_size=.5, shuffle=True, random_state=RANDOM_STATE)

    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, shuffle=True, random_state=RANDOM_STATE)
        return X_train, X_test, y_train, y_test,

    return X_train, X_dev, X_test, y_train, y_dev, y_test


def normalize(
        X_train: pd.DataFrame, 
        X_test: pd.DataFrame, 
        X_dev: pd.DataFrame | None = None,
        scaler: StandardScaler = StandardScaler(),
        return_scaler: bool = False
    ) -> None:

    X_train[features_numerical] = scaler.fit_transform(X_train[features_numerical])
    X_test[features_numerical] = scaler.transform(X_test[features_numerical])
    # if X_dev:
    X_dev[features_numerical] = scaler.transform(X_dev[features_numerical])

    if return_scaler:
        X_train, X_test, X_dev, scaler
    return X_train, X_test, X_dev


def balance_training_data(
        X: pd.DataFrame,
        y: pd.DataFrame
    ) -> Tuple[pd.DataFrame,torch.Tensor, pd.DataFrame]:
    df = X.copy()
    # X = X.drop(columns="desc")
    ros = RandomOverSampler(sampling_strategy=0.5, random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y)

    rus = RandomUnderSampler(sampling_strategy=1.0, random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X_resampled, y_resampled)

    # resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
    # resampled_df["loan_status"] = y_resampled
    # resampled_df["desc"] = df.loc[X_resampled.index, "desc"].values

    return X_resampled, y_resampled 


def preprocess_data(
        file_path: Union[str, Path] = Path("data/accepted_2007_to_2018Q4.csv"), 
        nlp_model: Callable = StanfordNLP(), 
        concat_train_dev_sets: bool = False
    ) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    print("Select features...")
    df = select_features(df)

    print("Preprocess hard-features...")
    df = preprocess_hard_features(df)

    print("Split dataset...")
    X_train, X_dev, X_test, y_train, y_dev, y_test = split_data(df)

    print("Balance training set...")
    X_train, y_train = balance_training_data(X_train, y_train)
    

    print("Preprocess training textual features...")
    train_desc = preprocess_textual_feature(X_train, nlp_model)
    dev_desc = preprocess_textual_feature(X_dev, nlp_model)
    test_desc = preprocess_textual_feature(X_test, nlp_model)

    print("Normalize...")
    X_train, X_test, X_dev, = normalize(X_train, X_test, X_dev)

    X_train = torch.Tensor(X_train.astype(float).values)
    X_dev = torch.Tensor(X_dev.astype(float).values)
    X_test = torch.Tensor(X_test.astype(float).values)
    
    y_train = torch.Tensor(y_train.astype(float).values)
    y_dev = torch.Tensor(y_dev.astype(float).values)
    y_test = torch.Tensor(y_test.astype(float).values)

    if concat_train_dev_sets:
        X_train = torch.cat((X_train, X_dev), dim=0)
        y_train = torch.cat((y_train, y_dev), dim=0)
        train_desc = torch.cat((train_desc, dev_desc), dim=0)
        

    return (X_train, train_desc, y_train), (X_test, test_desc, y_test), (X_dev, dev_desc, y_dev)

