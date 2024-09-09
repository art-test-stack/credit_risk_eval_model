import pandas as pd
from enum import Enum

class DataType(Enum):
    CATEGORICAL = "cat"
    NUMERICAL = "num"
    NUMERICAL_LOG = "num_log"


features = {
    # TARGET VARIABLE
    "loan_status": DataType.CATEGORICAL, # 

    # LOAN CHARACTERISTICS
    "loan_amnt": DataType.NUMERICAL_LOG,
    "term": DataType.CATEGORICAL, # 36, 60
    "int_rate": DataType.NUMERICAL,
    "purpose": DataType.CATEGORICAL, # "credit_card", "debt_consolidation", "educational", "major_purchase", "small_business", and "all_other"

    # CREDITWORTHINESS FEATURES
    # "fico": DataType.NUMERICAL, # fico_range_low, fico_range_high, last_fico_range_high, last_fico_range_low, sec_app_fico_range_low, sec_app_fico_range_high
    "total_il_high_credit_limit" : DataType.CATEGORICAL, # "credit_level": DataType.CATEGORICAL, # grade?
    "inq_last_6mths": DataType.NUMERICAL,
    "revol_util": DataType.NUMERICAL,
    "delinq_2yrs": DataType.NUMERICAL,
    "pub_rec": DataType.NUMERICAL,
    "open_acc": DataType.NUMERICAL,
    "revol_util": DataType.NUMERICAL, # "revolving_income_ratio": DataType.NUMERICAL,
    "total_acc": DataType.NUMERICAL,
    # "credit_age": DataType.NUMERICAL,

    # SOLVENCY FEATURES
    "annual_inc": DataType.NUMERICAL_LOG, # log.annual.inc
    "emp_length": DataType.NUMERICAL,
    "home_ownership": DataType.CATEGORICAL, # RENT, OWN, MORTGAGE, OTHER
    "dti": DataType.NUMERICAL,

    # DESCRIPTION FEATURE
    "desc_len": DataType.NUMERICAL
}