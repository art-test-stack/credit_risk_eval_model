import pandas as pd
from enum import Enum
import math


class DataType(Enum):
    CATEGORICAL = "cat"
    NUMERICAL = "num"
    NUMERICAL_LOG = "num_log"
    TEXTUAL = "text"

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

features_pre_processing = {
    # TARGET VARIABLE
    "loan_status": DataType.CATEGORICAL, # 

    # LOAN CHARACTERISTICS
    "loan_amnt": DataType.NUMERICAL_LOG,
    "term": DataType.CATEGORICAL, # 36, 60
    "int_rate": DataType.NUMERICAL,
    "purpose": DataType.CATEGORICAL, # "credit_card", "debt_consolidation", "educational", "major_purchase", "small_business", and "all_other"

    # CREDITWORTHINESS FEATURES
    # "fico": DataType.NUMERICAL, 
    "fico_range_low": DataType.NUMERICAL, 
    "fico_range_high": DataType.NUMERICAL, 
    # "credit_level": DataType.CATEGORICAL, # grade?
        # "total_il_high_credit_limit" : DataType.CATEGORICAL, # 43389 samples with nan filter on it
        # "tot_hi_cred_lim": DataType.NUMERICAL, # 43 389 samples with nan filter on it
    "grade": DataType.CATEGORICAL,
    "inq_last_6mths": DataType.NUMERICAL,
    "revol_util": DataType.NUMERICAL,
    "delinq_2yrs": DataType.NUMERICAL,
    "pub_rec": DataType.NUMERICAL,
    "open_acc": DataType.NUMERICAL,
    "revol_bal": DataType.NUMERICAL, # => "revolving_income_ratio": DataType.NUMERICAL, 
    "total_acc": DataType.NUMERICAL,
    "earliest_cr_line": "string",
    "issue_d": "string",

    # SOLVENCY FEATURES
    "annual_inc": DataType.NUMERICAL_LOG, # log.annual.inc
    "emp_length": DataType.NUMERICAL,
    "home_ownership": DataType.CATEGORICAL, # RENT, OWN, MORTGAGE, OTHER
    "verification_status": DataType.CATEGORICAL,
    "dti": DataType.NUMERICAL,

    # DESCRIPTION FEATURE
    "desc_len": DataType.NUMERICAL
}

# TARGET VARIABLE
target_variable = {"loan_status": DataType.CATEGORICAL,}

# LOAN CHARACTERISTICS
loan_charac_features = {
    "log_loan_amnt": DataType.NUMERICAL_LOG,
    "term": DataType.CATEGORICAL, # 36, 60
    "int_rate": DataType.NUMERICAL,
    "purpose": DataType.CATEGORICAL, # "credit_card", "debt_consolidation", "educational", "major_purchase", "small_business", and "all_other"
}

# CREDITWORTHINESS FEATURES
credit_worthiness_features = {
    "fico": DataType.NUMERICAL,
    "grade": DataType.CATEGORICAL, # = "credit_level": DataType.CATEGORICAL,
    "inq_last_6mths": DataType.NUMERICAL,
    "revol_util": DataType.NUMERICAL,
    "delinq_2yrs": DataType.NUMERICAL,
    "pub_rec": DataType.NUMERICAL,
    "open_acc": DataType.NUMERICAL,
    "revol_inc_rat": DataType.NUMERICAL,
    "total_acc": DataType.NUMERICAL,
    "credit_age": DataType.NUMERICAL, # earliest_cr_line - issue_d
}

# SOLVENCY FEATURES
solvency_features = {
    "log_annual_inc": DataType.NUMERICAL_LOG, # log.annual.inc
    "emp_length": DataType.NUMERICAL,
    "home_ownership": DataType.CATEGORICAL, # RENT, OWN, MORTGAGE, OTHER
    "verification_status": DataType.CATEGORICAL,
    "dti": DataType.NUMERICAL,
}

# DESCRIPTION FEATURE
desc_features = {
    "desc": str,
    "desc_len": DataType.NUMERICAL
}

features = target_variable | loan_charac_features | credit_worthiness_features | solvency_features | desc_features
features_numerical = [ ft for ft, dtype in features.items() if dtype == DataType.NUMERICAL or dtype == DataType.NUMERICAL_LOG ]

def select_features(df: pd.DataFrame, one_hot_output: bool = True) -> pd.DataFrame:
    raw = df.copy()

    raw["desc_len"] = raw.dropna(subset="desc")["desc"].apply(lambda x: len(str(x).split()))
    raw = raw.dropna(subset=list(features_pre_processing.keys()))

    raw = raw[raw["desc_len"] > 20]

    # COMPUTE fico
    raw["fico"] = (raw["fico_range_high"] + raw["fico_range_low"]) / 2 # or raw["fico_range_high"] 


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

    # CONVERT EMP LENGTH
    def convert_emp_length(emp_len: str) -> int:
        for k in range(1, 11):
            if str(k) in emp_len:
                return k
    raw["emp_length"] = raw["emp_length"].apply(convert_emp_length)

    # Select only interesting features
    res = raw[list(features.keys())]
    
    if one_hot_output:
        res = res[res["loan_status"].apply(lambda x: x in ["Fully Paid", "Charged Off"])]
        res["loan_status"] = res["loan_status"].apply(lambda x: 1 if x == "Fully Paid" else 0)

    return res
