import pandas as pd
from enum import Enum

class DataType(Enum):
    CATEGORICAL = "cat"
    NUMERICAL = "num"
    NUMERICAL_LOG = "num_log"


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

features = {
    # TARGET VARIABLE
    "loan_status": DataType.CATEGORICAL, # 

    # LOAN CHARACTERISTICS
    "log_loan_amnt": DataType.NUMERICAL_LOG,
    "term": DataType.CATEGORICAL, # 36, 60
    "int_rate": DataType.NUMERICAL,
    "purpose": DataType.CATEGORICAL, # "credit_card", "debt_consolidation", "educational", "major_purchase", "small_business", and "all_other"

    # CREDITWORTHINESS FEATURES
    "fico": DataType.NUMERICAL,
    # "fico_range_low": DataType.NUMERICAL, # temporary
    # "fico_range_high": DataType.NUMERICAL, # temporary
    "grade": DataType.CATEGORICAL, # = "credit_level": DataType.CATEGORICAL,
    "inq_last_6mths": DataType.NUMERICAL,
    "revol_util": DataType.NUMERICAL,
    "delinq_2yrs": DataType.NUMERICAL,
    "pub_rec": DataType.NUMERICAL,
    "open_acc": DataType.NUMERICAL,
    "revol_inc_rat": DataType.NUMERICAL,
    "total_acc": DataType.NUMERICAL,
    "credit_age": DataType.NUMERICAL, # earliest_cr_line - issue_d

    # SOLVENCY FEATURES
    "log_annual_inc": DataType.NUMERICAL_LOG, # log.annual.inc
    "emp_length": DataType.NUMERICAL,
    "home_ownership": DataType.CATEGORICAL, # RENT, OWN, MORTGAGE, OTHER
    "verification_status": DataType.CATEGORICAL,
    "dti": DataType.NUMERICAL,

    # DESCRIPTION FEATURE
    "desc_len": DataType.NUMERICAL
}