# Credit risk evaluation model with textual features

## Description

This repository is about the reimplementation of the result of the paper "Credit risk evaluation model with textual features from loan descriptions for P2P lending", [Zhang et al., [1]](#references). It has been made within my Master Thesis topic research at NTNU (Uncertainty Quantification for Language Models in Safety-Critical Systems). The model specificities are given bellow.

> [!NOTE]  
> Not everything has been made exactly like in the paper. Sometimes because the technologies improved and sometimes by lack of knowledge.


## Dataset

The dataset is from [LendingClub, [2]](https://www.lendingclub.com/) loans data from the American market.

Since LendingClub does not provide anymore this data, the dataset used have been taken from [Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club?resource=download). 

In this section the dataset used will be presented, the feature selection -as in the original paper- will be described, and it characteristics will be compared to the one used in the paper.

### Definitions and Descriptions of Variables from LendingClub

In this section the variables names and its description are given. There are few variables that has been computed from existing variables in the dataset, but where not in the orignal dataset. The code for this process can be found in [`src/data/features.py`](src/data/features.py).


| Variable Name             | Data Type          | Description                                                                 | Code Variable Name / Computation                                           |
|---------------------------|--------------------|-----------------------------------------------------------------------------|---------------------------------------------------------------------------|
| **Target Variable**       |                    |                                                                             |                                                                           |
| Loan status               | Categorical        | Whether the borrower has or has not paid the loan in full                   | `loan_status`                                                            |
| **Loan characteristics**  |                    |                                                                             |                                                                           |
| Loan amount               | Numerical (log)    | The total amount committed to the loan                                      | `log(loan_amnt)`  |
| Term                      | Categorical        | Number of payments on the loan. Values can be either 36 or 60               | `term`            |
| Interest rate             | Numerical          | Nominal interest on the loan                                                | `int_rate`        |
| Loan purpose              | Categorical        | Purpose of loan request, 13 purposes included                               | `purpose`         |
| **Creditworthiness features** |                |                                                                             |                                                                           |
| FICO score                | Numerical          | Borrower’s FICO score at loan organization                                  | (`fico_range_high` + `fico_range_low`) /2   |
| Credit level              | Categorical        | Internal credit risk level assigned by the platform                         | `grade`                                                                  |
| Inquiries last 6 months   | Numerical          | The number of inquiries within the last 6 months                            | `inq_last_6mths`                                                         |
| Revolving utilization rate| Numerical          | The amount of credit the borrower is using relative to all available revolving credit | `revol_util`                                                             |
| Delinquency 2 years       | Numerical          | The number of delinquencies within the past 2 years                         | `delinq_2yrs`                                                            |
| Public record             | Numerical          | The number of derogatory public records                                     | `pub_rec`                                                                |
| Open credit lines         | Numerical          | The number of open credit lines                                             | `open_acc`                                                               |
| Revolving income ratio    | Numerical          | The ratio of revolving line to monthly income                               | `revol_bal / annual_inc / 12`[[3]](#references)                                                |
| Total account             | Numerical          | The total number of credit lines currently                                  | `total_acc`                                                              |
| Credit age                | Numerical          | The number of months from the time at which the borrower opened his or her first credit card to the loan requests | `issue_d - months_since_earliest_cr_line`                                          |
| **Solvency features**     |                    |                                                                             |                                                                           |
| Annual income             | Numerical (log)    | The self-reported annual income provided by borrower                        | `log(annual_inc)`                                                        |
| Employ length             | Numerical          | Employment length in years                                                  | `emp_length` converted in months                                                            |
| House ownership           | Categorical        | House ownership provided by the borrower                                    | `home_ownership`                                                        |
| Income verification       | Categorical        | The status of income verification. Verified, source verified, or not verified | `verification_status`                                                   |
| DTI                       | Numerical          | Debt-to-income ratio                                                        | `dti`                                                                    |
| **Description feature**   |                    |                                                                             |                                                                           |
| Description length        | Numerical          | The length of the loan description                                          | `desc.length()`                                                         |

### Data selection 

The data selection has been done the closer to the one done in the original paper.

Hence, as mentioned in the article, I used the loans issued during the period of 2007–2014. After performing a data cleaning operation (discard loans with missing values and with a description length of less than
20 words), I have obtained 69,276 (70,488 in the paper) loan records, 10,151 (against 10,534) (14.65% against 14.94%) of which were ultimately placed in a default status.


|  **Paper samples** | | | | | | 
|-|-|--------------------|-|-|-|
| *Status* | Size  | Mean | Std | Q1 | Q2 | Q3|
| Paid                     | 59954 | 56.69| 45.80 | 30 | 45 |63|
| Default                  | 10534 | 55.78|40.52| 29 | 44 | 63|
| All                      | 70488 | 56.56|40.76| 29 | 45 | 63|
| **Samples processed** | |  | | | | 
| | Size  | Mean | Std | Q1 | Q2 | Q3|
| Paid                     | 58742 |  |     |    |    |   |
| Default                  | 10534 |  |     |    |    |  |
| All                      | 69276 |  |     |    |    |  |

### Comparison of cross Table on Continuous Variables for LendingClub Data (from paper's and processed's data)

| Variable                | Default (Mean) | Default (Median) | Default (Std) | Paid off (Mean) | Paid off (Median) | Paid off (Std) | Pbc         |
|-------------------------|----------------|------------------|---------------|-----------------|-------------------|----------------|-------------|
| Loan amount (Log)       | 9.444 / 9.45   | 9.556  / 9.575   | 0.651 / 0.648 | 9.346 /         | 9.393             | 0.652          | −0.053 / -0.057 |
| Interest rate           | 0.154 / 0.155  | 0.153 / 0.153    | 0.042 / 0.043 | 0.128           | 0.125             | 0.042          | −0.216 / -0.217 |
| Annual income (Log)     | 10.956 /10.969 | 10.951/10954     | 0.499 / 0.491 | 11.046          | 11.035            | 0.513          | 0.062  / 0.061  |
| DTI                     | 0.173 / 0.172  | 0.173 / 0.172    | 0.075 / 0.074 | 0.159           | 0.157             | 0.075          | −0.064 / -0.066 |
| FICO score              | 695.500 / 696.006| 687.000 / 692.000 | 28.238 / 28.278 | 706.960         | 702.000           | 33.667         | 0.123  / 0.125 |
| Delinquency in 2 years  | 0.190 / 0.211  | 0.000 / 0.000    | 0.490 / 0.660 | 0.170           | 0.000             | 0.470          | −0.012 / -0.013 |
| Open credit lines       | 10.870 / 10.817 | 10.000 / 10.000 | 4.760 / 4.713 | 10.580          | 10.000            | 4.615          | −0.022 / -0.022 |
| Inquiries last 6 months | 0.960 / 0.999  | 1.000 / 1.000  | 1.023 / 1.122  | 0.770           | 0.000             | 0.955          | −0.069 / -0.068 |
| Public records          | 0.100 / 0.088  | 0.000 / 0.000    | 0.320 / 0.331 | 0.080           | 0.000             | 0.294          | −0.020 / -0.022 |
| Revolving to income     | 2.940 / 2.935  | 2.499 /   | 2.289         | 2.736           | 2.297             | 2.205          | −0.033 / -0.033 |
| Revolving utilization   | 0.600          | 0.630            | 0.239         | 0.546           | 0.569             | 0.252          | −0.076 / -0.077 |
| Total account           | 23.860         | 22.000           | 11.132        | 24.110          | 23.000            | 11.257         | 0.008 / 0.007 |
| Credit age              | 174.440        | 159.000          | 81.699        | 177.400         | 163.000           | 81.983         | 0.013 / 0.017  |
| Description length      | 55.770         | 44.000           | 40.523        | 56.690          | 45.000            | 40.799         | 0.008 / -0.004* |

On each cells the data is given like this: `{paper} / {processed}`.

### Segmentation

[CoreNLP](https://github.com/stanfordnlp/CoreNLP/tree/375f24338c09b22d1596440864bc074f32c0feb9)

### Word embedding

[GloVe](https://github.com/stanfordnlp/GloVe/tree/a577eeeb8074f2c362fa7738143214eca9cb414f)

## The model

## Results

| Measures --| TE from the paper |  TE from this work |
|------------|-------------------|--------------------|
| AUC (%)    | 70.30             | Na                 |
| G-MEAN (%) | 65.32             | Na                 |

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/arthurtestard/credit_risk_eval_model.git
    ```
2. Navigate to the project directory:
    ```sh
    cd credit_risk_eval_model
    ```
3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

<!-- ## Usage

1. Preprocess the data:
    ```sh
    python preprocess.py
    ```
2. Train the model:
    ```sh
    python train.py
    ```
3. Evaluate the model:
    ```sh
    python evaluate.py
    ``` -->



## References

1. Zhang, et al. (2020). "Credit risk evaluation model with textual features from loan descriptions for P2P lending." *Electronic Commerce Research and Applications*, 39, 100989. [doi.org/10.1016/j.elerap.2020.100989](https://doi.org/10.1016/j.elerap.2020.100989)

2. LendingClub. (n.d.). LendingClub loans data. Retrieved from [https://www.lendingclub.com/](https://www.lendingclub.com/)

3. Milad Malekipirbazari, Vural Aksakalli, "Risk assessment in social lending via random forests," *Expert Systems with Applications*, Volume 42, Issue 10, 2015, Pages 4621-4631, ISSN 0957-4174, [doi.org/10.1016/j.eswa.2015.02.001](https://doi.org/10.1016/j.eswa.2015.02.001). [Link to article](https://www.sciencedirect.com/science/article/pii/S0957417415000937)


<!-- ## How to Cite

If you use this repository or any part of it in your research, please cite the original paper as follows:

```
@article{zhang2020credit,
    title={Credit risk evaluation model with textual features from loan descriptions for P2P lending},
    author={Zhang, et al.},
    journal={Electronic Commerce Research and Applications},
    volume={39},
    pages={100989},
    year={2020},
    publisher={Elsevier}
}
``` -->
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.