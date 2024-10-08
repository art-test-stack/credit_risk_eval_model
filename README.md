# Credit risk evaluation model with textual features

## Description

This repository is about the reimplementation of the result of the paper "Credit risk evaluation model with textual features from loan descriptions for P2P lending", [Zhang et al. (2020)](#references). It has been made within my Master Thesis topic research at NTNU (Uncertainty Quantification for Language Models in Safety-Critical Systems). The model specificities are given bellow.

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
| Loan amount               | Numerical (log)    | The total amount committed to the loan                                      | `log(loan_amnt)`                                                        |
| Term                      | Categorical        | Number of payments on the loan. Values can be either 36 or 60               | `term`                                                                    |
| Interest rate             | Numerical          | Nominal interest on the loan                                                | `int_rate`                                                                |
| Loan purpose              | Categorical        | Purpose of loan request, 13 purposes included                               | `purpose`                                                              |
| **Creditworthiness features** |                |                                                                             |                                                                           |
| FICO score                | Numerical          | Borrower’s FICO score at loan organization                                  | (`fico_range_high` + `fico_range_low`) /2   |
| Credit level              | Categorical        | Internal credit risk level assigned by the platform                         | `grade`                                                                  |
| Inquiries last 6 months   | Numerical          | The number of inquiries within the last 6 months                            | `inq_last_6mths`                                                         |
| Revolving utilization rate| Numerical          | The amount of credit the borrower is using relative to all available revolving credit | `revol_util`                                                 |
| Delinquency 2 years       | Numerical          | The number of delinquencies within the past 2 years                         | `delinq_2yrs`                                                            |
| Public record             | Numerical          | The number of derogatory public records                                     | `pub_rec`                                                                |
| Open credit lines         | Numerical          | The number of open credit lines                                             | `open_acc`                                                               |
| Revolving income ratio    | Numerical          | The ratio of revolving line to monthly income                               | `revol_bal / annual_inc / 12`[*]                        |
| Total account             | Numerical          | The total number of credit lines currently                                  | `total_acc`                                                              |
| Credit age                | Numerical          | The number of months from the time at which the borrower opened his or her first credit card to the loan requests | `issue_d - months_since_earliest_cr_line` |
| **Solvency features**     |                    |                                                                             |                                                                           |
| Annual income             | Numerical (log)    | The self-reported annual income provided by borrower                        | `log(annual_inc)`                                                        |
| Employ length             | Numerical          | Employment length in years                                                  | `emp_length` converted in months                                         |
| House ownership           | Categorical        | House ownership provided by the borrower                                    | `home_ownership`                                                        |
| Income verification       | Categorical        | The status of income verification. Verified, source verified, or not verified | `verification_status`                                                   |
| DTI                       | Numerical          | Debt-to-income ratio                                                        | `dti`                                                                    |
| **Description feature**   |                    |                                                                             |                                                                           |
| Description length        | Numerical          | The length of the loan description                                          | `desc.length()`                                                         |

[*]: [Malekipirbazari et al. (2015)](#references)
### Data selection 

The data selection has been done the closer to the one done in the original paper.

Hence, as mentioned in the article, I used the loans issued during the period of 2007–2014. After performing a data cleaning operation (discard loans with missing values and with a description length of less than
20 words), I have obtained 69,276 (70,488 in the paper) loan records, 10,151 (against 10,534) (14.65% against 14.94%) of which were ultimately placed in a default status.


#### Comparison Table of Summary of the length of loan description

|  **Paper samples** | | | | | | 
|-|-|--------------------|-|-|-|
| *Status* | Size  | Mean | Std | Q1 | Q2 | Q3|
| Paid     | 59954 | 56.69| 45.80 | 30 | 45 |63|
| Default  | 10534 | 55.78| 40.52 | 29 | 44 | 63|
| All      | 70488 | 56.56|40.76| 29 | 45 | 63|
| **Samples processed** | |  | | | | 
|         | Size  | Mean | Std | Q1 | Q2 | Q3|
| Paid    | 58742 | 60.96 | 60.95 | 32 | 47 | 64 |
| Default | 10151 | 61.52 | 58.24 | 21 | 46 | 64 |
| All     | 69276 | 61.04 | 54.64 | 31  | 47 | 64 |

#### Comparison of cross Table on Continuous Variables for LendingClub Data (from paper's and processed's data)

| Variable                | Default (Mean) | Default (Median) | Default (Std) | Paid off (Mean) | Paid off (Median) | Paid off (Std) | Pbc         |
|-------------------------|----------------|-------------------|---------------|-----------------|-------------------|----------------|-------------|
| Loan amount (Log)       | 9.444 / 9.447  | 9.556  / 9.575   | 0.651 / 0.648 | 9.346 / 9.341    | 9.393 / 9.393   | 0.652 / 0.654  | −0.053 / -0.057 |
| Interest rate           | 0.154 / 0.155  | 0.153 / 0.153    | 0.042 / 0.043 | 0.128 / 0.128   | 0.125 /  0.125   | 0.042 / 0.042 | −0.216 / -0.217 |
| Annual income (Log)     | 10.956 / 10.969 | 10.951/10954     | 0.499 / 0.491 | 11.046 / 11.056 | 11.035 / 11.051   | 0.513  / 0.509   | 0.062  / 0.061  |
| DTI                     | 0.173 / 0.172  | 0.173 / 0.172    | 0.075 / 0.074 | 0.159 / 0.158    | 0.157 / 0.156  | 0.075 / 0.074  | −0.064 / -0.066 |
| FICO score              | 695.500 / 696.006 | 687.000 / 692.000 | 28.238 / 28.278 | 706.960 / 707.796 | 702.000 / 702.000 | 33.667 / 33.747 | 0.123  / 0.125 |
| Delinquency in 2 years  | 0.190 / 0.211  | 0.000 / 0.000    | 0.490 / 0.660 | 0.170 / 0.189 | 0.000  / 0.000      | 0.470 / 0.603  | −0.012 / -0.013 |
| Open credit lines       | 10.870 / 10.817 | 10.000 / 10.000 | 4.760 / 4.713 | 10.580 / 10.531   | 10.000 / 10.000   | 4.615 / 4.559   | −0.022 / -0.022 |
| Inquiries last 6 months | 0.960 / 0.999  | 1.000 / 1.000  | 1.023 / 1.122  | 0.770 / 0.796  | 0.000 / 0.000   | 0.955 / 1.034   | −0.069 / -0.068 |
| Public records          | 0.100 / 0.088  | 0.000 / 0.000    | 0.320 / 0.331 | 0.080 / 0.70  | 0.000  / 0.000     | 0.294 / 0.292    | −0.020 / -0.022 |
| Revolving to income     | 2.940 / 2.935  | 2.499 / 2.496  | 2.289 / 2.290  | 2.736 / 2.732  | 2.297 / 2.294   | 2.205 / 2.199  | −0.033 / -0.033 |
| Revolving utilization   | 0.600 / 0.602  | 0.630 / 0.634    | 0.239 / 0.241 | 0.546 / 0.547  | 0.569 / 0.572      | 0.252 / 0.254 | −0.076 / -0.077 |
| Total account           | 23.860 / 23.681  | 22.000 / 22.000 | 11.132 /11.11  | 24.110 / 23.900  | 23.000 / 22.000  | 11.257 / 11.161 | 0.008 / 0.007 |
| Credit age              | 174.440 / 170.694  | 159.000 / 156.000 | 81.699 / 78.499 | 177.400 / 174.493 | 163.000 / 160.000 | 81.983 / 79.591 | 0.013 / 0.017  |
| Description length      | 55.770 / 61.520  | 44.000 / 46.000  | 40.523 / 58.236  | 56.690 / 60.960 | 45.000 / 47.000  | 40.799 / 53.995 | 0.008 / -0.004* |

On each cells the data is given like this: `{paper} / {processed}`. The results provided can be runned in [`read_csv.ipynb`](read_csv.ipynb).
Pbc means Point-biserial correlation.

### Preprocessing

#### Hard features
The first thing was the feature selection for the model. Hence the hard features (not textual features) are processed by one hot encoding them. Hence a splitting is done: 80% for training set, 10% for dev set and 10% for test set. A resampling strategy on the training set is done to address the data imbalance problem: I have performed an over-sampling strategy for the loans in default and an under-sampling strategy for the positive sample. Due to implementation issues, advanced techniques such SMOTE [Chawla et al. (2002)](#references) for oversampling or Tomek's link [Tomek (1975)](#references) for undersampling could not been processed here because of the textual features and that I have not found a way to track the ids. Because in the origial paper they do not mention a specific technique, I have decided to use both a random over sampling and under sampling [Brownlee (2021)](#references). 
Once the training set is balanced, I normalize the hard features by proceeding with standardization. Meanwhile, the textual features are processed as described in the [Textual features](#textual-features) section. Finally, the model is trained on both train and dev sets.

#### Textual features

The textual features preprocessing is basically processed in two parts. First, the [segmentation](#segmentation) and then converting the segmented words into [embeddings](#word-embedding).

##### Segmentation

According to the paper I used [CoreNLP](https://github.com/stanfordnlp/CoreNLP/tree/375f24338c09b22d1596440864bc074f32c0feb9) model from [Manning et al. (2014)](#references) for the segmentation of textual features. However, I understand that, from that time, the technology evolved and as been incorporated to the python package [`stanza`](https://github.com/stanfordnlp/stanza). So instead of using the Java server provided by the authors of the original paper of CoreNLP, I have chosen to use `stanza.Pipeline` which is more optimized for Python execution. Moreover, the authors of the paper that I tried to replicate do not mention which option (called annotations) of the `CoreNLP.Pipeline` they used. So I felt free to use some different types of them. I have also added an option to prune words which are included in the [`nltk.stopwords`](https://www.nltk.org/) set. The results are provided is the [Results](#results) section.

##### Word embedding

After tokenizing the words, I used a pre-trained word vectors model named [GloVe](https://github.com/stanfordnlp/GloVe/tree/a577eeeb8074f2c362fa7738143214eca9cb414f), from [Pennington et al. (2014)](#references) to create the embeddings of the tokens, as they did in the original paper. GloVe provide several pre-trained word vectors models which has for the embedding dimensions, only 3 sizes: 100, 200 and 300. As described in [Model](#the-model) section, and since my goal is to replicate the best model from the paper, the number of heads for the multi-head attention mechanism, from [Vaswani et al. (2017)](#references), is 8. Hence, I have chosen the 200 dimensions model since the article does not mention the dimension of the Transformer Encoder (TE) model. The implementation itself have been done with [`spacy`](https://spacy.io/), the most recent official package from Stanford for GloVe embeddings.

<!-- The embeddings are then averaged to get a single vector for each loan description. -->

## The model

The base of the model used is a Transformer Encoder (TE) model from [Vaswani et al. (2017)](#references). The textual features processed are then fed into the TE. As in the original paper, the TE is composed of 1 layers, with 8 heads for the multi-head attention mechanism, and feed-forward networks (FFN) has a size of 50 neurons and the activation function used is the rectified linear unit (ReLU) function. Then, as the TE is considered here as a feature exctractor, only the first sequence of the output of the TE is used. It's concatenated with the hard features and fed into a feed-forward neural network (FNN) with 2 hidden layers of 10 neurons each, separated by a ReLU activation function. The output layer is two neurons fed into a Softmax output. 

The model is trained with a binary cross-entropy loss function and the Adam optimizer. The model is trained with a batch size of 1024 and a learning rate of 0.0001. The model is trained on an early-stopping strategy. The model is evaluated on the test set using the ROC-AUC and G-MEAN metrics, developped in [Metrics](#metrics) section.

A dropout strategy of 0.3 (value not precised in the article) is used for the TE and after each layer of the FNN to avoid overfitting.


## Metrics


## Results

| Measures   | TE from the paper |  TE with stopwords | TE without stopwords |
|------------|-------------------|--------------------|--------------------|
| AUC (%)    | 70.30             | 65.74              | Na                 |
| G-MEAN (%) | 65.32             | 65.80              | Na                 |

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

1. Zhang, et al. (2020). "Credit risk evaluation model with textual features from loan descriptions for P2P lending." *Electronic Commerce Research and Applications*, 39, 100989. [pdf](https://doi.org/10.1016/j.elerap.2020.100989)

2. LendingClub. (n.d.). LendingClub loans data. Retrieved from [https://www.lendingclub.com/](https://www.lendingclub.com/)

3. Milad Malekipirbazari, Vural Aksakalli, "Risk assessment in social lending via random forests," *Expert Systems with Applications*, Volume 42, Issue 10, 2015, Pages 4621-4631, ISSN 0957-4174, [pdf](https://doi.org/10.1016/j.eswa.2015.02.001).

4. Nitesh V Chawla, Kevin W Bowyer, Lawrence O Hall, and W Philip Kegelmeyer. (2002). "SMOTE: Synthetic Minority Over-sampling Technique." *Journal of Artificial Intelligence Research*, 16, 321–357.

5. Ivan Tomek. (1976). "Two modifications of cnn." *IEEE Transactions on Systems, Man, and Cybernetics*, 6, 769–772.

6. Jason Brownlee. (2021). "Random Oversampling and Undersampling for Imbalanced Classification." Retrieved from [article](https://machinelearningmastery.com/random-oversampling-and-undersampling-for-imbalanced-classification/#:~:text=Random%20oversampling%20duplicates%20examples%20from,information%20invaluable%20to%20a%20model).

7. Manning, Christopher D., Mihai Surdeanu, John Bauer, Jenny Finkel, Steven J. Bethard, and David McClosky. (2014). "The Stanford CoreNLP Natural Language Processing Toolkit." *Proceedings of the 52nd Annual Meeting of the Association for Computational Linguistics: System Demonstrations*, pp. 55-60. [pdf](https://nlp.stanford.edu/pubs/StanfordCoreNlp2014.pdf).

8. Jeffrey Pennington, Richard Socher, and Christopher D. Manning. (2014). "GloVe: Global Vectors for Word Representation." [pdf](https://nlp.stanford.edu/pubs/glove.pdf)

9. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. (2017). "Attention is All You Need." *Advances in Neural Information Processing Systems*, 30, 5998-6008. [pdf](https://arxiv.org/abs/1706.03762)



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.