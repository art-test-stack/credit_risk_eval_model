from src.data.preprocess import preprocess_data
from src.data.corenlp import CoreNLP
from src.data.glove import GloVe

from src.model.model import CREModel

from src.eval.metric import GMean, AUC

import pandas as pd
from typing import Any
from pathlib import Path


def pipeline(
        loans_file: Path | str = Path("data/accepted_2007_to_2018Q4.csv"),
        model: CREModel = CREModel(),
        seg_model: CoreNLP = CoreNLP(),
        emb_model: GloVe = GloVe(),
    ) -> Any:
    
    train_set, test_set, _ = preprocess_data(loans_file, seg_model, emb_model)

    X_train, train_desc, y_train = train_set
    X_test, test_desc, y_test = test_set

    # TODO:
    # MODEL TRAINING
    # TO device

    # EVALUATION


