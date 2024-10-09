from src.pipeline import pipeline
from src.model.model import CREModel
from src.data.stanfordnlp import StanfordNLP

from utils import get_device, PREPROCESSED_FILE, LOANS_FILE, RANDOM_STATE, GLOVE_MODEL

import torch 
from pathlib import Path

import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Train a model to predict loan status from textual features",
        epilog="Enjoy the program! :)",
    )
    parser.add_argument(
        "-sw",
        "--no_stopwords",
        help="Use stop words",
        action="store_true",
    )
    parser.add_argument(
        "-sp",
        "--skip_preprocessing",
        help="Skip preprocessing",
        action="store_true",
    )
    parser.add_argument(
        "-bd",
        "--not_balance_dataset",
        help="Balance training set",
        action="store_true",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        help="Number of epochs to train the model",
        type=int,
        default=500,
    )
    parser.add_argument(
        "-d",
        "--dropout",
        help="Dropout rate",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        help="Batch size",
        type=int,
        default=1024,   
    )
    parser.add_argument(
        "-es",
        "--early_stopping_patience",
        help="Early stopping patience",
        type=int,
        default=100,
    )
    parser.add_argument(
        "-ed",
        "--early_stopping_min_delta",
        help="Early stopping min delta",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "-v",
        "--not_verbose",
        help="Disable verbose",
        action="store_true",
    )
    parser.add_argument(
        "-n",
        "--model_name",
        type=str,
        default="model.pt",
    )

    args = parser.parse_args()

    use_sw = not args.no_stopwords
    balance_training_set = not args.not_balance_dataset
    skip_preprocessing = args.skip_preprocessing
    epochs = args.epochs
    dropout = args.dropout
    batch_size = args.batch_size
    early_stopping_patience = args.early_stopping_patience
    early_stopping_min_delta = args.early_stopping_min_delta
    verbose = not args.not_verbose


    if use_sw and balance_training_set:
        preprocessed_file = PREPROCESSED_FILE
    elif use_sw and not balance_training_set:
        preprocessed_file = Path("data/preprocessed_not_b/")
    elif not use_sw and balance_training_set:
        preprocessed_file = Path("data/preprocessed_no_sw/") 
    else: 
        preprocessed_file = Path("data/preprocessed_no_sw_not_b/") 
    
    if not preprocessed_file.exists():
        preprocessed_file.mkdir(parents=True, exist_ok=True)
    
    model_name = args.model_name if args.model_name[-3:]==".pt" else args.model_name + ".pt"

    assert 0 <= dropout < 1, "Dropout rate must be in [0, 1)"
    assert epochs > 0, "Number of epochs must be positive"
    assert batch_size > 0, "Batch size must be positive"
    assert early_stopping_patience > 0, "Early stopping patience must be positive"
    assert early_stopping_min_delta >= 0, "Early stopping min delta must be positive or equal to zero"
    assert preprocessed_file.exists(), f"Preprocessed folder file does not exist, please create: '{preprocessed_file}'"
    if skip_preprocessing:
        assert preprocessed_file.joinpath("X_train.pt").exists(), "No preprocessed data found, please preprocess the data first by setting do_preprocessing to True"
    assert LOANS_FILE.exists(), "Loans file does not exist"
    assert GLOVE_MODEL.exists(), "Glove model file does not exist, please download it following the instructions in model/glove.6B/README"

    torch.manual_seed(RANDOM_STATE)
    pipeline(
        model=CREModel(name=model_name, dropout=dropout),
        nlp_model=StanfordNLP(stop_words=use_sw, verbose=verbose),
        loans_file=LOANS_FILE,
        device=get_device(),
        do_preprocessing=not skip_preprocessing,
        balance_training_set=balance_training_set,
        epochs=epochs,
        batch_size=batch_size,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        preprocessed_data_file=preprocessed_file,
        verbose=verbose
    )
    

    