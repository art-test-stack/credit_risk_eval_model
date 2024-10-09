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
        "--use_sw",
        help="Use stop words",
        action="store_true",
    )
    parser.add_argument(
        "-no_pp",
        "--do_preprocessing",
        help="Do data preprocessing",
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
        "--verbose",
        help="Verbose",
        action="store_true",
    )
    parser.add_argument(
        "-n",
        "--model_name",
        type=str,
        required=False,
    )

    args = parser.parse_args()

    use_sw = args.use_sw
    do_preprocessing = args.do_preprocessing
    epochs = args.epochs
    dropout = args.dropout
    batch_size = args.batch_size
    early_stopping_patience = args.early_stopping_patience
    early_stopping_min_delta = args.early_stopping_min_delta
    verbose = args.verbose

    if use_sw:
        preprocessed_file = PREPROCESSED_FILE
    else:
        preprocessed_file = Path("data/preprocessed_no_sw/") 
    
    model_name = (args.model_name if args.model_name[-3:]==".pt" else args.model_name + ".pt") if args.model_name else "model.pt"

    assert 0 <= dropout < 1, "Dropout rate must be in [0, 1)"
    assert epochs > 0, "Number of epochs must be positive"
    assert batch_size > 0, "Batch size must be positive"
    assert early_stopping_patience > 0, "Early stopping patience must be positive"
    assert early_stopping_min_delta > 0, "Early stopping min delta must be positive"
    assert preprocessed_file.exists(), "Preprocessed folder file does not exist"
    if not do_preprocessing:
        assert preprocessed_file.joinpath("X_train.pt").exists(), "No preprocessed data found, please preprocess the data first by setting do_preprocessing to True"
    assert LOANS_FILE.exists(), "Loans file does not exist"
    assert GLOVE_MODEL.exists(), "Glove model file does not exist, please download it following the instructions in model/glove.6B/README"

    torch.manual_seed(RANDOM_STATE)
    pipeline(
        model=CREModel(name=model_name, dropout=dropout),
        nlp_model=StanfordNLP(stop_words=use_sw, verbose=verbose),
        loans_file=LOANS_FILE,
        device=get_device(),
        do_preprocessing=do_preprocessing,
        epochs=epochs,
        batch_size=batch_size,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        preprocessed_data_file=preprocessed_file,
        verbose=verbose
    )
    

    