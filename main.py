from src.pipeline import pipeline
from src.model.model import CREModel
from src.data.stanfordnlp import StanfordNLP

from utils import get_device, PREPROCESSED_FILE, LOANS_FILE

from pathlib import Path


if __name__ == "__main__":
    use_sw = True
    do_preprocessing = False

    if use_sw:
        preprocessed_file = PREPROCESSED_FILE
    else:
        preprocessed_file = Path("data/preprocessed_no_sw/") 
    
    pipeline(
        loans_file=LOANS_FILE,
        model=CREModel(dropout=0.3),
        nlp_model=StanfordNLP(stop_words=use_sw),
        device=get_device(),
        do_preprocessing=do_preprocessing,
        epochs=500,
        preprocessed_data_file=preprocessed_file
    )
    

    