from src.pipeline import pipeline
from utils import get_device, PREPROCESSED_FILE, LOANS_FILE

from pathlib import Path


if __name__ == "__main__":
    use_sw = False
    do_preprocessing=True

    if use_sw:
        preprocessed_file = PREPROCESSED_FILE
    else:
        preprocessed_file = Path("data/preprocessed_no_sw/") 
    
    pipeline(
        loans_file=LOANS_FILE,
        device=get_device(),
        do_preprocessing=do_preprocessing,
        epochs=500,
        preprocessed_data_file=preprocessed_file
    )
    

    