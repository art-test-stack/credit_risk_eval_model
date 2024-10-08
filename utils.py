import torch
from pathlib import Path

RANDOM_STATE = 42
LOANS_FILE = Path("data/accepted_2007_to_2018Q4.csv")
PREPROCESSED_FILE = Path("data/preprocessed/") # Path("data/preprocessed_no_sw/") 
MODEL_DIR = Path("model/")
GLOVE_MODEL = Path("model/glove.6B/glove.6B.200d.txt")

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available. Using GPU.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        torch.backends.mps.set_per_process_memory_fraction(0.5)
        print("MPS is available. Using Apple Silicon GPU.")
    else:
        device = torch.device("cpu")
        print("CUDA and MPS are not available. Using CPU.")
    return device

