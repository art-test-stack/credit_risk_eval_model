import torch

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