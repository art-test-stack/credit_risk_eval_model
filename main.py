from src.pipeline.pipeline import pipeline
from utils import get_device

if __name__ == "__main__":
    pipeline(
        device=get_device(),
        do_preprocessing=False,
        epochs=3000
    )
    

    