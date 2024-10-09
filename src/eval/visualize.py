
from pathlib import Path
from matplotlib import pyplot as plt

from typing import Dict
import numpy as np


def plot_losses(hist: Dict[str,np.ndarray], display: bool = False):

    folder = Path('rsc/').joinpath(f"losses_{hist["model_name"][:-3]}.png")
    plt.ioff()

    fig = plt.figure()
    plt.plot(range(1, len(hist["train_loss"]) + 1), hist["train_loss"], label='Training Loss')
    plt.plot(range(1, len(hist["test_loss"]) + 1), hist["test_loss"], label='Testing Loss')
    plt.legend()
    plt.savefig(folder)
    plt.close(fig)
    if display:
        plt.show()
