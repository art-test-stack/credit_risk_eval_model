
from pathlib import Path
from matplotlib import pyplot as plt

from typing import Dict
import numpy as np

def plot_losses(self, hist: Dict[str,np.ndarray], display: bool = False):
    folder = Path('rsc/').joinpath('losses.png')
    plt.ioff()

    fig = plt.figure()
    plt.plot(range(len(1, hist["train_loss"] + 1)), hist["train_loss"], label='Training Loss')
    plt.plot(range(len(1, hist["test_loss"] + 1)), hist["test_loss"], label='Testing Loss')
    plt.legend()
    plt.savefig(folder)
    plt.close(fig)
    if display:
        plt.show()
