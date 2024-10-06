import torch
from torch import nn

from torcheval.metrics import AUC, BinaryAUROC
from torcheval.metrics.functional import multiclass_recall, multiclass_precision

class GMean:
    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        # self.sensitivities = []
        # self.specificities = []
        self.targets = []
        self.inputs = []
    
    def update(self, input, target):
        self.targets.extend(target)
        self.inputs.extend(input)
    
    def compute(self):      
        sensitivity = multiclass_recall(self.inputs, self.targets)       # TP / TP + FN
        specificity = multiclass_precision(self.inputs, self.targets, average='macro', num_classes=2, zero_division='warn')  # TN / TN + FP
        
        # self.sensitivities.append(sensitivity)
        # self.specificities.append(specificity)

        g_mean = torch.sqrt(sensitivity * specificity)
        return g_mean

    def __call__(self, input, target):
        self.update(input, target)
        g_mean = self.compute()
        return g_mean


class ROCAUC(BinaryAUROC):
    def __init__(self) -> None:
        super().__init__()
    
    def __call__(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        inp = torch.argmax(input, dim=1)
        tgt = torch.argmax(target, dim=1)
        self.update(inp, tgt)
    
