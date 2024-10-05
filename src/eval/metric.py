import torch
from torch import nn

from torcheval.metrics import AUC, BinaryAUROC
from torcheval.metrics.functional import multiclass_recall, multiclass_precision

class GMean(nn.Module):
    def __init__(self):
        super().__init__()
        self.sensitivities = []
        self.specificities = []

    def forward(self, target, pred):
        sensitivity = multiclass_recall(pred, target)       # TP / TP + FN
        specificity = multiclass_precision(pred, target)    # TP / TP + FP
        self.sensitivities.append(sensitivity)
        self.specificities.append(specificity)
        
        g_mean = torch.sqrt(sensitivity * specificity)
        return g_mean


# class AUC(AUC):
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)


class ROCAUC(BinaryAUROC):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def __call__(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self.reset()
        self.update(input, target)
        res = self.compute()
        return res
    
