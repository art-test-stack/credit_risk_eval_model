import torch
from torch import nn

from torcheval.metrics import AUC, BinaryAUROC, BinaryConfusionMatrix
from torcheval.metrics.functional import multiclass_recall, multiclass_precision

class GMean:
    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        # self.sensitivities = []
        # self.specificities = []
        self.bcm = BinaryConfusionMatrix()
        self.targets = []
        self.inputs = []
    
    def update(self, input: torch.Tensor, target: torch.Tensor):
        inp = torch.argmax(input, dim=1)
        tgt = target
        self.inputs = inp if isinstance(self.inputs, list) else torch.cat([self.inputs, inp])
        self.targets = tgt if isinstance(self.targets, list) else torch.cat([self.targets, tgt])

    def compute(self):      
    #     sensitivity = multiclass_recall(self.inputs, self.targets)       # TP / TP + FN
    #     specificity = multiclass_precision(self.inputs, self.targets, average='macro', num_classes=2)  # TN / TN + FP
        
    #     # self.sensitivities.append(sensitivity)
    #     # self.specificities.append(specificity)

    #     g_mean = torch.sqrt(sensitivity * specificity)
    #     return g_mean

    
        self.bcm.update(self.inputs, self.targets)
        confusion_matrix = self.bcm.compute()

        TN, FP, FN, TP = confusion_matrix.flatten().tolist()

        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0    # TP / TP + FN
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0    # TN / TN + FP

        return torch.sqrt(torch.tensor(sensitivity * specificity)) #.item()

    def __call__(self, input, target):
        self.update(input, target)
        g_mean = self.compute()
        return g_mean


class ROCAUC(BinaryAUROC):
    def __init__(self) -> None:
        super().__init__()
    
    def __call__(self, input: torch.Tensor, target: torch.Tensor) -> None:
        inp = torch.argmax(input, dim=1)
        tgt = target
        self.update(inp, tgt)
    
