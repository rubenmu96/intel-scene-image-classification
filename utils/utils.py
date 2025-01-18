import torch
import os
import numpy as np

# move to metrics.py?
class MetricCalculator:
    """
    Verify these calculations against some torch metrics?
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()
        
    def update(self, outputs, targets):
        _, preds = torch.max(outputs, 1)

        for c in range(self.num_classes):
            pred_eq_c = preds == c
            target_eq_c = targets == c
            self.tp[c] += (pred_eq_c & target_eq_c).sum().item()
            self.fp[c] += (pred_eq_c & ~target_eq_c).sum().item()
            self.fn[c] += (~pred_eq_c & target_eq_c).sum().item()
            self.tn[c] += (~pred_eq_c & ~target_eq_c).sum().item()
    
    def compute_metrics(self):
        accuracy = (self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn)
        precision = torch.where(self.tp + self.fp > 0, self.tp / (self.tp + self.fp), torch.zeros_like(self.tp))
        recall = torch.where(self.tp + self.fn > 0, self.tp / (self.tp + self.fn), torch.zeros_like(self.tp))
        f1 = torch.where(precision + recall > 0, 2 * (precision * recall) / (precision + recall), torch.zeros_like(precision))

        return {
            'accuracy': accuracy.mean().item(),
            'macro_precision': precision.mean().item(),
            'macro_recall': recall.mean().item(),
            'macro_f1': f1.mean().item(),
        }
    
    def reset(self):
        self.tp = torch.zeros(self.num_classes)
        self.fp = torch.zeros(self.num_classes)
        self.fn = torch.zeros(self.num_classes)
        self.tn = torch.zeros(self.num_classes)


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True