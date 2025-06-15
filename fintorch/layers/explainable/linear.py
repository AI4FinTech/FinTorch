import torch
import torch.nn as nn
import torch.nn.functional as F
from fintorch.layers.explainable.relevancepropagation import safe_divide_where
from fintorch.layers.explainable.ValueCaptureHook import ValueCaptureHook

class Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.hook = ValueCaptureHook()
        self.register_forward_hook(self.hook.hook_fn)

    def propagate(self, relevance):
        if hasattr(self.hook, 'X') and self.hook.X is not None:
            X = self.hook.X
            Z = F.linear(X, self.weight, self.bias)
            relevance = X * torch.autograd.grad(Z, X, safe_divide_where(relevance, Z), retain_graph=True)[0]
            return relevance
        else:
            return relevance
