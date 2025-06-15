import torch.nn as nn
from fintorch.layers.explainable.ValueCaptureHook import ValueCaptureHook

class Dropout(nn.Dropout):
    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__(p=p, inplace=inplace)
        self.hook = ValueCaptureHook()
        self.register_forward_hook(self.hook.hook_fn)

    def propagate(self, relevance):
        return relevance
