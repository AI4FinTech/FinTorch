import torch.nn as nn
from fintorch.layers.explainable.ValueCaptureHook import ValueCaptureHook

class ReLU(nn.ReLU):
    def __init__(self, inplace: bool = False):
        super().__init__(inplace=inplace)
        self.hook = ValueCaptureHook()
        self.register_forward_hook(self.hook.hook_fn)

    def propagate(self, relevance):
        return relevance
