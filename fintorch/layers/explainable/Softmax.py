import torch.nn as nn
from fintorch.layers.explainable.ValueCaptureHook import ValueCaptureHook

class Softmax(nn.Softmax):
    def __init__(self, dim=None):
        super().__init__(dim=dim)
        self.hook = ValueCaptureHook()
        self.register_forward_hook(self.hook.hook_fn)

    def propagate(self, relevance):
        return relevance
