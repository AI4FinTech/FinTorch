import torch

class ValueCaptureHook:
    def __init__(self):
        self.X = None
        self.Y = None
        self.handle = None

    def hook_fn(self, module, input, output):
        tensor_inputs = [arg for arg in input if torch.is_tensor(arg)]

        self.X = []
        for t in tensor_inputs:
            self.X.append(t.detach().requires_grad_(True))

        if len(self.X) == 1:
            self.X = self.X[0]

        self.Y = output

    def register_to(self, module: torch.nn.Module):
        self.handle = module.register_forward_hook(self.hook_fn)

    def remove(self):
        if self.handle:
            self.handle.remove()
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.remove()
