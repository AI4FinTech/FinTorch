import torch.nn as nn
import torch.nn.functional as F
import torch

from .ValueCaptureHook import ValueCaptureHook

def safe_divide_where(a, b):
  """
  A safe division function using torch.where.

  Args:
    a: The numerator tensor.
    b: The denominator tensor.

  Returns:
    The result of a / b, with 0 where b was 0.
  """
  # Create a tensor of zeros with the same shape and type as 'a'
  zeros = torch.zeros_like(a)

  # Where b is not zero, compute a/b. Otherwise, use the 'zeros' tensor.
  return torch.where(b != 0, a / b, zeros)


class LinearRelevancePropagation(nn.Module):
    def __init__(self):
        super(LinearRelevancePropagation, self).__init__()
        self.hook = ValueCaptureHook()
        self.register_forward_hook(self.hook.hook_fn)


    def propagate(self, relevance):
        Z = F.linear(self.X, self.W)

        S = safe_divide_where(relevance, Z)

        relevance = self.x * torch.autograd.grad(Z, self.X, S)[0]
        return relevance
