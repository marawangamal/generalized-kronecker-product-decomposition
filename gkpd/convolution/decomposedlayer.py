import torch.nn as nn


class DecomposedLayer(nn.Module):
    """Abstract class for decomposed layers"""

    def __init__(self):
        super(DecomposedLayer, self).__init__()
        pass

    @classmethod
    def from_conv(cls, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def from_tensor(cls, *args, **kwargs):
        raise NotImplementedError

    def get_flops(self, *args, **kwargs):
        raise NotImplementedError
