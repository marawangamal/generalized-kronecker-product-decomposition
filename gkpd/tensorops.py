from typing import Union

import torch
import numpy as np


def multidimensional_unfold(tensor: torch.Tensor, kernel_size: tuple, stride: tuple,
                            device: torch.device = torch.device('cpu')) -> torch.Tensor:
    r"""Unfolds `tensor` by extracting patches of shape `kernel_size`.

    Reshaping and traversal for patch extraction both follow C-order convention (last index changes the fastest).

    Args:
        tensor: Input tensor to be unfolded with shape [N, *spatial_dims] (N is batch dimension)
        kernel_size: Patch size.
        stride: Stride of multidimensional traversal.
        device: Device used for operations.

    Returns:
       Unfolded tensor with shape [N, :math:`\prod_k kernel_size[k]`, L]

    """

    s_dims = tensor.shape[1:]  # spatial dimensions

    # Number of positions along each axis
    num_positions = [np.floor((s_dims[i] - (kernel_size[i] - 1) - 1) / stride[i] + 1).astype(int)
                     for i in range(len(s_dims))]

    # Start indices for each position in each axis
    positions = [torch.tensor([n * stride[i] for n in range(num_positions[i] - 1, -1, -1)]) for i in
                 range(len(num_positions))]

    # Each column is a flattened patch
    output = torch.zeros(tensor.size(0), np.prod(kernel_size).item(), np.prod(num_positions).item(), device=device)

    for i, pos in enumerate(torch.cartesian_prod(*positions)):
        start_pos = torch.tensor([0, *pos])
        end_pos = torch.tensor([tensor.size(0), *(pos + torch.tensor(kernel_size))])
        patch = multidimensional_slice(tensor, start_pos, end_pos)  # n,f2,c2,h2,w2
        output[:, :, np.prod(num_positions).item() - 1 - i] = patch.reshape(tensor.size(0), -1)

    return output


def multidimensional_slice(tensor: torch.Tensor, start: torch.Tensor, stop: torch.Tensor) -> torch.Tensor:
    """Returns A[start_1:stop_1, ..., start_n:stop_n] for tensor A"

    Args:
        tensor: Input tensor `A`
        start: start indices
        stop: stop indices

    Returns:
         A[start_1:stop_1, ..., start_n:stop_n]
    """
    slices = [slice(start[i], stop[i]) for i in range(len(start))]
    return tensor[slices]


def kron(a: Union[torch.Tensor, np.ndarray], b: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    """Kronecker product between factors `a` and `b`

    Args:
        a: First factor
        b: Second factor

    Returns:
        Tensor containing kronecker product between `a` and `b`
    """

    a = torch.from_numpy(a) if isinstance(a, np.ndarray) else a
    b = torch.from_numpy(b) if isinstance(b, np.ndarray) else b

    return torch.stack([torch.kron(a[k], b[k]) for k in range(a.shape[0])]).sum(dim=0)