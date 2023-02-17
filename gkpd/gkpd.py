from typing import Union

import torch
import numpy as np

from gkpd.tensorops import multidimensional_unfold


def gkpd(tensor: torch.Tensor, a_shape: Union[list, tuple], b_shape: Union[list, tuple],
         atol: float = 1e-3) -> tuple:
    """Finds Kronecker decomposition of `tensor` via SVD.
    Patch traversal and reshaping operations all follow a C-order convention (last dimension changing fastest).
    Args:
        tensor (torch.Tensor): Tensor to be decomposed.
        a_shape (list, tuple): Shape of first Kronecker factor.
        b_shape (list, tuple): Shape of second Kronecker factor.
        atol (float): Tolerance for determining tensor rank.

    Returns:
        a_hat: [rank, *a_shape]
        b_hat: [rank, *b_shape]
    """

    if not np.all(np.array([a_shape, b_shape]).prod(axis=0) == np.array(tensor.shape)):
        raise ValueError("Received invalid factorization dimensions for tensor during its GKPD decomposition")

    with torch.no_grad():
        w_unf = multidimensional_unfold(
            tensor.unsqueeze(0), kernel_size=b_shape, stride=b_shape
        )[0].T  # [num_positions, prod(s_dims)]

        u, s, v = torch.svd(w_unf)
        rank = len(s.detach().numpy()[np.abs(s.detach().numpy()) > atol])

        # Note: pytorch reshaping follows C-order as well
        a_hat = torch.stack([s[i].item() * u[:, i].reshape(*a_shape) for i in range(rank)])  # [rank, *a_shape]
        b_hat = torch.stack([v.T[i].reshape(*b_shape) for i in range(rank)])  # [rank, *b_shape]

    return a_hat, b_hat