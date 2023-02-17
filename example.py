import torch
from gkpd import gkpd, KroneckerConv2d
from gkpd.tensorops import kron

if __name__ == '__main__':

    rank = 8
    a_shape, b_shape = (rank, 16, 16, 3, 1), (rank, 4, 4, 1, 3)

    # Full rank
    a, b = torch.randn(*a_shape), torch.randn(*b_shape)
    w = kron(a, b)

    # Approximation
    a_hat, b_hat = gkpd(w, a_shape[1:], b_shape[1:])
    w_hat = kron(a_hat, b_hat)

    # Reconstruction error
    print("Reconstruction error: {}".format(
       round((torch.linalg.norm((w.reshape(-1) - w_hat.reshape(-1))).detach().numpy()).item(), 4)
    ))