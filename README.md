# Convolutional Neural Network Compression through Generalized Kronecker Product Decomposition
This repository is the official implementation of [Convolutional Neural Network Compression through Generalized Kronecker Product Decomposition](https://arxiv.org/abs/2109.14710)

<p align="center">
<img width="500" alt="image" src="https://user-images.githubusercontent.com/44987843/219804868-365e1625-c5a7-4c3f-b8eb-ec05776f6608.png">
</p>

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Usage

To decompose a tensor:

```python

    from gkpd import gkpd, kron
    
    w = torch.randn(64, 64, 3, 3)
    
    # Decomposition
    a_shape, b_shape = (rank, 16, 16, 3, 1), (rank, 4, 4, 1, 3)
    w = kron(a, b)
    a_hat, b_hat = gkpd(w, a_shape[1:], b_shape[1:])

    # Reconstruct approximation
    w_hat = kron(a_hat, b_hat)
```


## Results

### Image Classification on CIFAR-10

Model | Params (M) | Compression   | Accuracy (%)
------------- | ------------- |---------------| ------------- |
Resnet32 | 0.46 | 1× | 92.55
TuckerResNet32 | 0.09 | 5× | 87.7
TensorTrainResNet32 | 0.096 | 4.8× | 88.3
TensorRingResNet32 | 0.09 | 5× | 90.6
**KroneckerResNet32** | **0.09** | **5×** | **91.52**





