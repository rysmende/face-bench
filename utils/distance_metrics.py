import torch
import numpy as np

def __l2_normalize(x: torch.Tensor) -> torch.Tensor:
    '''
    Normalize tensor via L2 Normalization.

        Parameters: 
            x (torch.Tensor): Input tensor
        Returns:
            x (torch.Tensor): Normalized tensor
    '''
    return x / torch.sqrt(torch.sum(torch.mul(x, x)))

def get_cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    '''
    Compute and return Cosine Distance.

        Parameters: 
            x (torch.Tensor): First  input tensor
            y (torch.Tensor): Second input tensor
        Returns:
            x (torch.Tensor): Computed distance between tensors
    '''
    a = torch.matmul(x, y)
    b = torch.sum(torch.mul(x, x))
    c = torch.sum(torch.mul(y, y))
    return 1 - (a / (torch.sqrt(b) * torch.sqrt(c)))

def get_euclidean_distance(x: torch.Tensor, y: torch.Tensor)-> torch.Tensor:
    '''
    Compute and return Euclidean Distance.

        Parameters: 
            x (torch.Tensor): First  input tensor
            y (torch.Tensor): Second input tensor
        Returns:
            x (torch.Tensor): Computed distance between tensors
    '''
    return torch.sqrt(torch.sum(torch.mul(x - y, x - y)))

def get_l2_euclidean_distance(x: torch.Tensor, y: torch.Tensor)-> torch.Tensor:
    '''
    Compute and return L2 Normalized Euclidean Distance.

        Parameters: 
            x (torch.Tensor): First  input tensor
            y (torch.Tensor): Second input tensor
        Returns:
            x (torch.Tensor): Computed distance between tensors
    '''
    x = __l2_normalize(x)
    y = __l2_normalize(y)
    return torch.sqrt(torch.sum(torch.mul(x - y, x - y))) 