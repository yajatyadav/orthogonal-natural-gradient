from utils.metric import accuracy
from ogd.ogd_utils import get_model_parameters, get_model_n_layers

import torch
from tqdm.auto import tqdm
from torch.nn.utils.convert_parameters import _check_param_device, parameters_to_vector, vector_to_parameters
import torch.nn as nn

from collections import defaultdict

from agents.exp_replay import Memory
from types import MethodType

import pytorch_lightning as pl
import wandb
import random
import models
from models.mlp import MLP


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__



import torch
from tqdm import tqdm
from math import sqrt

def orthonormalize_custom(vectors,
                   inner_prod,
                   normalize: bool = True,
                   start_idx: int = 0):
    """
    Orthonormalize the columns of `vectors` in place, using Gram–Schmidt
    with a custom inner product.

    Args:
      vectors:       Tensor of shape (dim, n_basis).
      inner_prod:    Function taking two (dim,) tensors → scalar tensor.
      normalize:     Whether to L2‐normalize each vector after orthogonalizing.
      start_idx:     Which column to start Gram–Schmidt from (0 ⇒ do all).
    Returns:
      The same Tensor `vectors`, with columns orthogonal (and normalized if requested).
    """
    dim, n = vectors.shape
    assert n <= dim, "number of vectors must be ≤ ambient dimension"

    # If requested, normalize the first column.
    if normalize and start_idx == 0:
        v0 = vectors[:, 0]
        norm0 = torch.sqrt(inner_prod(v0, v0))
        vectors[:, 0] = v0 / norm0

    # start_idx=0 is a special case: we want to do columns 1…n−1
    i0 = 1 if start_idx == 0 else start_idx

    for i in tqdm(range(i0, n), desc="orthonormalizing ..."):
        v = vectors[:, i]
        # subtract off projections onto all previous basis vectors
        for j in range(i):
            u = vectors[:, j]
            coeff = inner_prod(u, v)    # <u, v>
            v = v - coeff * u           # subtract projection

        # optional normalize
        if normalize:
            norm = torch.sqrt(inner_prod(v, v))
            v = v / norm

        vectors[:, i] = v

    return vectors


def project_vec_custom(vec,
                proj_basis,
                inner_prod):
    """
    Project `vec` onto the subspace spanned by the columns of `proj_basis`,
    using a custom inner product.

    Args:
      vec:         Tensor of shape (dim,)
      proj_basis:  Tensor of shape (dim, k), columns are basis vectors
      inner_prod:  Function taking two (dim,) tensors → scalar tensor.
    Returns:
      The projected vector of shape (dim,)
    """
    if proj_basis.numel() == 0:
        return torch.zeros_like(vec)

    # accumulate sum_j <u_j, vec> u_j
    projection = torch.zeros_like(vec)
    for j in range(proj_basis.shape[1]):
        uj = proj_basis[:, j]
        coeff = inner_prod(uj, vec)
        projection = projection + coeff * uj

    return projection











def orthonormalize(vectors, gpu, normalize=True, start_idx=0):
    assert (vectors.size(1) <= vectors.size(0)), 'number of vectors must be smaller or equal to the dimension'
    # TODO : Check if start_idx is correct :)
    # orthonormalized_vectors = torch.zeros_like(vectors)
    if normalize:
        vectors[:, 0] = vectors[:, 0] / torch.norm(vectors[:, 0], p=2)
    else:
        vectors[:, 0] = vectors[:, 0]

    if start_idx == 0 :
        start_idx = 1
    for i in tqdm(range(start_idx, vectors.size(1)), desc="orthonormalizing ..."):
        vector = vectors[:, i]
        V = vectors[:, :i]
        PV_vector = torch.mv(V, torch.mv(V.t(), vector))
        if normalize:
            vectors[:, i] = (vector - PV_vector) / torch.norm(vector - PV_vector, p=2)
        else:
            vectors[:, i] = (vector - PV_vector)

    return vectors


def project_vec(vec, proj_basis, gpu):
    if proj_basis.shape[1] > 0:  # param x basis_size
        dots = torch.matmul(vec, proj_basis)  # basis_size
        # out = torch.matmul(proj_basis, dots)
        # TODO : Check !!!!
        out = torch.matmul(proj_basis, dots.T)
        return out
    else:
        return torch.zeros_like(vec)


def parameters_to_grad_vector(parameters):
    # Flag for the device where the parameter is located
    param_device = None

    vec = []
    for param in parameters:
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)

        ## YY: fix for case when param.grad is None at the very start
        if param.grad is None:
            # no gradient → zero vector of appropriate length
            vec.append(torch.zeros(param.numel(), device=param_device))
        else:
            vec.append(param.grad.view(-1))
    return torch.cat(vec)


def grad_vector_to_parameters(vec, parameters):
    # Ensure vec of type Tensor
    if not isinstance(vec, torch.Tensor):
        raise TypeError('expected torch.Tensor, but got: {}'
                        .format(torch.typename(vec)))
    # Flag for the device where the parameter is located
    param_device = None

    # Pointer for slicing the vector for each parameter
    pointer = 0
    for param in parameters:
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)

        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        # param.data = vec[pointer:pointer + num_param].view_as(param).data
        param.grad = vec[pointer:pointer + num_param].view_as(param).clone()

        # Increment the pointer
        pointer += num_param


def validate(testloader, model, gpu, size):
    model.eval()

    acc = 0
    acc_cnt = 0
    with torch.no_grad():
        for idx, data in enumerate(testloader):
            if size is None or idx < size:
                data, target, task = data
                if gpu:
                    with torch.no_grad():
                        data = data.cuda()
                        target = target.cuda()

                outputs = model.forward(data, task)

                acc += accuracy(outputs, target)
                acc_cnt += 1

            else:
                break
    return acc / acc_cnt


def count_parameter(model):
    return sum(p.numel() for p in model.parameters())


def get_n_trainable(model):
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n_trainable