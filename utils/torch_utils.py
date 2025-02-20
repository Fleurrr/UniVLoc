import math
import random
from typing import Callable
from collections import OrderedDict

import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data
import torch.backends.cudnn as cudnn


# Distributed Data Parallel Utilities

def array_to_tensor(x):
    """Convert all numpy arrays to pytorch tensors."""
    if isinstance(x, list):
        x = [array_to_tensor(item) for item in x]
    elif isinstance(x, tuple):
        x = tuple([array_to_tensor(item) for item in x])
    elif isinstance(x, dict):
        x = {key: array_to_tensor(value) for key, value in x.items()}
    elif isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x

def all_reduce_tensor(tensor, world_size=1):
    r"""Average reduce a tensor across all workers."""
    reduced_tensor = tensor.clone()
    dist.all_reduce(reduced_tensor)
    reduced_tensor /= world_size
    return reduced_tensor

def all_reduce_tensors(x, world_size=1):
    r"""Average reduce all tensors across all workers."""
    if isinstance(x, list):
        x = [all_reduce_tensors(item, world_size=world_size) for item in x]
    elif isinstance(x, tuple):
        x = (all_reduce_tensors(item, world_size=world_size) for item in x)
    elif isinstance(x, dict):
        x = {key: all_reduce_tensors(value, world_size=world_size) for key, value in x.items()}
    elif isinstance(x, torch.Tensor):
        x = all_reduce_tensor(x, world_size=world_size)
    return x


# Dataloader Utilities


def reset_seed_worker_init_fn(worker_id):
    r"""Reset seed for data loader worker."""
    seed = torch.initial_seed() % (2 ** 32)
    # print(worker_id, seed)
    np.random.seed(seed)
    random.seed(seed)


def build_dataloader(
    dataset,
    batch_size=1,
    num_workers=1,
    shuffle=None,
    collate_fn=None,
    pin_memory=False,
    drop_last=False,
    distributed=False,
):
    if distributed:
        sampler = torch.utils.data.DistributedSampler(dataset)
        shuffle = False
    else:
        sampler = None
        shuffle = shuffle

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=collate_fn,
        worker_init_fn=reset_seed_worker_init_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    return data_loader


# Common Utilities


def initialize(seed=None, cudnn_deterministic=True, autograd_anomaly_detection=False):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
    if cudnn_deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
    else:
        cudnn.benchmark = True
        cudnn.deterministic = False
    torch.autograd.set_detect_anomaly(autograd_anomaly_detection)


def release_cuda(x):
    r"""Release all tensors to item or numpy array."""
    if isinstance(x, list):
        x = [release_cuda(item) for item in x]
    elif isinstance(x, tuple):
        x = (release_cuda(item) for item in x)
    elif isinstance(x, dict):
        x = {key: release_cuda(value) for key, value in x.items()}
    elif isinstance(x, torch.Tensor):
        if x.numel() == 1:
            x = x.item()
        else:
            x = x.detach().cpu().numpy()
    return x
    
def release_cuda_tensor(x):
    r"""Release all tensors to Tensor on CPU."""
    if isinstance(x, list):
        x = [release_cuda_tensor(item) for item in x]
    elif isinstance(x, tuple):
        x = (release_cuda_tensor(item) for item in x)
    elif isinstance(x, dict):
        x = {key: release_cuda_tensor(value) for key, value in x.items()}
    elif isinstance(x, torch.Tensor):
        if x.numel() == 1:
            x = x.cpu()
        else:
            x = x.cpu()
    return x


def to_cuda(x):
    r"""Move all tensors to cuda."""
    if isinstance(x, list):
        x = [to_cuda(item) for item in x]
    elif isinstance(x, tuple):
        x = (to_cuda(item) for item in x)
    elif isinstance(x, dict):
        x = {key: to_cuda(value) for key, value in x.items()}
    elif isinstance(x, torch.Tensor):
        x = x.cuda()
    return x


def load_weights(model, snapshot):
    r"""Load weights and check keys."""
    state_dict = torch.load(snapshot)
    model_dict = state_dict['model']
    model.load_state_dict(model_dict, strict=False)

    snapshot_keys = set(model_dict.keys())
    model_keys = set(model.model_dict().keys())
    missing_keys = model_keys - snapshot_keys
    unexpected_keys = snapshot_keys - model_keys

    return missing_keys, unexpected_keys


# Learning Rate Scheduler


class CosineAnnealingFunction(Callable):
    def __init__(self, max_epoch, eta_min=0.0):
        self.max_epoch = max_epoch
        self.eta_min = eta_min

    def __call__(self, last_epoch):
        next_epoch = last_epoch + 1
        return self.eta_min + 0.5 * (1.0 - self.eta_min) * (1.0 + math.cos(math.pi * next_epoch / self.max_epoch))


class WarmUpCosineAnnealingFunction(Callable):
    def __init__(self, total_steps, warmup_steps, eta_init=0.1, eta_min=0.1):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.normal_steps = total_steps - warmup_steps
        self.eta_init = eta_init
        self.eta_min = eta_min

    def __call__(self, last_step):
        # last_step starts from -1, which means last_steps=0 indicates the first call of lr annealing.
        next_step = last_step + 1
        if next_step < self.warmup_steps:
            return self.eta_init + (1.0 - self.eta_init) / self.warmup_steps * next_step
        else:
            if next_step > self.total_steps:
                return self.eta_min
            next_step -= self.warmup_steps
            return self.eta_min + 0.5 * (1.0 - self.eta_min) * (1 + np.cos(np.pi * next_step / self.normal_steps))


def build_warmup_cosine_lr_scheduler(optimizer, total_steps, warmup_steps, eta_init=0.1, eta_min=0.1, grad_acc_steps=1):
    total_steps //= grad_acc_steps
    warmup_steps //= grad_acc_steps
    cosine_func = WarmUpCosineAnnealingFunction(total_steps, warmup_steps, eta_init=eta_init, eta_min=eta_min)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_func)
    return scheduler
    
# calculate covariance of torch tensor (2d feature map)
def get_cov(x, rowvar=False, bias=False, ddof=None, aweights=None):
    """Estimates covariance matrix like numpy.cov"""
    # ensure at least 2D
    if x.dim() == 1:
        x = x.view(-1, 1)

    # treat each column as a data point, each row as a variable
    if rowvar and x.shape[0] != 1:
        x = x.t()

    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0

    w = aweights
    if w is not None:
        if not torch.is_tensor(w):
            w = torch.tensor(w, dtype=torch.float)
        w_sum = torch.sum(w)
        avg = torch.sum(x * (w/w_sum)[:,None], 0)
    else:
        avg = torch.mean(x, 0)

    # Determine the normalization
    if w is None:
        fact = x.shape[0] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof * torch.sum(w * w) / w_sum

    xm = x.sub(avg.expand_as(x))

    if w is None:
        X_T = xm.t()
    else:
        X_T = torch.mm(torch.diag(w), xm).t()

    c = torch.mm(X_T, xm)
    c = c / fact

    return c.squeeze()
    
def _batch_mahalanobis(bL, bx):
    r"""
    Computes the squared Mahalanobis distance :math:`\mathbf{x}^\top\mathbf{M}^{-1}\mathbf{x}`
    for a factored :math:`\mathbf{M} = \mathbf{L}\mathbf{L}^\top`.

    Accepts batches for both bL and bx. They are not necessarily assumed to have the same batch
    shape, but `bL` one should be able to broadcasted to `bx` one.
    """
    n = bx.size(-1)
    bx_batch_shape = bx.shape[:-1]

    # Assume that bL.shape = (i, 1, n, n), bx.shape = (..., i, j, n),
    # we are going to make bx have shape (..., 1, j,  i, 1, n) to apply batched tri.solve
    bx_batch_dims = len(bx_batch_shape)
    bL_batch_dims = bL.dim() - 2
    outer_batch_dims = bx_batch_dims - bL_batch_dims
    old_batch_dims = outer_batch_dims + bL_batch_dims
    new_batch_dims = outer_batch_dims + 2 * bL_batch_dims
    # Reshape bx with the shape (..., 1, i, j, 1, n)
    bx_new_shape = bx.shape[:outer_batch_dims]
    for (sL, sx) in zip(bL.shape[:-2], bx.shape[outer_batch_dims:-1]):
        bx_new_shape += (sx // sL, sL)
    bx_new_shape += (n,)
    bx = bx.reshape(bx_new_shape)
    # Permute bx to make it have shape (..., 1, j, i, 1, n)
    permute_dims = (list(range(outer_batch_dims)) +
                    list(range(outer_batch_dims, new_batch_dims, 2)) +
                    list(range(outer_batch_dims + 1, new_batch_dims, 2)) +
                    [new_batch_dims])
    bx = bx.permute(permute_dims)

    flat_L = bL.reshape(-1, n, n)  # shape = b x n x n
    flat_x = bx.reshape(-1, flat_L.size(0), n)  # shape = c x b x n
    flat_x_swap = flat_x.permute(1, 2, 0)  # shape = b x n x c
    M_swap = torch.triangular_solve(flat_x_swap, flat_L, upper=False)[0].pow(2).sum(-2)  # shape = b x c
    M = M_swap.t()  # shape = c x b

    # Now we revert the above reshape and permute operators.
    permuted_M = M.reshape(bx.shape[:-1])  # shape = (..., 1, j, i, 1)
    permute_inv_dims = list(range(outer_batch_dims))
    for i in range(bL_batch_dims):
        permute_inv_dims += [outer_batch_dims + i, old_batch_dims + i]
    reshaped_M = permuted_M.permute(permute_inv_dims)  # shape = (..., 1, i, j, 1)
    return reshaped_M.reshape(bx_batch_shape)


def _precision_to_scale_tril(P):
    # Ref: https://nbviewer.jupyter.org/gist/fehiepsi/5ef8e09e61604f10607380467eb82006#Precision-to-scale_tril
    Lf = torch.linalg.cholesky(torch.flip(P, (-2, -1)))
    L_inv = torch.transpose(torch.flip(Lf, (-2, -1)), -2, -1)
    Id = torch.eye(P.shape[-1], dtype=P.dtype, device=P.device)
    L = torch.linalg.solve_triangular(L_inv, Id, upper=False)
    return L