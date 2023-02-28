import torch
from torch.nn import functional as F


def info_value_of_dtype(dtype: torch.dtype):
    """
    Returns `torch.finfo` or `torch.iinfo`
    """
    if dtype.is_floating_point:
        return torch.finfo
    elif dtype == torch.bool:
        raise TypeError("Does not support torch.bool info value")
    else:
        return torch.iinfo


def tiny_value_of_dtype(dtype: torch.dtype):
    """
    Returns a moderately tiny value for a given PyTorch data type that is used to avoid numerical
    issues such as division by zero.
    This is different from `info_value_of_dtype(dtype).tiny` because it causes some NaN bugs.
    Only supports floating point dtypes.
    """
    if not dtype.is_floating_point:
        raise TypeError("Only supports floating point dtypes.")
    if dtype == torch.float or dtype == torch.double:
        return 1e-13
    elif dtype == torch.half:
        return 1e-4
    else:
        raise TypeError("Does not support dtype " + str(dtype))


def min_value_of_dtype(dtype: torch.dtype):
    """
    Returns the minimum value of a give PyTorch data type.
    """
    return info_value_of_dtype(dtype).min


def max_value_of_dtype(dtype: torch.dtype):
    """
    Returns the maximum value of a give PyTorch data type.
    """
    return info_value_of_dtype(dtype).max


def masked_softmax(
        vector: torch.Tensor,
        mask: torch.BoolTensor = None,
        dim: int = -1,
) -> torch.Tensor:
    if mask is None:
        return F.softmax(vector, dim=dim)
    while mask.dim() < vector.dim():
        mask = mask.unsqueeze(1)
    masked_vector = vector.masked_fill(~mask, min_value_of_dtype(vector.dtype))
    return F.softmax(masked_vector, dim=dim)
