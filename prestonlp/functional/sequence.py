import torch


def sequence_mask(length_tensor: torch.tensor, max_len: int = None):
    """
    Same as tf.sequence_mask (https://www.tensorflow.org/api_docs/python/tf/sequence_mask)

    Parameters
    ----------
    length_tensor: `torch.tensor`
        A tensor containing lengths of each sequence in a batch.
    max_len: `int`, optional
        The max length in a batch.

    Examples
    --------
    >>> sequence_mask(torch.tensor([1,4,3]), max_len=5)
    tensor([[ True, False, False, False, False],
            [ True,  True,  True,  True, False],
            [ True,  True,  True, False, False]])
    >>> sequence_mask(torch.tensor([[1,3], [2, 4]]))
    tensor([[[ True, False, False, False],
             [ True,  True,  True, False]],
    <BLANKLINE>
            [[ True,  True, False, False],
             [ True,  True,  True,  True]]])
    """
    lengths = length_tensor.reshape(-1)
    full_size = lengths.numel()
    max_len = max_len if max_len else int(lengths.max())
    ret = torch.arange(0, max_len, device=length_tensor.device).type_as(length_tensor).unsqueeze(0)
    ret = ret.expand(full_size, max_len).lt(lengths.unsqueeze(1))

    # lengths_tensor.shape is a tuple!
    return ret.reshape(length_tensor.shape + (max_len,))
