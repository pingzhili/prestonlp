import torch

from prestonlp.nn.util import masked_softmax, tiny_value_of_dtype


class AttentionBase(torch.nn.Module):
    """
    Base class for varies attention mechanisms.
    Attention takes batched query vectors and a key matrix as input, then compute the similarity between each query
    vector and each row of the key matrix. Optionally perform a normalization (masked) on the matrix.
    """

    def __init__(self, normal: bool = True) -> None:
        super().__init__()
        self._normal = normal

    def forward(
            self,
            vector: torch.Tensor,
            matrix: torch.Tensor,
            matrix_mask: torch.BoolTensor = None
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        vector: torch.Tensor, required
        matrix: torch.Tensor, required
        matrix_mask: torch.BoolTensor, optional

        Returns
        -------
        return the similarities matrix after mask normalized (optionally)
        """
        similarities = self._forward_internal(vector, matrix)
        if self._normal:
            return masked_softmax(similarities, mask=matrix_mask)
        else:
            return similarities

    def _forward_internal(self, vector: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class DotProductAttention(AttentionBase):
    """
    Computes attention scores using dot product function.
    Reference: [Attention Is All You Need (Vaswani et al, 2017)]

    Examples
    --------
    >>> batch_size, vocab_size, hid_dim, attention_module = 10, 16, 8, DotProductAttention()
    >>> queries, key_matrix = torch.randn(batch_size, hid_dim), torch.randn(vocab_size, hid_dim)
    >>> scores = attention_module(queries, key_matrix)
    >>> scores.shape
    torch.Size([10, 16])
    """

    def __init__(self):
        super().__init__()

    def _forward_internal(self, vector: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        return matrix.matmul(vector.unsqueeze(-1)).squeeze(-1)


class CosineAttention(AttentionBase):
    """
    Computes attention scores using cosine similarity function.

    Examples
    --------
    >>> batch_size, vocab_size, hid_dim, attention_module = 10, 16, 8, CosineAttention()
    >>> queries, key_matrix = torch.randn(batch_size, hid_dim), torch.randn(vocab_size, hid_dim)
    >>> scores = attention_module(queries, key_matrix)
    >>> scores.shape
    torch.Size([10, 16])
    """

    def __init__(self):
        super().__init__()

    def _forward_internal(self, vector: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        vector_norm = vector / (vector.norm(p=2, dim=-1, keepdim=True) + tiny_value_of_dtype(vector.dtype))
        matrix_norm = matrix / (matrix.norm(p=2, dim=-1, keepdim=True) + tiny_value_of_dtype(vector.dtype))
        return matrix_norm.matmul(vector_norm.unsqueeze(-1)).squeeze(-1)
