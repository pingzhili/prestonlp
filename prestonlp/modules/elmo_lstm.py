import torch
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class ElmoLstm(torch.nn.Module):
    """
    Simple implementation of ELMo ("Deep contextualized word representations", Peters et al.) model,
     using torch.nn.LSTM
    """

    def __init__(
            self,
            input_size: int,
            embedding_dim: int,
            hidden_size: int,
            num_layers: int = 2,
            enforce_sorted: bool = False
    ):
        super().__init__()
        self.num_layers = num_layers * 2
        self.hidden_size = hidden_size
        self.embedding = torch.nn.Embedding(input_size, embedding_dim)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.enforce_sorted = enforce_sorted

    def forward(
            self,
            seq_idx: torch.LongTensor,
            seq_len: torch.LongTensor
    ):
        """
        Parameters
        ----------
        seq_idx: Tensor
            a list of indices of padded sequences
        seq_len: Tensor
            lengths of the sequences
        Returns
        --------
        output: Tensor
            padded output of the last layer of the LSTM, of shape (batch_size, seq_len, hidden_size * num_directions)
        hn: Tensor
            hidden state of the last layer of the LSTM, of shape (num_layers * num_directions, batch_size, hidden_size)

        Examples
        --------
        >>> elmo = ElmoLstm(input_size=10, embedding_dim=16, hidden_size=8, num_layers=2)
        >>> seq_ids = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 0], [1, 2, 3, 4, 5, 6, 7, 0, 0, 0], [1, 2, 3, 4, 5, 0, 0, 0, 0, 0]])
        >>> seq_lens = torch.LongTensor([9, 7, 5])
        >>> outputs, hiddens = elmo(seq_ids, seq_lens)
        >>> outputs.shape
        torch.Size([3, 9, 16])
        >>> hiddens.shape
        torch.Size([4, 3, 8])
        """
        if seq_idx.shape[0] != seq_len.shape[0]:
            raise KeyError(f"Input sequence indices does not match with sequence length {seq_len.shape[0]}")
        seq = self.embedding(seq_idx)
        pack = pack_padded_sequence(seq, seq_len.cpu(), batch_first=True, enforce_sorted=self.enforce_sorted)
        h0 = torch.zeros(self.num_layers, seq.shape[0], self.hidden_size).to(seq_idx.device)
        c0 = torch.zeros(self.num_layers, seq.shape[0], self.hidden_size).to(seq_idx.device)
        output, (hn, _) = self.lstm(pack, (h0, c0))
        output, _ = pad_packed_sequence(output, batch_first=True)
        return output, hn
