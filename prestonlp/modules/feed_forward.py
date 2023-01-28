from typing import List, Union

import torch


class FeedForward(torch.nn.Module):
    """
    A sequence of `linear` layers with activation functions in between.

    Examples
    --------
    >>> FeedForward(16, [32, 8], 2, torch.nn.ReLU(), 0.3)
    FeedForward(
      (_linear_layers): ModuleList(
        (0): Linear(in_features=16, out_features=32, bias=True)
        (1): Linear(in_features=32, out_features=8, bias=True)
      )
      (_activate_funcs): ModuleList(
        (0): ReLU()
        (1): ReLU()
      )
      (_dropout_layers): ModuleList(
        (0): Dropout(p=0.3, inplace=False)
        (1): Dropout(p=0.3, inplace=False)
      )
    )
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dims: Union[int, List[int]],
            num_layers: int,
            activate_funcs: Union[torch.nn.Module, List[torch.nn.Module]],
            dropout: Union[float, List[float]] = 0.0
    ):
        super().__init__()

        if num_layers != len(hidden_dims):
            raise KeyError(f"num_layer {num_layers} does not match hidden_dims array length {len(hidden_dims)}")
        if isinstance(dropout, list) and num_layers != len(dropout):
            raise KeyError(f"num_layer {num_layers} does not match dropout array length {len(dropout)}")
        if isinstance(activate_funcs, list) and num_layers != len(activate_funcs):
            raise KeyError(f"num_layer {num_layers} does not match dropout array length {len(activate_funcs)}")

        _linear_layers = []
        for layer_input_dim, layer_output_dim in zip([input_dim] + hidden_dims[:-1], hidden_dims):
            _linear_layers.append(torch.nn.Linear(layer_input_dim, layer_output_dim))
        self._linear_layers = torch.nn.ModuleList(_linear_layers)
        activate_funcs = activate_funcs if isinstance(activate_funcs, list) else torch.nn.ModuleList(
            [activate_funcs] * num_layers)
        self._activate_funcs = activate_funcs
        dropout = [dropout] * num_layers if isinstance(dropout, float) else dropout
        self._dropout_layers = torch.nn.ModuleList([torch.nn.Dropout(dropout_rate) for dropout_rate in dropout])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = inputs
        for linear_layer, activate_func, dropout_layer in zip(self._linear_layers, self._activate_funcs,
                                                              self._dropout_layers):
            output = dropout_layer(activate_func(linear_layer(output)))
        return output
