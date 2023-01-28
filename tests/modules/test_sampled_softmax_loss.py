from torch import nn
from torch.nn.functional import sigmoid

from ...prestonlp.modules import SampledSoftmaxLoss


def test_sampled_softmax_loss():
    class TestNet(nn.Module):
        def __int__(self, input_dim, output_dim):
            self.repr_layer = nn.Embedding(input_dim, 64)
            self.cls_layer = nn.Embedding(64, output_dim)
            self.sampled_softmax = SampledSoftmaxLoss(num_words=input_dim,
                                                      embedding_dim=64,
                                                      num_samples=5,
                                                      sparse=True)

        def forward(self, input_tensor):
            input_tensor = self.repr_layer(input_tensor)
            input_tensor = sigmoid(input_tensor)
            input_tensor = self.sampled_softmax(input_tensor)
            return input_tensor

