"""Modules for building the CNN for MNIST"""

import torch
from torch import nn

# %%
ACTS = {
    'relu':nn.ReLU,
    'sigmoid':nn.Sigmoid,
    'tanh':nn.Tanh,}

# %%

class ConvNet(nn.Module):
    """
    CNN for MNIST using 3x3 convs and one final FC layer.
    """
    def __init__(self,
                 input_size=28,
                 channels=[32, 32, 16, 8],
                 output_size=10,
                 activation='relu'):
        """CNN for MNIST using 3x3 convs and one final fully connected layer.
        Performs one 2x2 max pool after the first conv.

        Parameters
        ----------
        input_size : int
            Dimension of input square image (the default is 28 for MNIST).
        channels : list of ints
            List of output channels of conv layers (the default is [32,32,16,8]).
        output_size : int
            Number of output classes (the default is 10 for MNIST).
        activation : str
            One of 'relu', 'sigmoid' or 'tanh' (the default is 'relu').

        """
        super().__init__()

        act = ACTS[activation]

        convs = [nn.Conv2d(kernel_size=3, in_channels=in_ch, out_channels=out_ch)
                 for in_ch, out_ch in zip([1]+channels[:-1], channels)]

        self.conv_net = nn.Sequential(
            convs[0],
            nn.MaxPool2d(kernel_size=2),
            act(),
            *[layer for tup in zip(convs[1:], [act() for _ in convs[1:]]) for layer in tup]
        )

        with torch.no_grad():
            test_inp = torch.randn(1,1,input_size,input_size)
            features = self.conv_net(test_inp)
            feature_count = features.view(-1).shape[0]

        self.dense = nn.Linear(feature_count, output_size)

    def forward(self, input):
        out = self.conv_net(input)
        out = self.dense(out.view(input.shape[0], -1))
        return out

# %%

# net = ConvNet()
# test_inp = torch.randn(32,1,28,28)
# out = net(test_inp)
# out.shape
