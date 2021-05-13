from src.Model.ConvBlock import ConvBlock
import torch

class DownSample(torch.nn.Module):
    """ Apply max pooling operation to downsample the image and then a convolution block. """
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, dropout=False, p=0.1, **kwargs):
        super().__init__()
        # Dropout flag
        self.__has_dropout = dropout

        # Define the max pooling layer
        self.__pool = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride, **kwargs)

        # Optional dropout layer
        if self.__has_dropout:
            self.__dropout = torch.nn.Dropout2d(p)
        
        # Define the convolution block
        self.__conv_block = ConvBlock(in_channels, out_channels)

    def forward(self, x):
        # Apply a pooling operation, which downscales the image
        x = self.__pool(x)

        # Optionally apply a dropout regularization layer
        if self.__has_dropout:
            x = self.__dropout(x)

        # Apply a convolutional block
        x = self.__conv_block(x)
        return x