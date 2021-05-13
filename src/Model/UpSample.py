from src.Model.ConvBlock import ConvBlock
import torch

class UpSample(torch.nn.Module):
    """ 
    Apply a transpose convolution operation to upscale the image and reduce the number of channels. 
    Add skip connection: concatenate the current output with the output of a previous layer.
    Then apply a convolution block.
    """
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, dropout=False, p=0.1, **kwargs):
        super().__init__()
        # Dropout flag
        self.__has_dropout = dropout

        # Define the transpose convolution layer
        self.__tconv = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, bias=True, **kwargs)

        # Optional dropout layer
        if dropout:
            self.__dropout = torch.nn.Dropout2d(p)

        # Create a convolutional block
        self.__conv_block = ConvBlock(in_channels, out_channels)

    def forward(self, x, x_prev):
        # Apply a transpose convolution, which upscales the image
        x = self.__tconv(x)

        # Add a skip connection to a previous symmetrical layer
        x = torch.cat((x_prev, x), dim=1)

        # Optionally apply a dropout regularization layer
        if self.__has_dropout:
            x = self.__dropout(x)

        # Apply a convolutional block
        x = self.__conv_block(x)
        return x