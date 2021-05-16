from src.Model.Blocks.ConvBlock import ConvBlock
import torch

class UpSample(torch.nn.Module):
    """ 
        Apply a transpose convolution operation to upscale the image.
        Then apply a convolution block.
    """
    def __init__(self, in_channels, out_channels, skip, kernel_size=2, stride=2, batch_norm=True, dropout=False, p=0.1, **kwargs):
        super().__init__()
        # Dropout flag
        self.__has_dropout = dropout

        # Add skip connections flag
        self.__has_skip = skip

        # Define the transpose convolution layer
        self.__tconv = torch.nn.ConvTranspose2d(in_channels, out_channels if skip else in_channels, kernel_size, stride, bias=True, **kwargs)

        # Optional dropout layer
        if dropout:
            self.__dropout = torch.nn.Dropout2d(p)

        # Create a convolutional block
        self.__conv_block = ConvBlock(in_channels, out_channels, batch_norm=batch_norm)

    def forward(self, x, x_prev=None):
        # Apply a transpose convolution, which upscales the image
        x = self.__tconv(x)

        # Add a skip connection to a previous symmetrical layer
        if self.__has_skip:
            x = torch.cat((x_prev, x), dim=1)

        # Optionally apply a dropout regularization layer
        if self.__has_dropout:
            x = self.__dropout(x)

        # Apply a convolutional block
        x = self.__conv_block(x)
        return x