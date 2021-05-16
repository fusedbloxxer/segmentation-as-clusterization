import torch

class OutConv(torch.nn.Module):
    """ 
        Apply an output convolution operation to change the number of channels.
     """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, **kwargs):
        super().__init__()
        self.__conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=True, **kwargs)

    def forward(self, x):
        # Apply the output convolution layer
        x = self.__conv(x)
        return x