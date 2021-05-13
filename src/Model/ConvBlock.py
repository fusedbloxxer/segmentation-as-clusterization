import torch

class ConvBlock(torch.nn.Module):
    """ Apply consecutive convolutions, followed by an activation function. """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, batch_norm=True, **kwargs):
        super().__init__()
        # Batch normalization flag
        self.__has_batch_norm = batch_norm

        # Convolutions
        self.__conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=True, **kwargs)
        self.__conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, bias=True, **kwargs)

        # Batch normalization
        if self.__has_batch_norm:
            self.__batch_norm1 = torch.nn.BatchNorm2d(num_features=in_channels)
            self.__batch_norm2 = torch.nn.BatchNorm2d(num_features=out_channels)

        # Activation function
        self.__activ_fun = torch.nn.ReLU()

    def forward(self, x):
        # Apply a batch normalization layer
        if self.__has_batch_norm:
            x = self.__batch_norm1(x)
            
        # Apply the first convolution
        x = self.__activ_fun(self.__conv1(x))

        # Apply a batch normalization layer
        if self.__has_batch_norm:
            x = self.__batch_norm2(x)

        # Apply the second convolution
        x = self.__activ_fun(self.__conv2(x))
        return x