from src.Model.Blocks.DownSample import DownSample
from src.Model.Blocks.ConvBlock import ConvBlock
from src.Model.Blocks.UpSample import UpSample
from src.Model.Blocks.OutConv import OutConv
import torch

class AutoEncoderCNN(torch.nn.Module):
    """ 
        This AutoEncoder has 4 encoding layers followed by 4 decoding layers with a varying number of filters.
        It also has the possibility of adding skip connections but the U-Net class should be used instead.
    """
    def __init__(self, io_channels=3, n_filters=8, skip=False, batch_norm=True, dropout=False, p=0.1) -> None:
        super().__init__()
        # Input Convolutional Layer
        self.__in_conv_block = ConvBlock(io_channels, n_filters, batch_norm=batch_norm)

        # Encoder Layers
        self.__down_sample1 = DownSample(n_filters, n_filters * 2, batch_norm=batch_norm, dropout=dropout, p=p)
        self.__down_sample2 = DownSample(n_filters * 2, n_filters * 4, batch_norm=batch_norm, dropout=dropout, p=p)
        self.__down_sample3 = DownSample(n_filters * 4, n_filters * 8, batch_norm=batch_norm, dropout=dropout, p=p)
        self.__down_sample4 = DownSample(n_filters * 8, n_filters * 16, batch_norm=batch_norm, dropout=dropout, p=p)

        # Decoder Layers (with optional skip connections)
        self.__up_sample4 = UpSample(n_filters * 16, n_filters * 8, skip=skip, batch_norm=batch_norm, dropout=dropout, p=p)
        self.__up_sample3 = UpSample(n_filters * 8, n_filters * 4, skip=skip, batch_norm=batch_norm, dropout=dropout, p=p)
        self.__up_sample2 = UpSample(n_filters * 4, n_filters * 2, skip=skip, batch_norm=batch_norm, dropout=dropout, p=p)
        self.__up_sample1 = UpSample(n_filters * 2, n_filters, skip=skip, batch_norm=batch_norm, dropout=dropout, p=p)

        # Output Convolutional Layer
        self.__out_conv_block = OutConv(n_filters, io_channels)

        # Intermediary Outputs
        self.feature_maps = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Save the current input image
        self.__input = x.clone()

        # Encode the data
        self.__encoder_1 = self.__in_conv_block(x)
        self.__encoder_2 = self.__down_sample1(self.__encoder_1)
        self.__encoder_3 = self.__down_sample2(self.__encoder_2)
        self.__encoder_4 = self.__down_sample3(self.__encoder_3)

        # Obtain the bottleneck of the network
        self.__bottleneck = self.__down_sample4(self.__encoder_4)

        # Decode the bottleneck
        self.__decoder_4 = self.__up_sample4(self.__bottleneck, self.__encoder_4)
        self.__decoder_3 = self.__up_sample3(self.__decoder_4, self.__encoder_3)
        self.__decoder_2 = self.__up_sample2(self.__decoder_3, self.__encoder_2)
        self.__decoder_1 = self.__up_sample1(self.__decoder_2, self.__encoder_1)
        
        # Reduce the number of channels
        self.__output = self.__out_conv_block(self.__decoder_1)

        # Save the intermediary outputs
        self.feature_maps = self.__save_feature_maps()

        # Return the final output of the model
        return self.__output

    def __save_feature_maps(self) -> list:
        # Save the intermediary outputs
        return [
            # The input image
            self.__input,

            # The encoder outputs
            self.__encoder_1,
            self.__encoder_2,
            self.__encoder_3,
            self.__encoder_4,

            # The bottleneck output
            self.__bottleneck,

            # The decoder outputs
            self.__decoder_4,
            self.__decoder_3,
            self.__decoder_2,
            self.__encoder_1,

            # The final output
            self.__output,
        ]