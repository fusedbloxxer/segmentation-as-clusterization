from src.Model.DownSample import DownSample
from src.Model.ConvBlock import ConvBlock
from src.Model.UpSample import UpSample
from src.Model.OutConv import OutConv
import torch

class UNetCNN(torch.nn.Module):
    def __init__(self, n_filters: int = 8) -> None:
        super().__init__()
        # Input Convolutional Layer
        self.__in_conv_block = ConvBlock(3, n_filters)

        # Encoder Layers
        self.__down_sample1 = DownSample(n_filters, n_filters * 2)
        self.__down_sample2 = DownSample(n_filters * 2, n_filters * 4)
        self.__down_sample3 = DownSample(n_filters * 4, n_filters * 8)
        self.__down_sample4 = DownSample(n_filters * 8, n_filters * 16)

        # Decoder Layers
        self.__up_sample4 = UpSample(n_filters * 16, n_filters * 8)
        self.__up_sample3 = UpSample(n_filters * 8, n_filters * 4)
        self.__up_sample2 = UpSample(n_filters * 4, n_filters * 2)
        self.__up_sample1 = UpSample(n_filters * 2, n_filters)

        # Output Convolutional Layer
        self.__out_conv_block = OutConv(n_filters, 3)

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