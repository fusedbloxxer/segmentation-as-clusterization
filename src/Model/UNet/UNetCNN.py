from src.Model.AutoEncoder.AutoEncoderCNN import AutoEncoderCNN

class UNetCNN(AutoEncoderCNN):
    """ 
        The U-Net is an AutoEncoder with the skip connections being active. 
    """
    def __init__(self, io_channels=3, n_filters=8, batch_norm=True, dropout=False, p=0.1) -> None:
        super(UNetCNN, self).__init__(skip=True, io_channels=io_channels, n_filters=n_filters, batch_norm=batch_norm, dropout=dropout, p=p)