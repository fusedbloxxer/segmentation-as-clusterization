import random
import torch
import json
import os

class Args():
    def __init__(self):
        """
            Read configuration file and import settings
        """
        # Check if the configuration file exists
        if not os.path.exists('src/Config/Config.json'):
            raise ValueError('The src/config.json file does not exist.')
        
        # Open and read the configuration settings from the file
        with open('src/Config/Config.json', 'r') as config_file:
            # Append values to current instance
            self.__dict__.update(json.load(config_file))
        
        # Set initial random seeds
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Find device type
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        # Configure runtime values
        self.dtype = torch.float32 if self.use_cuda else torch.float64
        self.kwargs = { 'num_workers': 1, 'pin_memory': True } if self.use_cuda else {}
        

# Instantiate the configuration and expose it
args = Args()