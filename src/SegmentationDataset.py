from src.FileLoader import load_data_file
from torch.utils.data import Dataset
from src.Config.Args import args
import random
import torch
import os
import re

class SegmentationDataset(Dataset):
    def __init__(self, mode='train', input_dir=args.input_dir, imgs_per_file=args.imgs_per_file, shuffle=args.shuffle, seed=args.seed):
        """
        Specify the directory containing all data
        """
        self.__input_dir = input_dir

        # Fetch the relative paths to the data files
        self.__data_files = [f'{self.__input_dir}/{fname}' for fname in os.listdir(self.__input_dir) if re.search(mode, fname)]
        
        # Specify how many entries are in one file
        self.__imgs_per_file = imgs_per_file

        # Use specified seed to shuffle the data
        if shuffle:
            random.seed(seed)
            random.shuffle(self.__data_files)

    def __getitem__(self, index) -> torch.Tensor:
        """
        Reads the image at position 'index % imgs_per_file' in the 'index / imgs_per_file' file.
        Returns a Tensor.
        """
        # Load the file in memory using a caching strategy
        data = load_data_file(self.__data_files[index // self.__imgs_per_file])
        
        # Select the item at 'index % imgs_per_file'
        item = data[index % self.__imgs_per_file]

        # Return a pair containing (image, mask)
        return item['image'], item['mask'], item['count']

    def __len__(self):
        """
        Return the size of this dataset. 
        This is given by the number of images in all data files.
        """
        return len(self.__data_files) * self.__imgs_per_file