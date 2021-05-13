from src.Config.Args import args
from functools import lru_cache
import pickle
import torch

@lru_cache(maxsize=args.lru_cache_size)
def load_data_file(filepath):
    """
    Receives the relative path to a file, and uses LRU caching strategy to return the result.
    """
    with open(filepath, 'rb') as handle:
        # Load the data file from disk into memory
        data = pickle.load(handle)
        print(f'Loaded {filepath} in memory.')
        
        # Iterate through the rows in the data file
        for entry in data:
            # Transform relevant columns to tensors
            for key in ['image', 'mask']:
                # Transform data to Pytorch tensors
                entry[key] = torch.from_numpy(entry[key])

                # Remove extra nesting
                entry[key] = entry[key].squeeze(0)

                # Permute dimensions according to Pytorch requirements
                if key == 'image':
                    # Permute dimensions from (H, W, C) to (C, H, W)
                    entry[key] = entry[key].permute(2, 0, 1)
                elif key == 'mask':
                    # Permute dimensions from (N, H, W, C) to (N, C, H, W)
                    entry[key] = entry[key].permute(0, 3, 1, 2)

            # Get list of visible objects on screen
            entry['count'] = torch.from_numpy(entry['visibility'])

            # Remove extra nesting
            entry['count'] = entry['count'].squeeze(0)

            # Sum the number of visible objects
            entry['count'] = entry['count'].sum()

            # Remove unnecessary columns
            for key in set(entry.keys()) - {'image', 'mask', 'count'}:
                entry[key] = None
                
    # Return the processed data in Pytorch format
    return data