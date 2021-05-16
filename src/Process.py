import torch

def gray_map(images: torch.Tensor, gray_threshold=10) -> torch.Tensor:
    """
        images:  torch.tensor of dimensions (N, 3, H, W)
        return: torch.tensor of dimensions (N, H, W)
    """
    # Obtain the gray maps for each color pair (i.e. RG, BG, RB)
    gray_r_g = torch.abs(images[:, 0, ...] - images[:, 1, ...]) <= gray_threshold
    gray_g_b = torch.abs(images[:, 1, ...] - images[:, 2, ...]) <= gray_threshold
    gray_r_b = torch.abs(images[:, 0, ...] - images[:, 2, ...]) <= gray_threshold

    # Combine all gray maps into a single map
    gray_rgb = gray_r_g * gray_g_b * gray_r_b

    # Return the resulting gray map as a boolean map
    return gray_rgb


def gray_maps(images: torch.Tensor, gray_threshold=10) -> torch.Tensor:
    """
        images:  torch.tensor of dimensions (N, 3, H, W)
        return: torch.tensor of dimensions (N, 3, H, W), where N, H, W is repeated 3 times along the channel dimension
    """
    # Insert back the channel dimension and repeat the map three times
    gray_rgb = gray_map(images, gray_threshold).unsqueeze(1).repeat(1, 3, 1, 1)

    # Return the resulting gray maps as float maps
    return gray_rgb.float()


def make_grayscale(data: torch.Tensor) -> torch.Tensor:
    """
        Input:  tensor of shape (N, 3, H, W)
        Output: tensor of shape (N, 3, H, W)
        Combine three RGB maps to a single, repeated, grayscale map.
    """
    # Separate the RGB maps
    R, G, B = data[:, 0, ...], data[:, 1, ...], data[:, 2, ...]

    # Combine them using a weighted method
    return (R * 0.299 + G * 0.587 + B * 0.114).to(torch.uint8).unsqueeze(1).repeat(1, 3, 1, 1)


def process(data: torch.Tensor) -> torch.Tensor:
    # Format the data to be in [0, 1]
    return data.float() / 255.0


def unprocess(data: torch.Tensor) -> torch.Tensor:
    # Transform the data from [0, 1] to [0, 255]
    return torch.clamp(data * 255.0, 0.0, 255.0).to(torch.uint8)

