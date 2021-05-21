from scipy.optimize import linear_sum_assignment
from torch.nn.functional import interpolate
from src.KMeans import KMeans_cosine
from torch.nn.functional import pad
from src.Config.Args import args
import torch


def upsample(feature_map: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
        Upsample the feature map from (N, C, H, W) to (N, C, 240, 320).
    """
    # Fetch the current feature map dimensions
    n, c, h, w = feature_map.shape
        
    # Upsample the feature map map to the desired resolution
    feature_map = interpolate(feature_map, scale_factor=H / h, mode='nearest')
    
    # Return the output
    return feature_map


def linearize(feature_map: torch.Tensor) -> torch.Tensor:
    """
        Reshape the feature map from an image of size (1, C, H, W) to (HxW, C).
    """
    # Extract individual shapes from the feature map
    N, C, H, W = feature_map.shape

    # Permute dimension from (1, C, H, W) to (1, H, W, C)
    feature_map = feature_map.permute(0, 2, 3, 1)

    # Reshape the feature maps from (1, H, W, C) to (HxW, C)
    feature_map = feature_map.reshape((H * W, C))

    # Return the output
    return feature_map


def clusterize(features: torch.Tensor, **kwargs) -> torch.Tensor:
    """
        Apply K-Means to features vector. The input is of size 
        (HxW, C) and the algorithm outputs an array of (HxW) labels. 
    """
    # Make the feature map contiguous as per K-Means requirement
    features = features.contiguous()

    # Clusterize the feature map and obtain the labels
    labels = KMeans_cosine(features, **kwargs)

    # Replace the feature maps with the labels
    features = labels

    # Return the output
    return features


def segregate(labels: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
        Segregate the labels into multiple masks.
        This functions takes an array of labels as input,
        reshapes them to an image of size (H, W). Then it 
        extracts multiple masks by using the unique labels.
        It outputs a tensor of size (UNIQUE_LEN, H, W).
    """
    # Extract unique labels from the feature map
    unique_labels = labels.unique()
    
    # Prepare the unique labels for broadcast segmentation
    unique_labels = unique_labels.reshape((-1, 1, 1))

    # Reshape the labels to a map of size (H, W)
    labels_map = labels.reshape((H, W))

    # Prepare the feature map for broadcast segmentation
    labels_map = labels_map.unsqueeze(0)

    # Segment the feature map by the unique labels
    masks = unique_labels == labels_map

    # Return the output
    return masks


def jaccard_score_matrix(true_masks: torch.Tensor, pred_masks: torch.Tensor) -> torch.Tensor:
    """
        For each pair of masks obtained by the cartezian
        product of true_masks and pred_masks, calculate
        the Intersection Over Union Metric, also known as
        the Jaccard Score.

        Assemble the scores in a matrix of size (TML, PML),
        where aij = jaccard_score(TM[i], PM[j]).

        If TM and PM differ in length:
            1. If PML < TML then zeros will be appended to PM,
        penalizing the score.
            2. If PML > TML the additional masks will be ignored
        later, for now they will be ignored.
    """
    # Prepare masks for broadcasting to create a matrix of scores
    TM, PM = true_masks.unsqueeze(1), pred_masks.unsqueeze(0)

    # Calculate the overlapping areas
    intersections = TM.logical_and(PM)

    # Calculate the reunited ares
    unions = TM.logical_or(PM)

    # Compute the Jaccard Matrix
    jaccard_score_matrix = intersections.sum(dim=(2, 3)) / unions.sum(dim=(2, 3))

    # Extract the new matrix dimensions
    TML, PML = jaccard_score_matrix.shape

    # Add padding to the Jaccard Matrix according to TML and PML sizes
    jaccard_score_matrix = pad(jaccard_score_matrix, pad=(0, max(0, TML - PML)))
    
    # Return the output
    return jaccard_score_matrix


def intersection_over_union(jaccard_matrix: torch.Tensor) -> torch.Tensor:
    """
        From a Jaccard Matrix compute the average IoU scores,
        by selecting the best mask pairs scores using a hungarian 
        matching algorithm.
    """
    # Calculate the most likely mask pairs according to the Jaccard Matrix
    rows, cols = linear_sum_assignment(-jaccard_matrix.cpu())

    # Send the results back to the GPU
    rows = torch.Tensor(rows).to(torch.long).to(args.device)
    cols = torch.Tensor(cols).to(torch.long).to(args.device)

    # Extract the optimum matching scores
    scores = jaccard_matrix[rows, cols]

    # Compute the IoU mean score
    iou = scores.mean()

    # Return the output
    return iou

