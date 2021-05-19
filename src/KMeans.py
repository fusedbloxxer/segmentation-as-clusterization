from pykeops.torch import LazyTensor
from src.Config.Args import args
import torch

# Define the KMeans algorithm using cosine similarity
def KMeans_cosine(x, K=10, n_init=10, max_iter=300, tol=1e-4):
    """Implements Lloyd's algorithm for the Cosine similarity metric."""
    # Number of samples, dimension of the ambient space
    N, D = x.shape

    # Normalize the input for the cosine similarity:
    x = torch.nn.functional.normalize(x, dim=1, p=2)

    # Get the unique data points
    unique_samples = x.unique(dim=0)

    # Initialize the input data points
    x_i = LazyTensor(x.view(N, 1, D))  # (N, 1, D) samples

    # Retain the best results across different trials
    best = {'labels': None, 'similarity': None, 'centroids': None}

    # Run the K-means algorithm with different centroid seeds
    for trial in range(n_init):
        # Random initialization for the centroids from the unique data points
        c = unique_samples[torch.randperm(unique_samples.shape[0])[:K], :].clone()

        # Initialize the centroids
        c_j = LazyTensor(c.view(1, K, D))  # (1, K, D) centroids

        # Run the K-means loop:
        # - x  is the (N, D) point cloud,
        # - cl is the (N,) vector of class labels
        # - c  is the (K, D) cloud of cluster centroids
        for iter in range(max_iter):
            # Retain old centroids
            c_old = c.clone()

            # E step: assign points to the closest cluster -------------------------
            S_ij = x_i | c_j  # (N, K) symbolic Gram matrix of dot products
            cl = S_ij.argmax(dim=1).long().view(-1)  # Points -> Nearest cluster

            # M step: update the centroids to the normalized cluster average: ------
            # Compute the sum of points per cluster:
            c.zero_()
            c.scatter_add_(0, cl[:, None].repeat(1, D), x)

            # Normalize the centroids, in place:
            c[:] = torch.nn.functional.normalize(c, dim=1, p=2)

            # Compute differences between the old and new centrois
            error = torch.mean((c - c_old) ** 2)

            # Check for convergence
            if error <= tol:
                # Caclulate the similarity
                sim = S_ij.max(dim=1).view(-1).sum()
                
                # Compare with previous results
                if best['similarity'] is None or sim > best['similarity']:
                    # Retain the current new best
                    best['similarity'] = sim
                    best['centroids'] = c
                    best['labels'] = cl
                
                # Exit the current run
                break

    return best['labels']