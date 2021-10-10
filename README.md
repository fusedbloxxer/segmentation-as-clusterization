# SegmentationAsClusterization

Goal: Unsupervised segmentation of simple images

1. Train a simple U-net model for supervised segmentation.
2. To extract useful features train an autoencoder U-net model that reconstructs an image.
3. Cluster the intermediate features of the model.
4. Use cluster assignments as weak segments
5. Compare performance at different depths in the model

Bonus:
Learn weak clusterization (Slot attention)

![research results](./data/results/research_results.jpg)
