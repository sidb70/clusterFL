from logging import config
import random
from typing import Dict, List
import torch
from aggregation.strategies import load_aggregator
import numpy as np
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids


def tensorSum(tensors):
    return torch.sum(torch.stack(tensors))


class ClusterAlgorithm:
    def __init__(self, clusters: int = 5):
        self.clusters = clusters
        self.aggregator = load_aggregator("fedavg")

    def cluster(self, state_dicts: List[Dict[str, torch.Tensor]], *args, **kwargs):
        raise NotImplementedError


class PointWiseKMeans(ClusterAlgorithm):
    def __init__(self, clusters: int = 5, *args, **kwargs):
        super().__init__(clusters)

    def normalize(self, state_dicts: List[Dict[str, torch.Tensor]]):
        class Node:
            def __init__(self, tensor: torch.Tensor):
                self.values = [tensor]
                self.mean = None
                self.std = None

        noramlizationDict = {}

        # Prepare tensors for normalization
        for weight in state_dicts:
            for layer, tensor in weight.items():
                if layer in noramlizationDict:
                    noramlizationDict[layer].values.append(tensor)
                else:
                    noramlizationDict[layer] = Node(tensor)

        # Calculate normalization params
        for layer, node in noramlizationDict.items():
            noramlizationDict[layer].mean = self.tensorSum(
                list(noramlizationDict[layer].values)
            ) / len(noramlizationDict[layer].values)
            noramlizationDict[layer].std = torch.sqrt(
                sum(
                    (tensor - noramlizationDict[layer].mean) ** 2
                    for tensor in noramlizationDict[layer].values
                )
                / len(noramlizationDict[layer].values)
            )

        # Normalize tensors based on layer values accross classes
        normalizedWeights = []
        for weight in state_dicts:
            normalWeight = {}
            for layer, tensor in weight.items():
                normalWeight[layer] = (tensor - noramlizationDict[layer].mean) / (
                    noramlizationDict[layer].std + 1e-12
                )
            normalizedWeights.append(normalWeight)

        return normalizedWeights

    def cluster(self, state_dicts: List[Dict[str, torch.Tensor]], *args, **kwargs):
        k_iter = kwargs.get("k_iter", 10)
        normalize = kwargs.get("normalize", False)
        # Assume we have a fixed number clusters
        clusterList = [[] for i in range(self.clusters)]
        clients = [i for i in range(len(state_dicts))]

        km = KMedoids(n_clusters=self.clusters, metric="manhattan")

        # Assign intial centroids
        for clus in clusterList:
            randK = random.choice(clients)
            clus.append(randK)
            clients.remove(randK)

        # Calculate each clients distance from the centroid node
        weights = normalize(state_dicts) if normalize else state_dicts
        flattened_weights = []
        for state_dict in weights:
            flattened_weights.append(
                torch.cat([tensor.detach().cpu().flatten() for tensor in state_dict.values()]).numpy()
            )
        flattened_weights = np.array(flattened_weights)
        km.fit(flattened_weights)
        labels = km.labels_.tolist()
        cluster_list = [[] for _ in range(self.clusters)]
        for i, label in enumerate(labels):
            cluster_list[label].append(i)
        return cluster_list
        # print(type(weights[]))
        firstPass = True
        for i in range(k_iter):
            distanceMap = {}
            for client in clients:
                distanceMap[client] = []
                for clus in clusterList:
                    dist = 0
                    for layer, tensor in weights[client].items():
                        dist += torch.abs(
                            tensor
                            - (weights[clus[0]][layer] if firstPass else clus[0][layer])
                        ).sum()
                    distanceMap[client].append(dist)

            # Assign each cluster to the closest centroid
            for client, distances in distanceMap.items():
                clusterList[distances.index(min(distances))].append(client)

            # Reset Clients
            if i + 1 == k_iter:
                for clus in clusterList:
                    clus.pop(0)
            else:
                clients = [j for j in range(len(state_dicts))]
                centroids = []
                for clus in clusterList:
                    if not firstPass:
                        clus.pop(0)
                    centroids.append(
                        self.aggregator.aggregate(
                            [state_dicts[client] for client in clus]
                        )
                    )
                clusterList = [[centroid] for centroid in centroids]
                firstPass = False
        return clusterList


class FilterMatching(ClusterAlgorithm):
    def __init__(self, clusters: int = 5, **kwargs):
        super().__init__(clusters)
        distance_selection = kwargs.get("filter_distance", "max")
        if distance_selection == "max":
            self.distance_selection = max
        elif distance_selection == "mean":
            self.distance_selection = np.mean
        elif distance_selection == "min":
            self.distance_selection = min

    def filter_dist(self, filter1, filter2):
        return torch.dist(filter1, filter2, p=1)

    def compute_filter_dist(self, client1_filters, client2_filters):
        dist_mat = np.zeros((len(client1_filters), len(client2_filters)))
        for i, filter1 in enumerate(client1_filters):
            for j, filter2 in enumerate(client2_filters):
                dist_mat[i, j] = self.filter_dist(filter1, filter2)
        return dist_mat

    def match_filters(self, dist_mat):
        # match filters. once two filters are matched, remove them from the list
        matches = []
        for _ in range(dist_mat.shape[0]):
            i, j = np.unravel_index(np.argmin(dist_mat, axis=None), dist_mat.shape)
            dist = dist_mat[i, j]
            matches.append((i, j, dist))
            dist_mat[i, :] = np.inf
            dist_mat[:, j] = np.inf
        return matches

    def max_matching_dist(self, layer_weights):
        """
        Compute the maximum matching distance between all the clients for one layer
        Args:
            layer_weights: list of weights for one layer for all the clients
        Returns:
            dist_mat: matrix of distances between all the clients
        """
        dist_mat = np.zeros((len(layer_weights), len(layer_weights)))
        for i, client1_weights in enumerate(layer_weights):
            for j, client2_weights in enumerate(layer_weights):
                filter_dists = self.compute_filter_dist(
                    client1_weights, client2_weights
                )
                matches = self.match_filters(filter_dists)
                dist = self.distance_selection([match[2] for match in matches])
                dist_mat[i, j] = dist
        return dist_mat

    def cluster(self, state_dicts: List[Dict[str, torch.Tensor]], *args, **kwargs):
        km = KMeans(n_clusters=self.clusters)
        # take only first layer
        weights = [list(weight.values())[0] for weight in state_dicts]
        dist_mat = self.max_matching_dist(weights)
        km.fit(dist_mat)
        clusterList = [[] for i in range(self.clusters)]
        for i, label in enumerate(km.labels_):
            clusterList[label].append(i)
        return clusterList


def load_cluster_algorithm(name: str, *args, **kwargs):
    if name == "kmeans":
        return PointWiseKMeans(*args, **kwargs)
    elif name == "filter":
        return FilterMatching(*args, **kwargs)
    else:
        raise NotImplementedError
