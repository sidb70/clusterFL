from logging import config
import random
from typing import Dict, List
import torch
from aggregation.strategies import load_aggregator
import numpy as np
from sklearn.cluster import KMeans


torch.manual_seed(0)
random.seed(0) 

class ClusterDaddy:
    def __init__(self, weights: List[Dict[str, torch.Tensor]], clusters: int = 5):
        self.weights = weights
        self.clusters = clusters
        self.aggregator = load_aggregator("fedavg")

    def tensorSum(tensors):
        return torch.sum(torch.stack(tensors))

    def normalize(self):
        class Node:
            def __init__(self, tensor: torch.Tensor):
                self.values = [tensor]
                self.mean = None
                self.std = None

        noramlizationDict = {}

        # Prepare tensors for normalization
        for weight in self.weights:
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
        for weight in self.weights:
            normalWeight = {}
            for layer, tensor in weight.items():
                normalWeight[layer] = (tensor - noramlizationDict[layer].mean) / (
                    noramlizationDict[layer].std + 1e-12
                )
            normalizedWeights.append(normalWeight)

        return normalizedWeights
    
    def filter_dist(self, filter1, filter2):
        return torch.dist( filter1, filter2,p=1)
    def compute_filter_dist(self, client1_filters, client2_filters):
        dist_mat = np.zeros((len(client1_filters), len(client2_filters)))
        for i, filter1 in enumerate(client1_filters):
            for j, filter2 in enumerate(client2_filters):
                dist_mat[i, j] = self.filter_dist(filter1, filter2)
        return dist_mat
    def match_filters(self,dist_mat):
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
        '''
        Compute the maximum matching distance between all the clients for one layer
        Args:
            layer_weights: list of weights for one layer for all the clients
        Returns:
            dist_mat: matrix of distances between all the clients
        '''
        dist_mat = np.zeros((len(layer_weights), len(layer_weights)))
        for i, client1_weights in enumerate(layer_weights):
            for j, client2_weights in enumerate(layer_weights):
                filter_dists = self.compute_filter_dist(client1_weights, client2_weights)
                matches = self.match_filters(filter_dists)
                max_dist = max([match[2] for match in matches])
                dist_mat[i, j] = max_dist
        return dist_mat
    def _kmeans_dist_map(self, weights, clusterList):
        distanceMap = {}
        for client, weight in enumerate(weights):
            distanceMap[client] = []
            for clus in clusterList:
                dist = 0
                for layer, tensor in weight.items():
                    dist += torch.abs(
                        tensor - clus[0][layer]
                    ).sum()
                distanceMap[client].append(dist)
        return distanceMap
    def kMeans(self, k_iter: int = 10, normalize: bool = False):
        km = KMeans(n_clusters=self.clusters)
        # take only first layer
        weights = [list(weight.values())[0] for weight in self.weights]
        dist_mat = self.max_matching_dist(weights)
        km.fit(dist_mat)
        clusterList = [[] for i in range(self.clusters)]
        for i, label in enumerate(km.labels_):
            clusterList[label].append(i)
        return clusterList

        # Assume we have a fixed number clusters
        clusterList = [[] for i in range(self.clusters)]
        clients = [i for i in range(len(self.weights))]

        # Assign intial centroids
        for clus in clusterList:
            randK = random.choice(clients)
            clus.append(randK)
            clients.remove(randK)

        # Calculate each clients distance from the centroid node
        weights = normalize(self.weights) if normalize else self.weights
        # print(type(weights[]))
        firstPass = True
        for i in range(k_iter):
            distanceMap = self._kmeans_dist_map(weights, clusterList)
            # Assign each cluster to the closest centroid
            for client, distances in distanceMap.items():
                clusterList[distances.index(min(distances))].append(client)

            # Reset Clients
            if i + 1 == k_iter:
                for clus in clusterList:
                    clus.pop(0)
            else:
                clients = [j for j in range(len(self.weights))]
                centroids = []
                for clus in clusterList:
                    if not firstPass:
                        clus.pop(0)
                    centroids.append(
                        self.aggregator.aggregate(
                            [self.weights[client] for client in clus]
                        )
                    )
                clusterList = [[centroid] for centroid in centroids]
                firstPass = False
        return clusterList

    def bruteCluster(self, lamb: int = 1):
        # Calculate distances between each clients tensors
        # similarityMatrix = []
        # for ind,weight in enumerate(normalizedWeights):
        #     similarityMatrix.append([])
        #     for w2 in normalizedWeights:
        #         diff = 0
        #         for layer,tensor in weight.items():
        #             diff += torch.abs(tensor - w2[layer]).sum()
        #         similarityMatrix[ind].append(diff)

        return
