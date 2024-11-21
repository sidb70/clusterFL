from logging import config
import random
from typing import Dict, List
import torch
from aggregation.strategies import load_aggregator
from operator import itemgetter

torch.manual_seed(0)
random.seed(0) 

class ClusterDaddy:
    def __init__(self, weights: List[Dict[str, torch.Tensor]], clusters: int = 5):
        self.weights = weights
        self.clusters = clusters
        self.aggregator = load_aggregator("fedavg")
        self.mode = 'L1'

    def tensorSum(tensors):
        return torch.sum(torch.stack(tensors))
    
    def filter_dist(self,filter1, filter2):
        return torch.dist(filter1.float(), filter2.float(),p=1)
    
    def l1norm(self,t1,t2):
        return torch.abs(t1-t2).sum()

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
    
    def _kmeans_dist_map(self, weights, clusterList, firstPass):
        distanceMap = {}
        for client, weight in enumerate(weights):
            distanceMap[client] = []
            for clus in clusterList:
                dist = 0
                for layer, tensor in weight.items():
                    tensor2 = weights[clus[0]][layer] if firstPass else clus[0][layer]
                    dist += self.l1norm(tensor, tensor2) if self.mode == 'L1' else self.layerDistance(tensor, tensor2)
                distanceMap[client].append(dist)
        return distanceMap
    
    def kMeans(self, k_iter: int = 10, normalize: bool = False, mode: str = "L1"):
        # Assume we have a fixed number clusters
        clusterList = [[] for i in range(self.clusters)]
        clients = [i for i in range(len(self.weights))]
        self.mode = mode

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
            distanceMap = self._kmeans_dist_map(weights, clusterList, firstPass)
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

    def layerDistance(self,client1_filters,client2_filters):
        filter_matches = []
        removed = set()
        for filter1 in client1_filters:
            min_dist = float('inf')
            closest_filter = None
            for filter2 in client2_filters:
                if filter2 in removed:
                    continue
                dist = self.filter_dist(filter1, filter2)
                if dist < min_dist:
                    min_dist = dist
                    closest_filter = filter2
            filter_matches.append((filter1, closest_filter, min_dist))
            removed.add(filter2)

        sorted_dists = sorted(filter_matches,key=itemgetter(2))
        largest_dist = sorted_dists[-1][2]
        return largest_dist

    