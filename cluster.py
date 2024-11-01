from logging import config
import random
from typing import Dict, List
import torch

from aggregation.strategies import load_aggregator

class ClusterDaddy():
  def __init__(self, weights: List[Dict[str, torch.Tensor]], clusters: int = 5):
      self.weights = weights
      self.clusters = clusters
      self.aggregator = load_aggregator('fedavg')

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
          for layer,tensor in weight.items():
              if layer in noramlizationDict:
                  noramlizationDict[layer].values.append(tensor)
              else:
                  noramlizationDict[layer] = Node(tensor)

      # Calculate normalization params
      for layer, node in noramlizationDict.items():
          noramlizationDict[layer].mean = self.tensorSum(list(noramlizationDict[layer].values)) / len(noramlizationDict[layer].values)
          noramlizationDict[layer].std = torch.sqrt(sum((tensor - noramlizationDict[layer].mean) ** 2 for tensor in noramlizationDict[layer].values) / len(noramlizationDict[layer].values))

      # Normalize tensors based on layer values accross classes
      normalizedWeights = []
      for weight in self.weights:
          normalWeight = {}
          for layer,tensor in weight.items():
              normalWeight[layer] = (tensor - noramlizationDict[layer].mean) / (noramlizationDict[layer].std + 1e-12)
          normalizedWeights.append(normalWeight)

      return normalizedWeights

  def kMeans(self, k_iter: int = 10, normalize: bool = False):
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
      print(type(weights[0]))
      print(weights[0])
      exit(0)
      #print(type(weights[]))
      firstPass = True
      for i in range(k_iter):
        distanceMap = {}
        for client in clients:
            distanceMap[client] = []
            for clus in clusterList:
                dist = 0
                for layer, tensor in weights[client].items():
                    dist += torch.abs(tensor - (weights[clus[0]][layer] if firstPass else clus[0][layer])).sum()
                distanceMap[client].append(dist)
        
        # Assign each cluster to the closest centroid
        for client,distances in distanceMap.items():
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
                centroids.append(self.aggregator.aggregate([self.weights[client] for client in clus]))
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
