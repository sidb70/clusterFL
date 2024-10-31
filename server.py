import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from typing import List, Dict
from models.loader import load_model    
from client import Client
from datasets.dataloader import load_global_dataset, create_clustered_dataset
from aggregation.strategies import load_aggregator
import random
from copy import deepcopy

DEVICES = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())] \
    if torch.cuda.is_available() else [torch.device('cpu')]



class Server:
    def __init__(self,config):
        self.config = config
        self.global_model = load_model(config['model'])
        self.global_train_set, self.global_test_set = load_global_dataset(config['dataset'])

        self.create_clients(config['clients'])
        self.num_clusters = config['num_clusters']
        self.clustering = self.cluster([[] for _ in range(len(self.clients))]) # initial clustering TEMP
        self.cluster_models = [deepcopy(self.global_model.state_dict()) for _ in range(self.num_clusters)]
        self.clustered_train_sets = create_clustered_dataset(self.global_train_set, config['num_clusters'], config['cluster_split_type'])
        self.clustered_test_sets = create_clustered_dataset(self.global_test_set, config['num_clusters'], config['cluster_split_type'])
        self.aggregator = load_aggregator(config['aggregator'])
        self.lr = config['lr']
        self.local_epochs = config['local_epochs']

    def create_clients(self, num_clients):
        self.clients = [Client(i, DEVICES[i % len(DEVICES)]) for i in range(num_clients)]
        # assign selected indices to clients
        num_clients = len(self.clients)
        train_samples_per_client = len(self.global_train_set) // num_clients
        test_samples_per_client = len(self.global_test_set) // num_clients
        self.client_train_indices = {i: [] for i in range(num_clients)}
        self.client_test_indices = {i: [] for i in range(num_clients)}
        for client_id in range(num_clients):
            train_start = client_id * train_samples_per_client % len(self.global_train_set)
            train_end = train_start + train_samples_per_client
            test_start = client_id * test_samples_per_client % len(self.global_test_set)
            test_end = test_start + test_samples_per_client
            if train_end > len(self.global_train_set):
                self.client_train_indices[client_id] = list(range(train_start, len(self.global_train_set)))
                self.client_train_indices[client_id] += list(range(0, train_end - len(self.global_train_set)))
            else:
                self.client_train_indices[client_id] = list(range(train_start, train_end))
            if test_end > len(self.global_test_set):
                self.client_test_indices[client_id] = list(range(test_start, len(self.global_test_set)))
                self.client_test_indices[client_id] += list(range(0, test_end - len(self.global_test_set)))
            else:
                self.client_test_indices[client_id] = list(range(test_start, test_end))
            print(f"Client {client_id} - train start: {train_start}, train end: {train_end}, test start: {test_start}, test end: {test_end}")

    def cluster(self, weights: List[Dict[str, torch.Tensor]]):
        clusters = self.num_clusters
        return [i % clusters for i in range(len(weights))]
        class Node:
            def __init__(self, tensor: torch.Tensor):
                self.values = [tensor]
                self.mean = None
                self.std = None

        noramlizationDict = {}

        # Prepare tensors for normalization
        for weight in weights:
            for layer,tensor in weight.items():
                if layer in noramlizationDict:
                    noramlizationDict[layer].values.append(tensor)
                else:
                    noramlizationDict[layer] = Node(tensor)

        # Calculate normalization params
        for layer, node in noramlizationDict.items():
            noramlizationDict[layer].mean = sum(noramlizationDict[layer].values) / len(noramlizationDict[layer].values)
            noramlizationDict[layer].std = torch.sqrt(sum((tensor - noramlizationDict[layer].mean) ** 2 for tensor in noramlizationDict[layer].values) / len(noramlizationDict[layer].values))

        # Normalize tensors based on layer values accross classes
        normalizedWeights = []
        for weight in weights:
            normalWeight = {}
            for layer,tensor in weight.items():
                normalWeight[layer] = (tensor - noramlizationDict[layer].mean) / (normalWeight[layer].std + 1e-12)
            normalizedWeights.append(normalWeight)
        
        # Calculate distances between each clients tensors
        # similarityMatrix = []
        # for ind,weight in enumerate(normalizedWeights):
        #     similarityMatrix.append([])
        #     for w2 in normalizedWeights:
        #         diff = 0
        #         for layer,tensor in weight.items():
        #             diff += torch.abs(tensor - w2[layer]).sum()
        #         similarityMatrix[ind].append(diff)
        
        # Assume we have a fixed number clusters
        # numPerCluster = clusters / len(weights) -> if we want to specify how many clusters there should be
        clusterList = [[] for i in clusters]
        clients = [i for i in range(len(weights))]

        # Assign intial centroids
        for clus in clusterList:
            randK = random.choice(clients)
            clus.append(randK)
            clients.remove(randK)

        # Calculate each clients distance from the centroid node
        distanceMap = {}
        for client in clients:
            distanceMap[client] = []
            for clus in clusterList:
                dist = 0
                for layer, tensor in weights[client]:
                    dist += torch.abs(tensor - weights[clus[0]][layer]).sum()
                distanceMap[client].append(dist)
        
        # Assign each cluster to the closest centroid
        for client,distances in distanceMap.items():
            clus[distances.index(min(distances))].append(client)

        return clusterList

    def aggregate(self, gradients: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return self.aggregator.aggregate(gradients)

    def get_client_data(self, client_id, batch_size):
        # subset_train_data = Subset(self.global_train_set, self.client_train_indices[client_id])
        # subset_test_data = Subset(self.global_test_set, self.client_test_indices[client_id])
        # return DataLoader(subset_train_data, batch_size=batch_size), DataLoader(subset_test_data, batch_size=batch_size)
        assigned_cluster = self.clustering[client_id]
        subset_train_data = Subset(self.clustered_train_sets[assigned_cluster], self.client_train_indices[client_id])
        subset_test_data = Subset(self.clustered_test_sets[assigned_cluster], self.client_test_indices[client_id])
        client_train_loader = DataLoader(subset_train_data, batch_size=batch_size)
        client_test_loader = DataLoader(subset_test_data, batch_size=batch_size)
        return client_train_loader, client_test_loader

    def fl_round(self):
        num_clients = len(self.clients)
        num_sampled = max(1, int(self.config.get('client_sample_rate', 1) * num_clients))
        sampled_clients = random.sample(self.clients, num_sampled)
        updated_models = [[] for _ in range(self.num_clusters)]
        criterion = nn.CrossEntropyLoss()
        for client in sampled_clients:
            client_train_loader, _ = self.get_client_data(client.id, batch_size=32) ## TODO: change this to selected classes
            cluster_id = self.clustering[client.id]
            client_state_dict= self.cluster_models[cluster_id]
            client_model = load_model(self.config['model'])
            client_model.load_state_dict(client_state_dict)
            optimizer = torch.optim.SGD(client_model.parameters(), lr=self.lr)
            updated_model = client.train(client_model, client_train_loader, criterion, optimizer, self.local_epochs)
            updated_models[cluster_id].append(updated_model.state_dict())
        for cluster_id in range(self.num_clusters):
            self.cluster_models[cluster_id] = self.aggregate(updated_models[cluster_id])
        
    def evaluate(self, batch_size: int = 32):
        accuracies = []
        losses = []
        for client in self.clients:
            _, test_loader = self.get_client_data(client.id, batch_size=batch_size)
            loss, acc = client.evaluate(self.global_model, test_loader, nn.CrossEntropyLoss())
            accuracies.append(acc)
            losses.append(loss)
        print(f"Average Accuracy: {sum(accuracies)/len(accuracies)}, Average Loss: {sum(losses)/len(losses)}")