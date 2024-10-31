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
        self.global_train_set, self.global_test_set = load_global_dataset(config['dataset'])
        self.clustered_train_sets = create_clustered_dataset(self.global_train_set, config['num_clusters'], config['cluster_split_type'])
        self.clustered_test_sets = create_clustered_dataset(self.global_test_set, config['num_clusters'], config['cluster_split_type'])
        

        self.num_clients = config['clients']
        self.num_clusters = config['num_clusters']
        self.lr = config['lr']
        self.local_epochs = config['local_epochs']
        
        self.clients_to_clusters = self.cluster([[] for _ in range(self.num_clients)]) # initial clustering TEMP
        self.clusters_to_clients = {}
        for i, cluster in enumerate(self.clients_to_clusters):
            if cluster not in self.clusters_to_clients:
                self.clusters_to_clients[cluster] = []
            self.clusters_to_clients[cluster].append(i)
        self.create_clients(self.num_clients)
        initial_model = load_model(config['model'])
        self.cluster_models = [deepcopy(initial_model.state_dict()) for _ in range(self.num_clusters)]
        self.aggregator = load_aggregator(config['aggregator'])
        

    def create_clients(self, num_clients):
        self.clients = []
        self.client_train_indices = []  
        self.client_test_indices = []

    def aggregate(self, gradients: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return self.aggregator.aggregate(gradients)

    def get_client_data(self, client_id, batch_size):
        # subset_train_data = Subset(self.global_train_set, self.client_train_indices[client_id])
        # subset_test_data = Subset(self.global_test_set, self.client_test_indices[client_id])
        # return DataLoader(subset_train_data, batch_size=batch_size), DataLoader(subset_test_data, batch_size=batch_size)
        assigned_cluster = self.clients_to_clusters[client_id]
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
        
        for client in sampled_clients:
            client_train_loader, _ = self.get_client_data(client.id, batch_size=32) ## TODO: change this to selected classes
            cluster_id = self.clients_to_clusters[client.id]
            client_state_dict= self.cluster_models[cluster_id]
            client_model = load_model(self.config['model'])
            client_model.load_state_dict(client_state_dict)

            criterion = nn.CrossEntropyLoss()
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
            cluster_model = load_model(self.config['model'])
            cluster_model.load_state_dict(self.cluster_models[client.cluster_assignment])
            loss, acc = client.evaluate(cluster_model, test_loader, nn.CrossEntropyLoss())
            accuracies.append(acc)
            losses.append(loss)
        print(f"Average Accuracy: {sum(accuracies)/len(accuracies)}, Average Loss: {sum(losses)/len(losses)}")