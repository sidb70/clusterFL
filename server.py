import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from typing import List, Dict
from models.loader import load_model    
from client import Client
from datasets.dataloader import load_global_dataset, load_selected_classes
from aggregation.strategies import load_aggregator
import random
import numpy as np

DEVICES = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())] \
    if torch.cuda.is_available() else [torch.device('cpu')]
class Server:
    def __init__(self,config):
        self.config = config
        self.global_model = load_model(config['model'])
        self.global_train_set, self.global_test_set = load_global_dataset(config['dataset'])
        self.create_clients(config['clients'])
        self.aggregator = load_aggregator(config['aggregator'])
        self.lr = config['lr']

    def create_clients(self, num_clients):
        self.clients = [Client(i, DEVICES[i % len(DEVICES)]) for i in range(num_clients)]
        # assign selected indices to clients
        num_clients = len(self.clients)
        samples_per_client = len(self.global_train_set) // num_clients
        self.client_train_indices = {i: [] for i in range(num_clients)}
        self.client_test_indices = {i: [] for i in range(num_clients)}
        for client_id in range(num_clients):
            train_start = client_id * samples_per_client % len(self.global_train_set)
            train_end = train_start + samples_per_client
            self.client_train_indices[client_id] = list(range(train_start, train_end))
            test_start = client_id * samples_per_client % len(self.global_test_set)
            test_end = test_start + samples_per_client
            self.client_test_indices[client_id] = list(range(test_start, test_end))
            if train_end > len(self.global_train_set):
                self.client_train_indices[client_id] += list(range(0, train_end - len(self.global_train_set)))
                self.client_test_indices[client_id] += list(range(0, test_end - len(self.global_train_set)))


    def aggregate_gradients(self, gradients: List[Dict[str, torch.Tensor]]):
        return self.aggregator.aggregate(gradients)

    def get_client_data(self, client_id, batch_size):
        subset_train_data = Subset(self.global_train_set, self.client_train_indices[client_id])
        subset_test_data = Subset(self.global_test_set, self.client_test_indices[client_id])
        return DataLoader(subset_train_data, batch_size=batch_size), DataLoader(subset_test_data, batch_size=batch_size)

    def fl_round(self):
        num_clients = len(self.clients)
        num_sampled = max(1, int(self.config.get('client_sample_rate', 1) * num_clients))
        sampled_clients = random.sample(self.clients, num_sampled)
        gradients = []  
        criterion = nn.CrossEntropyLoss()
        for client in sampled_clients:
            client_train_loader, _ = self.get_client_data(client.id, batch_size=32) ## TODO: change this to selected classes
            grad = client.compute_gradients(self.global_model, client_train_loader, criterion)
            gradients.append(grad)
        aggregated_gradients = self.aggregate_gradients(gradients)
        state_dict = self.global_model.state_dict()
        for key in state_dict.keys():
            state_dict[key] -= self.lr * aggregated_gradients[key]
        self.global_model.load_state_dict(state_dict)

    def evaluate(self, batch_size: int = 32):
        for client in self.clients:
            _, test_loader = self.get_client_data(client.id, batch_size=batch_size)
            loss, acc = client.evaluate(self.global_model, test_loader, nn.CrossEntropyLoss())
            print(f"Client {client.id} - Loss: {loss}, Accuracy: {acc}")
