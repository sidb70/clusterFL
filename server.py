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

    def aggregate_gradients(self, gradients: List[Dict[str, torch.Tensor]]):
        return self.aggregator.aggregate(gradients)


    def get_client_data(self, client_id, batch_size):
        num_clients = len(self.clients)
        samples_per_client = len(self.global_train_set) // num_clients
        start = client_id * samples_per_client % len(self.global_train_set)
        end = start + samples_per_client

        if end > len(self.global_train_set):
            end = len(self.global_train_set)
        
        indices = list(range(start, end))
        np.random.shuffle(indices)
        subset_data = Subset(self.global_train_set, indices)
        if end - start <samples_per_client:
            subset_data += Subset(self.global_train_set, list(range(0, samples_per_client - (end-start))))
        return DataLoader(subset_data, batch_size=batch_size)

    def fl_round(self):
        num_clients = len(self.clients)
        num_sampled = max(1, int(self.config.get('client_sample_rate', 1) * num_clients))
        sampled_clients = random.sample(self.clients, num_sampled)
        gradients = []  
        criterion = nn.CrossEntropyLoss()
        for client in sampled_clients:
            client_loader = self.get_client_data(client.id, batch_size=32) ## TODO: change this to selected classes
            grad = client.compute_gradients(self.global_model, client_loader, criterion)
            gradients.append(grad)
        aggregated_gradients = self.aggregate_gradients(gradients)
        self.global_model.load_state_dict(self.global_model.state_dict() - self.lr*aggregated_gradients)

    def evaluate(self):
        for client in self.clients:
            loss, acc = client.evaluate(self.global_model, self.global_test_set, nn.CrossEntropyLoss())
            print(f"Client {client.id} - Loss: {loss}, Accuracy: {acc}")
