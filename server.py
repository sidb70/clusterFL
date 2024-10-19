import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Dict
from models.loader import load_model    
from client import Client
from datasets.dataloader import load_global_dataset, load_selected_classes
from aggregation.strategies import load_aggregator
import random

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
    
    def fl_round(self):
        num_clients = len(self.clients)
        num_sampled = max(1, int(self.config.get('client_sample_rate', 1) * num_clients))
        sampled_clients = random.sample(self.clients, num_sampled)
        gradients = []  
        criterion = nn.CrossEntropyLoss()
        for client in sampled_clients:
            client_data = self.global_train_set ## TODO: change this to selected classes
            client_loader = DataLoader(client_data, batch_size=32, shuffle=True)
            grad = client.compute_gradients(self.global_model, client_loader, criterion)
            gradients.append(grad)
        aggregated_gradients = self.aggregate_gradients(gradients)
        self.global_model.load_state_dict(self.global_model.state_dict() - self.lr*aggregated_gradients)
