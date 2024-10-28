import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from typing import List, Dict
from models.loader import load_model    
from client import Client
from datasets.dataloader import load_global_dataset, load_selected_classes
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
        


    def aggregate(self, gradients: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return self.aggregator.aggregate(gradients)

    def get_client_data(self, client_id, batch_size):
        subset_train_data = Subset(self.global_train_set, self.client_train_indices[client_id])
        subset_test_data = Subset(self.global_test_set, self.client_test_indices[client_id])
        return DataLoader(subset_train_data, batch_size=batch_size), DataLoader(subset_test_data, batch_size=batch_size)

    def fl_round(self):
        num_clients = len(self.clients)
        num_sampled = max(1, int(self.config.get('client_sample_rate', 1) * num_clients))
        sampled_clients = random.sample(self.clients, num_sampled)
        updated_models = []  
        criterion = nn.CrossEntropyLoss()
        for client in sampled_clients:
            client_train_loader, _ = self.get_client_data(client.id, batch_size=32) ## TODO: change this to selected classes
            client_model = deepcopy(self.global_model)
            optimizer = torch.optim.SGD(client_model.parameters(), lr=self.lr)
            updated_model = client.train(client_model, client_train_loader, criterion, optimizer, self.local_epochs)
            updated_models.append(updated_model.state_dict())
        aggregated_models_state_dict = self.aggregate(updated_models)
        self.global_model.load_state_dict(aggregated_models_state_dict)
        
    def evaluate(self, batch_size: int = 32):
        accuracies = []
        losses = []
        for client in self.clients:
            _, test_loader = self.get_client_data(client.id, batch_size=batch_size)
            loss, acc = client.evaluate(self.global_model, test_loader, nn.CrossEntropyLoss())
            accuracies.append(acc)
            losses.append(loss)
        print(f"Average Accuracy: {sum(accuracies)/len(accuracies)}, Average Loss: {sum(losses)/len(losses)}")