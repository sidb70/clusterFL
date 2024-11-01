import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from typing import List, Dict
from models.loader import load_model
from client import Client
from cluster import ClusterDaddy
from datasets.dataloader import load_global_dataset, create_clustered_dataset
from aggregation.strategies import load_aggregator
import random
from copy import deepcopy

DEVICES = (
    [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    if torch.cuda.is_available()
    else [torch.device("cpu")]
)


class Server:
    def __init__(self, config):
        self.config = config
        self.global_train_set, self.global_test_set = load_global_dataset(
            config["dataset"]
        )
        self.clustered_train_sets = create_clustered_dataset(
            self.global_train_set, config["num_clusters"], config["cluster_split_type"]
        )
        self.clustered_test_sets = create_clustered_dataset(
            self.global_test_set, config["num_clusters"], config["cluster_split_type"]
        )

        self.num_clients = config["clients"]
        self.num_clusters = config["num_clusters"]
        assert self.num_clusters <= self.num_clients
        self.lr = config["lr"]
        self.local_epochs = config["local_epochs"]
        self.initial_epochs = config["initial_epochs"]

        self.clients_to_clusters = self.cluster(
            [[] for _ in range(self.num_clients)]
        )  # initial clustering TEMP
        self.clusters_to_clients = {}
        for i, cluster in enumerate(self.clients_to_clusters):
            if cluster not in self.clusters_to_clients:
                self.clusters_to_clients[cluster] = []
            self.clusters_to_clients[cluster].append(i)
        self.create_clients(self.num_clients)
        initial_model = load_model(config["model"])
        self.cluster_models = [
            deepcopy(initial_model.state_dict()) for _ in range(self.num_clusters)
        ]
        self.aggregator = load_aggregator(config["aggregator"])

    def create_clients(self, num_clients):
        self.clients = []
        self.client_train_indices = []
        self.client_test_indices = []

        for i in range(num_clients):
            cluster_assignment = self.clients_to_clusters[i]
            self.clients.append(
                Client(
                    id=i,
                    device=DEVICES[i % len(DEVICES)],
                    cluster_assignment=cluster_assignment,
                )
            )

            cluster_train_data = self.clustered_train_sets[cluster_assignment]
            cluster_test_data = self.clustered_test_sets[cluster_assignment]

            client_in_cluster_id = self.clusters_to_clients[cluster_assignment].index(i)

            num_clients_in_cluster = len(self.clusters_to_clients[cluster_assignment])
            train_samples_per_client = len(cluster_train_data) // num_clients_in_cluster
            test_samples_per_client = len(cluster_test_data) // num_clients_in_cluster
            train_start = (
                client_in_cluster_id
                * train_samples_per_client
                % len(cluster_train_data)
            )
            train_end = train_start + train_samples_per_client
            test_start = (
                client_in_cluster_id * test_samples_per_client % len(cluster_test_data)
            )
            test_end = test_start + test_samples_per_client
            if train_end > len(cluster_train_data):
                client_train_indices = list(range(train_start, len(cluster_train_data)))
                client_train_indices += list(
                    range(0, train_end - len(cluster_train_data))
                )
            else:
                client_train_indices = list(range(train_start, train_end))
            if test_end > len(cluster_test_data):
                client_test_indices = list(range(test_start, len(cluster_test_data)))
                client_test_indices += list(range(0, test_end - len(cluster_test_data)))
            else:
                client_test_indices = list(range(test_start, test_end))
            self.client_train_indices.append(client_train_indices)
            self.client_test_indices.append(client_test_indices)
        for i, (cluster_num, clients_list) in enumerate(
            self.clusters_to_clients.items()
        ):
            print(f"Cluster {cluster_num} has {len(clients_list)} clients")
            for client in clients_list:
                print(
                    f"client {client} train start {self.client_train_indices[client][0]} end {self.client_train_indices[client][-1]} device {self.clients[client].device}"
                )

    def aggregate(
        self, gradients: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        return self.aggregator.aggregate(gradients)

    def get_client_data(self, client_id, batch_size):
        # subset_train_data = Subset(self.global_train_set, self.client_train_indices[client_id])
        # subset_test_data = Subset(self.global_test_set, self.client_test_indices[client_id])
        # return DataLoader(subset_train_data, batch_size=batch_size), DataLoader(subset_test_data, batch_size=batch_size)
        assigned_cluster = self.clients_to_clusters[client_id]
        subset_train_data = Subset(
            self.clustered_train_sets[assigned_cluster],
            self.client_train_indices[client_id],
        )
        subset_test_data = Subset(
            self.clustered_test_sets[assigned_cluster],
            self.client_test_indices[client_id],
        )
        client_train_loader = DataLoader(subset_train_data, batch_size=batch_size)
        client_test_loader = DataLoader(subset_test_data, batch_size=batch_size)
        return client_train_loader, client_test_loader

    def cluster(self, state_dicts: List[Dict[str, torch.Tensor]]) -> List[List[int]]:
        cluster_daddy = ClusterDaddy(state_dicts, clusters=self.num_clusters)
        return cluster_daddy.kMeans(k_iter=40)

    def initial_cluster_rounds(self):
        state_dicts = []
        for client in self.clients:
            client_train_loader, _ = self.get_client_data(client.id, batch_size=32)
            client_model = load_model(self.config["model"])
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(client_model.parameters(), lr=self.lr)
            updated_model = client.train(
                client_model,
                client_train_loader,
                criterion,
                optimizer,
                self.initial_epochs,
            )
            state_dicts.append(updated_model.state_dict())
        clusters = self.cluster(state_dicts)
        for i, clients in enumerate(clusters):
            self.clusters_to_clients[i] = clients
            for client in clients:
                self.clients_to_clusters[client] = i

    def fl_round(self):
        num_clients = len(self.clients)
        num_sampled = max(
            1, int(self.config.get("client_sample_rate", 1) * num_clients)
        )
        sampled_clients = random.sample(self.clients, num_sampled)
        updated_models = [[] for _ in range(self.num_clusters)]

        for client in sampled_clients:
            client_train_loader, _ = self.get_client_data(
                client.id, batch_size=32
            )  ## TODO: change this to selected classes
            cluster_id = self.clients_to_clusters[client.id]
            client_state_dict = self.cluster_models[cluster_id]
            client_model = load_model(self.config["model"])
            client_model.load_state_dict(client_state_dict)

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(client_model.parameters(), lr=self.lr)
            updated_model = client.train(
                client_model,
                client_train_loader,
                criterion,
                optimizer,
                self.local_epochs,
            )
            updated_models[cluster_id].append(updated_model.state_dict())
        for cluster_id in range(self.num_clusters):
            self.cluster_models[cluster_id] = self.aggregate(updated_models[cluster_id])

    def evaluate(self, batch_size: int = 32):
        accuracies = []
        losses = []
        for client in self.clients:
            _, test_loader = self.get_client_data(client.id, batch_size=batch_size)
            cluster_model = load_model(self.config["model"])
            cluster_model.load_state_dict(
                self.cluster_models[client.cluster_assignment]
            )
            loss, acc = client.evaluate(
                cluster_model, test_loader, nn.CrossEntropyLoss()
            )
            accuracies.append(acc)
            losses.append(loss)
        print(
            f"Average Accuracy: {sum(accuracies)/len(accuracies)}, Average Loss: {sum(losses)/len(losses)}"
        )
