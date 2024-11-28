import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
from typing import List, Dict, Tuple
from models.loader import load_model
from client import Client
from cluster import load_cluster_algorithm
from datasets.dataloader import load_global_dataset, create_clustered_dataset
from aggregation.strategies import load_aggregator
import random
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from utils import load_state_dict
import os
DEVICES = (
    [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    if torch.cuda.is_available()
    else [torch.device("cpu")]
)


class Server:
    """
    Server class that manages the clients and the training process
    Attributes:
        config (Dict): Configuration dictionary
        global_train_set (torch.utils.data.Dataset): Global training dataset
        global_test_set (torch.utils.data.Dataset): Global test dataset
        clustered_train_sets (List[torch.utils.data.Dataset]): List of training datasets for each cluster
        clustered_test_sets (List[torch.utils.data.Dataset]): List of test datasets for each cluster
        num_clients (int): Number of clients
        num_clusters (int): Number of clusters
        lr (float): Learning rate
        local_epochs (int): Number of local epochs
        initial_epochs (int): Number of initial epochs
        clients_to_clusters (List[int]): List of cluster assignments for each client
        clusters_to_clients (Dict[int, List[int]]): Dictionary of clients in each cluster
        clients (List[Client]): List of clients
        cluster_models (List[Dict[str, torch.Tensor]]): List of model weights for each cluster
        aggregator (Aggregator): Aggregator object
    """

    def __init__(self, config, experiment_id):
        """
        Initializes the server object
        Args:
            config (Dict): Configuration dictionary
            experiment_id (str): Experiment ID
        """
        self.config = config
        self.experiment_id = experiment_id
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
        self.clients_to_clusters = [
            i % self.num_clusters for i in range(self.num_clients)
        ]
        self.clusters_to_clients = {}
        for i, cluster in enumerate(self.clients_to_clusters):
            if cluster not in self.clusters_to_clients:
                self.clusters_to_clients[cluster] = []
            self.clusters_to_clients[cluster].append(i)

        self.model_save_dir = os.path.join("data", "models",experiment_id)
        os.makedirs(self.model_save_dir, exist_ok=True)

        self.create_clients(self.num_clients)
        self.aggregator = load_aggregator(config["aggregator"])
        self.cluster_params = config.get("cluster_params", {})
        self.cluster_algorithm = load_cluster_algorithm(
            config["cluster"], clusters=self.num_clusters, **self.cluster_params
        )

    def create_clients(self, num_clients):
        """
        Creates the clients and their respective training and test data, in accordance with the cluster assignments
        Args:
            num_clients (int): Number of clients
        """
        self.clients = []
        self.client_train_indices = []
        self.client_test_indices = []
        initial_model = load_model(self.config["model"])
        for i in range(num_clients):

            client_model = deepcopy(initial_model)
            torch.save(client_model.state_dict(), os.path.join(self.model_save_dir, f"client_{i}.pt"))

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
                    f"client {client} train start {self.client_train_indices[client][0]} end {self.client_train_indices[client][-1]}\
                         on dataset {cluster_num}. Device {self.clients[client].device}"
                )

    def aggregate(
        self, models: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregates the models from the clients
        Args:
            models (List[Dict[str, torch.Tensor]]): List of gradients from the clients
        Returns:
            Dict[str, torch.Tensor]: Aggregated model
        """
        return self.aggregator.aggregate(models)

    def get_client_data(self, client_id, batch_size):
        """
        Gets the training and test data for a client
        Args:
            client_id (int): Client ID
            batch_size (int): Batch size
        Returns:
            DataLoader: Training data loader
            DataLoader: Test data loader
        """
        assigned_cluster = self.clients[client_id].cluster_assignment
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
        """
        Clusters the clients based on the model weights
        Args:
            state_dicts (List[Dict[str, torch.Tensor]]): List of model weights
        Returns:
            List[List[int]]: List of clients in each cluster
        """
        return self.cluster_algorithm.cluster(state_dicts, **self.cluster_params)

    def run_local_update_worker(
        self, client_id: int
    ) -> Tuple[int, int, Dict[str, torch.Tensor]]:
        """
        Run a local update on a single client
        Args:
            client_id: ID of the client
        Returns:
            int: ID of the client
            Dict[str, torch.Tensor]: Updated model weights
        """

        client_train_loader, _ = self.get_client_data(client_id, batch_size=32)
        cluster_id = self.clients_to_clusters[client_id]
        client_model = load_state_dict(load_model(self.config["model"]), os.path.join(self.model_save_dir, f"client_{client_id}.pt"))

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(client_model.parameters(), lr=self.lr)
        print("Training client", client_id, "cluster", cluster_id)
        updated_model = self.clients[client_id].train(
            client_model,
            client_train_loader,
            criterion,
            optimizer,
            self.local_epochs,
        )
        return client_id, cluster_id, updated_model

    def initial_cluster_rounds(self) -> None:
        """
        Performs the initial clustering of the clients by training them for a few epochs and clustering them based on the model weights
        """
        if self.initial_epochs == 0:
            print("Initial epochs is 0, skipping initial clustering")
            return
        clients_models = []
        with ThreadPoolExecutor(max_workers=len(DEVICES)) as executor:
            futures = []
            for client in self.clients:
                futures.append(executor.submit(self.run_local_update_worker, client.id))
            for future in futures:
                client_id, _, updated_model = future.result()
                clients_models.append((client_id, updated_model.state_dict()))
                print("Client", client_id, "finished training")
        random.shuffle(clients_models)
        assert len(clients_models) == len(self.clients)
        clusters = self.cluster([model for _, model in clients_models])

        for i, assignments in enumerate(clusters):
            cluster_clients = [clients_models[j][0] for j in assignments]
            print(f"Cluster {i} estimated assignments: {sorted(cluster_clients)}")
            for client in cluster_clients:
                self.clients_to_clusters[client] = i

    def fl_round(self) -> None:
        """
        Performs a clustered federated learning round
        """
        num_clients = len(self.clients)
        num_sampled = max(1, int(self.config.get("train_sample_rate", 1) * num_clients))
        sampled_clients = random.sample(self.clients, num_sampled)
        updated_models = [[] for _ in range(self.num_clusters)]

        with ThreadPoolExecutor(max_workers=len(DEVICES)) as executor:
            futures = []
            for client in sampled_clients:
                futures.append(executor.submit(self.run_local_update_worker, client.id))
            for future in futures:
                client_id, cluster_id, updated_model = future.result()
                updated_models[cluster_id].append(updated_model.state_dict())
                print("Client", client_id, "finished training")
        if not self.config.get("baseline_avg_whole_network"):
            for cluster_id in range(self.num_clusters):
                print("Aggregating cluster", cluster_id)
                cluster_model = self.aggregate(
                    updated_models[cluster_id]
                )
                for client in self.clusters_to_clients[cluster_id]:
                    torch.save(cluster_model, os.path.join(self.model_save_dir, f"client_{client}.pt"))
        else:
            allmodels = []
            for cluster_id in range(self.num_clusters):
                allmodels.extend(updated_models[cluster_id])
            whole_network_aggregated = self.aggregate(allmodels)
            for client in range(num_clients):
                torch.save(whole_network_aggregated, os.path.join(self.model_save_dir, f"client_{client}.pt"))

    def evaluate(
        self, batch_size: int = 32
    ) -> Tuple[List[Tuple[int, int, float]], List[Tuple[int, int, float]], List[int]]:
        """
        Evaluates the clients
        Args:
            batch_size (int): Batch size
        Returns:
            List[Tuple[int, int, float]]: List of accuracies for each client
            List[Tuple[int, int, float]]: List of losses for each client
            List[int]: List of current cluster predictions for each client
        """
        accuracies = []
        losses = []
        for client in self.clients:
            _, test_loader = self.get_client_data(client.id, batch_size=batch_size)
            current_cluster_id = self.clients_to_clusters[client.id]
            client_model = load_state_dict(load_model(self.config["model"]), os.path.join(self.model_save_dir, f"client_{client.id}.pt"))
            loss, acc = client.evaluate(
                client_model, test_loader, nn.CrossEntropyLoss()
            )
            accuracies.append((client.id, acc))
            losses.append((client.id, loss))
        avg_acc = sum([acc for _, acc in accuracies]) / len(accuracies)
        avg_loss = sum([loss for _, loss in losses]) / len(losses)
        print(f"Average Accuracy: {avg_acc:.2f}%, Average Loss: {avg_loss:.4f}")
        true_cluster_assignments = [
            client.cluster_assignment for client in self.clients
        ]
        current_cluster_predictions = [
            self.clients_to_clusters[client] for client in range(self.num_clients)
        ]
        cluster_results = [
            (
                client,
                true_cluster_assignments[client],
                current_cluster_predictions[client],
            )
            for client in range(self.num_clients)
        ]
        return accuracies, losses, cluster_results
