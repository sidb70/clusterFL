import yaml
from server import Server

# from datasets.dataloader import cluster_cifar100
import torch
import torch.nn as nn
import random
import numpy as np
import uuid
import json

experiment_id = str(uuid.uuid4())


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi gpu
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FLNetwork:
    def __init__(self, config):
        self.config = config
        self.num_rounds = config["num_rounds"]
        self.num_clients = config["clients"]
        self.num_clusters = config["num_clusters"]
        self.server = Server(config, experiment_id)
        set_global_seed(0)

    def run(self):
        num_rounds = config["num_rounds"]
        self.server.initial_cluster_rounds()
        results = []
        for r in range(num_rounds):
            print("\nRound: ", r)
            self.server.fl_round()
            accuracies, losses, cluster_results = self.server.evaluate()
            round_data = {
                "round": r,
                "accuracies": [
                    {"client_id": client_id, "accuracy": accuracy}
                    for client_id, accuracy in accuracies
                ],
                "losses": [
                    {"client_id": client_id, "loss": loss} for client_id, loss in losses
                ],
                "cluster_results": cluster_results,
            }

            # Append round data to results
            results.append(round_data)
        # Save results to json
        json_data = {"results": results, "config": self.config}
        with open(experiment_id + ".json", "w") as json_file:
            json.dump(json_data, json_file, indent=4)


if __name__ == "__main__":
    yaml_path = "config.yaml"
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    print("Loaded config")
    print(config)

    fl_net = FLNetwork(config)
    fl_net.run()
