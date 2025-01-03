import torch
from typing import List, Dict


class Aggregator:
    """
    Base class for aggregation strategies
    """

    def aggregate(
        self, models: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


class FedAvg(Aggregator):
    """
    Federated model averaging aggregation strategy
    """

    def aggregate(
        self, models: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        avg_dict = {
            key: torch.zeros_like(models[0][key], dtype=torch.float)
            for key in models[0].keys()
        }
        for model in models:
            for key in model.keys():
                avg_dict[key] += model[key].float()
        for key in avg_dict.keys():
            avg_dict[key] /= len(models)
            avg_dict[key] = avg_dict[key].type(models[0][key].dtype)
        return avg_dict


def load_aggregator(name: str) -> Aggregator:
    """
    Helper function to load aggregation strategy from yaml file attribute
    """
    if name == "fedavg":
        return FedAvg()
    else:
        raise NotImplementedError
