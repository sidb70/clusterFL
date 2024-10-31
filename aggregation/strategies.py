import torch
from typing import List, Dict

class Aggregator:
    def aggregate(self, models: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:   
        raise NotImplementedError
    
class FedAvg(Aggregator):
    def aggregate(self, models: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        avg_dict = {}
        for key in models[0].keys():
            avg_dict[key] = torch.stack([model[key] for model in models]).mean(dim=0)
        return avg_dict
    
def load_aggregator(name: str) -> Aggregator:
    if name == 'fedavg':
        return FedAvg()
    else:
        raise NotImplementedError