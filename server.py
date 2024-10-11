import torch

class Server:
    def __init__(self,config):
        self.config = config
        self.cluster_estimates = {}
    def receive_gradient(self,grad_tensor: torch.Tensor, node_id: int):
        # save gradient tensor
        pass
    def update_cluster_estimates(self):
        # cluster all gradient tensors into unknown number of clusters
        # update self.cluster_estimates
        pass
    def aggregate_gradients(self):
        # aggregate gradients in each cluster
        # return aggregated gradients
        pass