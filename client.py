import torch

class Client:
    def __init__(self, id, device: torch.device):
        self.id = id
        self.device = device

    def compute_gradients(self, model, data_loader, criterion):
        # accumulate gradients, do not update weights
        for x, y in data_loader:
            x, y = x.to(self.device), y.to(self.device)
            output = model.forward(x)
            loss = criterion(output, y)
            loss.backward()
        grad_state_dict = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_state_dict[name] = param.grad.clone()
            else:
                grad_state_dict[name] = torch.zeros_like(param)
        return grad_state_dict
