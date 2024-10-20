import torch
from typing import Tuple, Dict
class Client:
    def __init__(self, id, device: torch.device):
        self.id = id
        self.device = device

    def compute_gradients(self, model, data_loader, criterion) -> Dict[str, torch.Tensor]:
        '''
        Compute gradients for the model on the given data loader.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of gradients for each parameter in the model.
        '''
        # accumulate gradients, do not update weights
        model.train()
        model.to(self.device)
        for x, y in data_loader:
            x, y = x.to(self.device), y.to(self.device)
            output = model.forward(x)
            loss = criterion(output, y) / len(data_loader)
            loss.backward()
        grad_state_dict = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_state_dict[name] = param.grad.clone()
                param.grad.zero_()
            else:
                grad_state_dict[name] = torch.zeros_like(param)
        return grad_state_dict
    
    def train(self, model, data_loader, criterion, optimizer, num_epochs: int):
        '''
        Train the model on the given data loader for the given number of epochs.

        Returns:
            None
        '''
        model.train()
        model.to(self.device)
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            for i, (x, y) in enumerate(data_loader):
                x, y = x.to(self.device), y.to(self.device)
                
                output = model.forward(x)
                loss = criterion(output, y)/len(data_loader)
                loss.backward()
            # optimizer.step()
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param -= 0.4*param.grad
                        param.grad.zero_()
            optimizer.zero_grad()
    def evaluate(self, model, data_loader, criterion) -> Tuple[float, float]:
        '''
        Evaluate the model on the given data loader.

        Returns:
            Tuple[float, float]: A tuple containing the average loss and accuracy of the model.
        '''
        model.eval()
        model.to(self.device)
        total_loss = 0
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for x, y in data_loader:
                x, y = x.to(self.device), y.to(self.device)
                output = model.forward(x)
                loss = criterion(output, y)
                total_loss += loss.item()
                total_correct += (output.argmax(dim=1) == y).sum().item()
                total_samples += len(y)
        return total_loss / total_samples, total_correct / total_samples