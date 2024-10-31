import torch
from typing import Tuple, Dict
class Client:
    def __init__(self, id, device: torch.device, cluster_assignment):
        self.id = id
        self.device = device
        self.cluster_assignment = cluster_assignment
        print(f"Client {self.id} initialized on device: ", self.device)

    # def compute_gradients(self, model, data_loader, criterion) -> Dict[str, torch.Tensor]:
    #     '''
    #     Compute gradients for the model on the given data loader.

    #     Returns:
    #         Dict[str, torch.Tensor]: A dictionary of gradients for each parameter in the model.
    #     '''
    #     # accumulate gradients, do not update weights
    #     model.train()
    #     model.to(self.device)
    #     num_batches = len(data_loader)
        
    #     grad_state_dict = {}
    #     for name, param in model.named_parameters():
    #         grad_state_dict[name] = torch.zeros_like(param)
    #     # optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    #     for x, y in data_loader:
    #         x, y = x.to(self.device), y.to(self.device)
    #         output = model.forward(x)
    #         loss = criterion(output, y) 
    #         loss.backward()
    #         # optimizer.step()
    #         # optimizer.zero_grad()
    #         # Accumulate normalized gradients
    #         for name, param in model.named_parameters():
    #             if param.grad is not None:
    #                 grad_state_dict[name] += param.grad.clone()
    #                 param.grad.zero_()

    #     for name, param in model.named_parameters():
    #         grad_state_dict[name] /= num_batches
            
    #     return grad_state_dict
    
    def train(self, model, data_loader, criterion, optimizer, num_epochs: int) -> torch.nn.Module:
        '''
        Train the model on the given data loader for the given number of epochs.

        Returns:
            torch.nn.Module: the updated model
        '''
        model.train()
        model.to(self.device)
        optimizer.zero_grad()
        for e in range(num_epochs):
            for batch_idx, (x, y) in enumerate(data_loader):
                x, y = x.to(self.device), y.to(self.device)
                output = model.forward(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        return model

                
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
        return total_loss /len(data_loader), total_correct / total_samples