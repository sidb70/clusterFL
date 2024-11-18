import torch
from typing import Tuple, Dict

torch.manual_seed(0)

class Client:
    def __init__(self, id, device: torch.device, cluster_assignment):
        """
        Initialize a client with the given id, device, and cluster assignment.

        Args:
            id (int): the id of the client
            device (torch.device): the device on which to run the client
            cluster_assignment (int): the cluster to which the client is
        """
        self.id = id
        self.device = device
        self.cluster_assignment = cluster_assignment
        print(f"Client {self.id} initialized on device: ", self.device)

    def train(
        self, model, data_loader, criterion, optimizer, num_epochs: int
    ) -> torch.nn.Module:
        """
        Train the model on the given data loader for the given number of epochs.

        Returns:
            torch.nn.Module: the updated model
        """
        model.train()
        model.to(self.device)
        optimizer.zero_grad()
        for e in range(num_epochs):
            running_loss = 0
            for batch_idx, (x, y) in enumerate(data_loader):
                x, y = x.to(self.device), y.to(self.device)
                output = model.forward(x)
                loss = criterion(output, y)
                running_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            print(f"Client {self.id} epoch {e} loss: {running_loss / len(data_loader)}")
        return model

    def evaluate(self, model, data_loader, criterion) -> Tuple[float, float]:
        """
        Evaluate the model on the given data loader.

        Returns:
            Tuple[float, float]: A tuple containing the average loss and accuracy of the model.
        """
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
        return total_loss / len(data_loader), total_correct / total_samples
