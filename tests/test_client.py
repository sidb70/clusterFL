import unittest
import sys

sys.path.append("./")
from datasets.dataloader import load_cifar10
from models.loader import load_model
from client import Client
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class TestModelTrain(unittest.TestCase):
    def test_train(self):
        trainset, testset = load_cifar10()
        train_loader = DataLoader(trainset, batch_size=32, shuffle=True)
        test_loader = DataLoader(testset, batch_size=32, shuffle=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        client = Client(id=0, device=torch.device(device), cluster_assignment=0)
        model = load_model("cifarcnn")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001)
        for i in range(5):
            model = client.train(model, train_loader, criterion, optimizer, 1)
            # print(f"Here are the keys: {[(x,y) for x,y in model.state_dict().items()]}")
            with torch.no_grad():
                loss, acc = client.evaluate(model, test_loader, criterion)
                print(f"Epoch {i+1}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")

        self.assertGreater(acc, 0.3)


if __name__ == "__main__":
    unittest.main()
