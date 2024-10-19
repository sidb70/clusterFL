import unittest
import sys
sys.path.append('./')
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
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        client = Client(id=0, device=torch.device(device))
        model = load_model('cnn')
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        for i in range(10):
            model.train()
            grad_dict = client.compute_gradients(model, train_loader, criterion)
            optimizer.zero_grad()
            for name, param in model.named_parameters():
                self.assertEqual(grad_dict[name].shape, param.shape)
                # set model gradients to the computed gradients
                param.grad = grad_dict[name]

            optimizer.step()
            optimizer.zero_grad()
            for name, param in model.named_parameters():
                self.assertIsNone(param.grad)


            model.eval()
            with torch.no_grad():
                loss, acc = client.evaluate(model, test_loader, criterion)
                print(f'Epoch {i+1}, Loss: {loss:.4f}, Accuracy: {acc:.4f}')



if __name__ == '__main__':
    unittest.main()