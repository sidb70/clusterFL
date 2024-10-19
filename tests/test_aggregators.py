import unittest
import sys
sys.path.append('./')
import torch
from aggregation.strategies import Aggregator, FedAvg
from models.loader import load_model

class TestAggregators(unittest.TestCase):

    def test_fedavg(self):
        fedavg = FedAvg()
        models = [load_model('cnn') for _ in range(3)]
        print(models)
        for i in range(len(models)):
            for name, param in models[i].named_parameters():
                models[i].state_dict()[name] = torch.rand_like(param)

        state_dicts = [model.state_dict() for model in models]
        aggregated = fedavg.aggregate(state_dicts)
        for name, param in aggregated.items():
            self.assertEqual(param.shape, models[0].state_dict()[name].shape)
            

        test_inp_1 = {'1': torch.tensor([-1,-1,-1], dtype=torch.float32), '2': torch.tensor([-1,-1,-1], dtype=torch.float32)}
        test_inp_2 = {'1': torch.tensor([0, 0, 0], dtype=torch.float32), '2': torch.tensor([0, 0, 0], dtype=torch.float32)}
        test_inp_3 = {'1': torch.tensor([1, 1, 1], dtype=torch.float32), '2': torch.tensor([1, 1, 1], dtype=torch.float32)}
        aggregated = fedavg.aggregate([test_inp_1, test_inp_2, test_inp_3])
        for name, param in aggregated.items():
            self.assertEqual(param.shape, test_inp_1[name].shape)
            for i in range(len(param)):
                self.assertEqual(param[i].item(), 0)




if __name__ == '__main__':
    unittest.main()