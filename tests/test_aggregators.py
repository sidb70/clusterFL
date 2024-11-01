import unittest
import sys

sys.path.append("./")
import torch
from aggregation.strategies import Aggregator, FedAvg
from models.loader import load_model


class TestAggregators(unittest.TestCase):

    def test_fedavg(self):
        fedavg = FedAvg()
        models = [load_model("cnn") for _ in range(3)]
        print(models)
        for i in range(len(models)):
            for name, param in models[i].named_parameters():
                models[i].state_dict()[name] = torch.rand_like(param)

        state_dicts = [model.state_dict() for model in models]
        aggregated = fedavg.aggregate(state_dicts)
        for name, param in aggregated.items():
            self.assertEqual(param.shape, models[0].state_dict()[name].shape)

        test_inp_1 = {"0": torch.tensor([1, 2, 3])}
        test_inp_2 = {"0": torch.tensor([2, 3, 4])}
        test_inp_3 = {"0": torch.tensor([3, 4, 5])}
        coord1_avg = (1 + 2 + 3) / 3
        coord2_avg = (2 + 3 + 4) / 3
        coord3_avg = (3 + 4 + 5) / 3
        aggregated = fedavg.aggregate([test_inp_1, test_inp_2, test_inp_3])
        for name, param in aggregated.items():
            self.assertEqual(param.shape, test_inp_1[name].shape)
            self.assertEqual(param[0].item(), coord1_avg)
            self.assertEqual(param[1].item(), coord2_avg)
            self.assertEqual(param[2].item(), coord3_avg)


if __name__ == "__main__":
    unittest.main()
