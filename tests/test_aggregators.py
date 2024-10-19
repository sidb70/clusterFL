import unittest
import sys
sys.path.append('./')
import torch
from aggregation.strategies import Aggregator, FedAvg

class TestAggregators(unittest.TestCase):
    def test_base_class(self):
        aggregator = Aggregator()
        with self.assertRaises(NotImplementedError):
            aggregator.aggregate([])

    def test_fedavg(self):
        fedavg = FedAvg()
        raise NotImplementedError

if __name__ == '__main__':
    unittest.main()