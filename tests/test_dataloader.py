import unittest
import sys
sys.path.append('./')
from datasets.dataloader import load_cifar10, load_cifar100, load_selected_classes

class TestDataloader(unittest.TestCase):
    def test_cifar10(self):
        trainset, testset = load_cifar10()
        self.assertEqual(len(trainset), 50000)
        self.assertEqual(len(testset), 10000)
    def test_selected_classes(self):
        trainset, testset = load_cifar10()
        selected_classes = [0, 1, 2]
        selected_trainset = load_selected_classes(trainset, selected_classes)
        self.assertEqual(len(selected_trainset), 12750)


if __name__ == '__main__':
    unittest.main()