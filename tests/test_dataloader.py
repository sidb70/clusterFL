import unittest
import sys

sys.path.append("./")
from datasets.dataloader import (
    load_cifar10,
    create_clustered_dataset,
    SelectedClassesDataset,
    RotatedDataset,
)


class TestDataloader(unittest.TestCase):
    def setUp(self):
        self.trainset, self.testset = load_cifar10()

    def test_cifar10(self):
        self.assertEqual(len(self.trainset), 50000)
        self.assertEqual(len(self.testset), 10000)

    def test_selected_classes(self):
        selected_classes = [0, 1, 2]
        selected_trainset = SelectedClassesDataset(self.trainset, selected_classes)
        self.assertEqual(len(selected_trainset), 15000)

    def test_rotation(self):
        rotated_trainset = RotatedDataset(dataset=self.trainset, rotation=90)
        self.assertEqual(len(rotated_trainset), 50000)
        self.assertEqual(rotated_trainset[0][0].shape, (3, 32, 32))

    def test_clustered_dataset(self):
        clustered_trainset = create_clustered_dataset(self.trainset, 3, "rotation")
        self.assertEqual(len(clustered_trainset), 3)
        self.assertEqual(len(clustered_trainset[0]), 50000)


if __name__ == "__main__":
    unittest.main()
