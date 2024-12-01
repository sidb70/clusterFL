import unittest
import sys

sys.path.append("./")
from datasets.dataloader import (
    load_cifar10,
    create_clustered_dataset,
    ClusterDataset,
    Transform,
)


class TestDataloader(unittest.TestCase):
    def setUp(self):
        self.trainset, self.testset = load_cifar10()

    def test_cifar10(self):
        self.assertEqual(len(self.trainset), 50000)
        self.assertEqual(len(self.testset), 10000)

    def test_clustered_dataset(self):
        clustered_trainset = create_clustered_dataset(self.trainset, 2, "rotation")
        self.assertEqual(len(clustered_trainset), 2)
        self.assertEqual(len(clustered_trainset[0]), 50000)

    def test_rotated_dataset(self):
        unrotated, rotated = create_clustered_dataset(self.trainset, 2, "rotation")
        self.assertEqual(len(rotated), len(self.trainset))
        self.assertEqual(len(rotated[0]), 50000)

    def test_selected_classes_dataset(self):
        cluster1data, cluster2data = create_clustered_dataset(
            self.trainset, 2, "selected_classes"
        )
        self.assertEqual(len(cluster1data), 25000)
        cluster1_classes = set([label for _, label in cluster1data])
        # make sure same classes are not in cluster2
        cluster2_classes = set([label for _, label in cluster2data])
        self.assertTrue(cluster1_classes.isdisjoint(cluster2_classes))


if __name__ == "__main__":
    unittest.main()
