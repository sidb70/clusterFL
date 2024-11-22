import unittest
import torch
import sys
import yaml
sys.path.append("./")
from mockWeightTensors import MockTensors
from cluster import load_cluster_algorithm, PointWiseKMeans, FilterMatching


class TestClusterAlgos(unittest.TestCase):
    def setUp(self):
        self.weights = MockTensors().weights
        self.clusters = 3
        self.config = yaml.safe_load("config.yaml")

    def test_loader(self):
        cluster = load_cluster_algorithm("kmeans", self.clusters)
        self.assertIsInstance(cluster, PointWiseKMeans)
        cluster = load_cluster_algorithm("filter", self.clusters, filter_distance='max')
        self.assertIsInstance(cluster, FilterMatching)
    def test_kmeans(self):
        cluster = PointWiseKMeans(self.clusters)
        clusterList = cluster.cluster(self.weights)
        self.assertEqual(len(clusterList), self.clusters)
        self.assertEqual(sum([len(cluster) for cluster in clusterList]), len(self.weights))

    def test_filter(self):
        cluster = FilterMatching(self.clusters, filter_distance='max')
        clusterList = cluster.cluster(self.weights)
        self.assertEqual(len(clusterList), self.clusters)
        self.assertEqual(sum([len(cluster) for cluster in clusterList]), len(self.weights))

    # def testNormalize(self):
    #     # custom = self.cluster.normalize()
    #     # gpt = normalize_weights(self.weights)
    #     # for client_c,client_g in zip(custom,gpt):
    #     #     for label in client_c:
    #     #       print(client_c[label])
    #     #       print(client_g[label])
    #     #       self.assertTrue(torch.allclose(client_c[label], client_g[label]))
    #     self.assertTrue(True)


# GPT normalize
def normalize_weights(data):
    normalized_data = []

    # Collect tensors by label across all data points
    tensor_groups = {}
    for point in data:
        for label, tensor in point.items():
            if label not in tensor_groups:
                tensor_groups[label] = []
            tensor_groups[label].append(tensor)

    # Normalize tensors for each label
    normalized_tensors = {}
    for label, tensors in tensor_groups.items():
        # Stack tensors and compute mean and std for normalization
        stacked = torch.stack(tensors)
        mean = stacked.mean()
        std = stacked.std()
        normalized_tensors[label] = [(tensor - mean) / std for tensor in tensors]

    # Create normalized data points
    for i in range(len(data)):
        normalized_point = {label: normalized_tensors[label][i] for label in data[i]}
        normalized_data.append(normalized_point)

    return normalized_data


if __name__ == "__main__":
    unittest.main()
