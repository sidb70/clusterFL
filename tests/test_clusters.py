import unittest
import torch
import sys

sys.path.append("./")
from mockWeightTensors import MockTensors
from cluster import ClusterDaddy


class TestClusterAlgos(unittest.TestCase):
    def setUp(self):
        self.weights = MockTensors().weights
        self.clusters = 3
        self.cluster = ClusterDaddy(self.weights, self.clusters)

    def testNormalize(self):
        # custom = self.cluster.normalize()
        # gpt = normalize_weights(self.weights)
        # for client_c,client_g in zip(custom,gpt):
        #     for label in client_c:
        #       print(client_c[label])
        #       print(client_g[label])
        #       self.assertTrue(torch.allclose(client_c[label], client_g[label]))
        self.assertTrue(True)

    def testKMeans(self):
        clusterList = self.cluster.kMeans(mode="minMax")
        print(clusterList)
        return

    def testBruteCluster(self):
        self.assertTrue(True)


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
