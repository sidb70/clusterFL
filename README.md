# clusterFL

**Abstract**\
Federated learning (FL) requires each client device in a collaborative training network to train a local model
using on-device training data for some iterations, then transmitting the model updates to a central server,
which will aggregate received updates and send back the global model. However, clients in the network may
have local training data generated from different statistical distributions, violating the common assumption of
independently and identically distributed (iid) data in machine learning. Thus, we consider a network of clients
with different distributions of training data (non-iid), with the objective of minimizing the loss function of
each client within their respective domains. We evaluate the ability of state-of-the-art unsupervised clustering
algorithms to correctly cluster clients based on their model parameters and to converge the global models
within each cluster.

[Manuscript](https://github.com/sidb70/clusterFL/blob/main/bhattacharya_aridi.pdf)

## Instructions

To parse the experimental result files into readable format and figures, run the notebook in ./experiments/results/results.ipynb. 

To run the clustered federated learning experiment, modify the `config.yaml` file with the desired experimental parameters.

To run the experiment, use the following command:

```bash
python3 experiment_runner.py 
```

 Below is an example configuration:
```yaml
description: "baseline 5 clusters cifar 100 selected classes"
clients: 25
num_clusters: 5
uneven_clusters: false
num_rounds: 50
local_epochs: 1
train_sample_rate: 1.0
initial_epochs: 0
evaluation_sampling_rate: 1.0
task: 'cifar100'
aggregator: 'fedavg'
lr: 0.01

cluster_split_type: 'selected_classes'
baseline_avg_whole_network: true

cluster: 'kmeans'
# can also be 'filter'

cluster_params:
    k_iter: 10
    filter_distance: 'max'

random_seed: 322
```

Make sure to adjust the parameters as needed for your specific experiment.
