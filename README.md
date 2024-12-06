# clusterFL

## Instructions

To run the clustered federated learning experiment, modify the `config.yaml` file with the desired experimental section.


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