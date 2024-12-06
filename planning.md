### Research Questions

**Background:** clustered federated learning works better than traditional federated learning when clients have different data distributions
	- compare simple network-wide averaging vs if we assume oracle clustering

**Assumptions:** Each cluster has unique data distribution, each client shares same CNN architecture. 

**Hypothesis** It is possible to accuratley cluster clients into their respective data distributions based on their model parameters without the server having knowledge of their training data and without different clients having knowledge of each others training data.

**Question**: How to cluster models without knowing their data? 
  - Does normalization help to accuratley compute distance between models?
  - What is an accurate distance metric for neural networks?
        original hypothesis: parameter-wise manhattan distance is a good metric.\
        **Experiment**: Run k-means with Manhattan. Vary the cluster types (rotation, selected classes). 10 clients. 2 clusters.

    problem: permutation invariance. many reorderings of the a NN can be equivalent networks. So parameter-wise distance is not a good metric.\

    **Experiment** to show that this is a problem with a simple example
    Train a single model, then permute the weights. Show that:\
    1. the permuted model has the same loss\
    2. But, if we calculate distance between the original_model and permuted_model we get a high distance because the weights are different.
      - Also, compare this distance to the distance between the original model and a model trained on a different data distribution.
    3. Try a custom layer distance algorithm that uses the largest distance between matched layers. (Rotation, selected classes). 10 clients, 2 clusters.
    
    

**Experiment**: Try to break the k-means clustering
~~- try on cifar 100, where each distribution has many types of data.~~  Done
- ~~clustering performance and model performance of evenly split clusters (10 client, 5 client, 5 clients) vs even split (5 clients in each cluster).~~ Done



## Consolidating results



<make a table of results>

| Experiment Type | Dataset       | Clusters | Clients | Results Summary                                                                 |
|-----------------|---------------|----------|---------|---------------------------------------------------------------------------------|
| Even Clusters   | MNIST, CIFAR-10 | 5        | 25      | Rotations and Selected classes.                                                 |
| Even Clusters   | MNIST, CIFAR-10, CIFAR-100 | 2        | 10      | Rotations and Selected classes. CIFAR-100 different initial epochs.             |
| Uneven Clusters | MNIST, CIFAR-10 | 5        | 25      | Rotations and Selected classes.                                                 |
| Uneven Clusters | MNIST, CIFAR-10 | 2        | 10      | Rotations and Selected classes.                                                 |


| even vs uneven | dataset | # of clusters | type of split | init epochs | cluster algo | accuracy | cluster purity
| Even vs Uneven | Dataset       | # of Clusters | Type of Split | Init Epochs | Cluster Algo | Accuracy change | Cluster Purity |
|----------------|---------------|---------------|---------------|-------------|--------------|----------|----------------|
| Even           | MNIST         | 2             | Rotation          | 1          | K-means      | 5%      | 1.0          |
| Even           | CIFAR-10      | 2             | Rotation          | 1          | K-means      | 5%      | 1.0          |
| Even           | CIFAR-100     | 2             | Rotation          | 1          | K-means      | -%     | -            |
| Even           | MNIST         | 2             | Selected Classes  | 1          | K-means      | 3%      | 1.0          |
| Even           | CIFAR-10      | 2             | Selected Classes  | 1          | K-means      | 28%     | 1.0          |
| Even           | CIFAR-100     | 2             | Selected Classes  | 1          | K-means      | -%      | -            |
| Even           | MNIST         | 5             | Rotation          | 1          | K-means      | 21%     | 0.88         |
| Even           | CIFAR-10      | 5             | Rotation          | 1          | K-means      | 17%     | 1.0          |
| Even           | CIFAR-100     | 5             | Rotation          | 1          | K-means      | -%      | -            |
| Even           | MNIST         | 5             | Selected Classes  | 1          | K-means      | 12%     | 0.88         |
| Even           | CIFAR-10      | 5             | Selected Classes  | 1          | K-means      | 50%     | 1.0          |
| Even           | CIFAR-100     | 5             | Selected Classes  | 1          | K-means      | -%      | -            |
| Uneven         | CIFAR-10      | 2             | Rotation          | 1          | K-means      | 7%      | 1.0          |
| Uneven         | CIFAR-10      | 5             | Rotation          | 1          | K-means      | 12%     | 1.0          |
| Uneven         | CIFAR-10      | 2             | Selected Classes  | 1          | K-means      | 25%     | 1.0          |
| Uneven         | CIFAR-10      | 5             | Selected Classes  | 1          | K-means      | 43%     | 1.0          |