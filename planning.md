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
- try on cifar 100, where each distribution has many types of data
- clustering performance and model performance of evenly split clusters (10 client, 5 client, 5 clients) vs even split (5 clients in each cluster).
