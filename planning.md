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
    
    

**Exploration**
- ~~look at the eigenvalues and eigenvectors of the weight matrices of the models.~~
- ~~have a fixed vector or tensor applied to the weights of each layer of the model.~~
- sample a subset of clients for validation
