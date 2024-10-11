import yaml
from server import Server
from dataloader import cluster_cifar100



class FLNetwork:
    def __init__(self, config):
        self.config = config
        self.num_rounds = config['num_rounds']
        self.num_clients = config['clients']
        self.num_clusters = config['num_clusters']
        self.server = Server(config)
        self.clustered_data = cluster_cifar100(self.num_clusters)
        self.create_clients()

    def create_clients(self):
        self.clients = [(i,i%self.num_clusters) for i in range(self.num_clients)]
    def run_fl(self):
        num_rounds = config['num_rounds']

        for r in range(num_rounds):
            print("Round: ", r)
            for client in self.clients:
                # send model to client
                # train model on client
                # send gradient to server
                pass

            self.server.update_cluster_estimates()
            for client in self.clients:
                # send cluster estimate to client
                pass


if __name__=='__main__':
    yaml_path = 'config.yaml'
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    print("Loaded config")
    print(config)

    fl_net = FLNetwork(config)
    fl_net.run()