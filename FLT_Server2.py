import numpy as np
import flwr as fl
from torch.utils.data import TensorDataset, DataLoader
import torch
from flwr.common import Context, ndarrays_to_parameters
from flwr.server.strategy import FedAvg
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from FLT_Loader2 import n_classes, n_features, load_dataset, split_data_across_clients

from flwr.client import Client, ClientApp, NumPyClient
from flwr.simulation import run_simulation

from utils2 import *

random.seed(42)

# Load and shuffle
path = 'NFCS2021StateData220627.csv'
raw_data = load_dataset(path)
random.shuffle(raw_data)
split1 = int(0.8 * len(raw_data))
train_data = raw_data[:split1]
test_data = raw_data[split1:]
print(f"Total samples: {len(raw_data)}  → Train: {len(train_data)}, Test: {len(test_data)}")

train_sets = split_data_across_clients(train_data)
test_sets = split_data_across_clients(test_data)



# Sets the parameters of the model
def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict(
        {k: torch.tensor(v) for k, v in params_dict}
    )
    net.load_state_dict(state_dict, strict=True)

# Retrieves the parameters from the model
def get_weights(net):
    ndarrays = [
        val.cpu().numpy() for _, val in net.state_dict().items()
    ]
    return ndarrays

#declare Epoch law
def fit_config(server_round: int):
    config_dict = {
        "local_epochs": 4 if server_round < 75 else 10,
    }
    return config_dict

#Connect the training in the pipeline using the Flower Client
class FlowerClient(NumPyClient):
    def __init__(self, net, trainset, testset):
        # Convert raw lists into TensorDataset once at init:
        self.net = net
        xs_train = torch.tensor([x for x, y in trainset], dtype=torch.float32)
        ys_train = torch.tensor([y for x, y in trainset], dtype=torch.long)
        self.trainset = TensorDataset(xs_train, ys_train)

        xs_test = torch.tensor([x for x, y in testset], dtype=torch.float32)
        ys_test = torch.tensor([y for x, y in testset], dtype=torch.long)
        self.testset = TensorDataset(xs_test, ys_test)

    # Train the model
    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        epochs = config["local_epochs"]
        log(INFO, f"client trains for {epochs} epochs")
        train_model(self.net, self.trainset, epochs)
        return get_weights(self.net), len(self.trainset), {}

    # Test the model
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        set_weights(self.net, parameters)
        loss, accuracy = evaluate_model(self.net, self.testset)
        return loss, len(self.testset), {"accuracy": accuracy}

# Client function, simulates every client on a single machine
def client_fn(context: Context) -> Client:
    net = SimpleModel()
    partition_id = int(context.node_config["partition-id"])
    client_train = train_sets[int(partition_id)]
    client_test = test_sets[int(partition_id)]
    return FlowerClient(net, client_train, client_test).to_client()
#Create an instance of the ClientApp
client = ClientApp(client_fn)

#Server-Side Evaluation
def evaluate(server_round, parameters, config):
    net = SimpleModel()
    set_weights(net, parameters)

    # 2) Flatten all clients' samples into one list of (features, label)
    flat = [(x, y) for client in test_sets for (x, y) in client]
    # 3) Convert to Tensors
    features = torch.tensor([x for x, _ in flat], dtype=torch.float32)
    labels   = torch.tensor([y for _, y in flat], dtype=torch.long)
    full_test_dataset = TensorDataset(features, labels)
    # 4) Evaluate global
    loss, accuracy = evaluate_model(net, full_test_dataset)



    log(INFO, "test accuracy on all States: %.4f", accuracy)
    #log(INFO, "test accuracy on Alabama: %.4f", accuracy1)
    #log(INFO, "test accuracy on Alaska: %.4f", accuracy2)
    #log(INFO, "test accuracy on Arizona: %.4f", accuracy3)

    if server_round == 100: #Final Round
        cm = compute_confusion_matrix(net, full_test_dataset)
        plot_confusion_matrix(cm, "Final Global Model")

    # 8) Return metrics for Flower
    return loss, {"accuracy": accuracy}


net = SimpleModel()
params = ndarrays_to_parameters(get_weights(net))

def server_fn(context: Context): #Federated Averaging
    strategy = FedAvg(
        fraction_fit=0.2, #Fraction of avaliable clients selected for training
        fraction_evaluate=0.4, #Fraction of avaliable clients selected for evaluation
        initial_parameters=params, #initial model weights
        evaluate_fn=evaluate, #function to use for server-side evaluation
        on_fit_config_fn=fit_config, #Epoch decleration
    )
    config=ServerConfig(num_rounds=100)
    return ServerAppComponents(
        strategy=strategy,
        config=config,
    )
#Create an instance of severapp
server = ServerApp(server_fn=server_fn)
# Initiate the simulation passing the server and client apps
# Specify the number of super nodes that will be selected on every round
run_simulation(
    server_app=server,
    client_app=client,
    num_supernodes=51, #number of clients
    backend_config=backend_setup,
)

