import warnings
import logging
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("ray").setLevel(logging.WARNING)

import numpy as np
import flwr as fl
from torch.utils.data import TensorDataset, DataLoader
import torch  
from flwr.common import Context, ndarrays_to_parameters
from flwr.server.strategy import FedAvg, DifferentialPrivacyClientSideAdaptiveClipping
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from FLT_Loader2 import n_classes, n_features, load_dataset, split_data_across_clients

from flwr.client import Client, ClientApp, NumPyClient
from flwr.simulation import run_simulation
from flwr.client.mod import adaptiveclipping_mod

from utils2 import *



random.seed(42)


# Load and shuffle

raw_data = load_dataset(csv_path)

random.shuffle(raw_data)
split1 = int(0.8 * len(raw_data))
train_data = raw_data[:split1]
test_data = raw_data[split1:]
print(f"Total samples: {len(raw_data)}  → Train: {len(train_data)}, Test: {len(test_data)}")

train_sets = split_data_across_clients(train_data)
test_sets = split_data_across_clients(test_data)
round_accuracies = []
round_recalls = []
round_f1s = []
round_aucs = []
comms_log = defaultdict(lambda: {"upload": 0}) #Communication cost logs

tmp_file = "client_comms_tmp.txt"
if os.path.exists(tmp_file):
    os.remove(tmp_file)



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
        val.cpu().numpy() for _, val in net.state_dict().items() if isinstance(val, torch.Tensor)
    ]
    return ndarrays

#declare Epoch law
def fit_config(server_round: int):
    config_dict = {
        "local_epochs": 10 if server_round < ro*0.9 else 30,
    }
    return config_dict #find a way to stop epoch running when accuracy stagnates/drops

#Connect the training in the pipeline using the Flower Client
class FlowerClient(NumPyClient):
    def __init__(self, net, trainset, testset, idd):
        # Convert raw lists into TensorDataset once at init:
        self.net = net
        xs_train = torch.tensor([x for x, y in trainset], dtype=torch.float32)
        ys_train = torch.tensor([y for x, y in trainset], dtype=torch.long)
        self.trainset = TensorDataset(xs_train, ys_train)

        xs_test = torch.tensor([x for x, y in testset], dtype=torch.float32)
        ys_test = torch.tensor([y for x, y in testset], dtype=torch.long)
        self.testset = TensorDataset(xs_test, ys_test)
        self.partition_id = idd

    # Train the model
    def fit(self, parameters, config):
        global comms_log
        # If this client has no training data, just echo back the incoming weights
        if len(self.trainset) == 0:
            log(INFO, f"client has no data!")
            return parameters, 0, {}
        set_weights(self.net, parameters)
        epochs = config["local_epochs"]
         # only print once, when partition-id==0
        if self.partition_id == 0:
            log(INFO, f"client trains for {epochs} epochs")
        train_model(self.net, self.trainset, epochs)
        #calculate upload size in bytes
        updated_params = get_weights(self.net)
        upload_cost = sum(param.nbytes for param in updated_params)
        for i, param in enumerate(updated_params):
            log(INFO, f"[DEBUG] Upload param {i} size: {param.nbytes} bytes")
        upload_cost_kb = upload_cost / 1024
        with open("client_comms_tmp.txt", "a") as f:
            f.write(f"{self.partition_id},upload,{upload_cost_kb}\n")

        # Store per-client communication cost (you can also aggregate per-round)
        if self.partition_id not in comms_log:
            comms_log[self.partition_id] = {"upload": 0}
        comms_log[self.partition_id]["upload"] += upload_cost


        return get_weights(self.net), len(self.trainset), {}

    # Test the model
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        set_weights(self.net, parameters)
        loss, accuracy, recall, f1, auc = evaluate_model(self.net, self.testset)
        return loss, len(self.testset), {"accuracy": accuracy} # TODO: add recall and f1?

# Client function, simulates every client on a single machine
def client_fn(context: Context) -> Client:
    global comms_log
    net = HighwayModel2(n_features, n_classes)
    partition_id = int(context.node_config["partition-id"])
    client_train = train_sets[int(partition_id)]
    client_test = test_sets[int(partition_id)]

    # Log the initial download of weights
    initial_params = get_weights(net)
    download_cost = sum(p.nbytes for p in initial_params)

    return FlowerClient(net, client_train, client_test, partition_id).to_client()


#Create an instance of the ClientApp
client = ClientApp(
    client_fn, 
    #mods=[adaptiveclipping_mod], #modifiers
)

#Server-Side Evaluation
def evaluate(server_round, parameters, config):
    net = HighwayModel2(n_features, n_classes)
    set_weights(net, parameters)

    # 2) Flatten all clients' samples into one list of (features, label)
    flat = [(x, y) for client in test_sets for (x, y) in client]
    # 3) Convert to Tensors
    features = torch.tensor([x for x, _ in flat], dtype=torch.float32)
    labels   = torch.tensor([y for _, y in flat], dtype=torch.long)
    full_test_dataset = TensorDataset(features, labels)
    # 4) Evaluate global
    loss, accuracy, recall, f1, auc = evaluate_model(net, full_test_dataset)

    round_accuracies.append(accuracy)
    round_recalls.append(recall)
    round_f1s.append(f1)
    round_aucs.append(auc)

    states = [
        "Alabama","Alaska","Arizona","Arkansas","California","Colorado","Connecticut",
        "Delaware","District of Columbia","Florida","Georgia","Hawaii","Idaho","Illinois",
        "Indiana","Iowa","Kansas","Kentucky","Louisiana","Maine","Maryland","Massachusetts",
        "Michigan","Minnesota","Mississippi","Missouri","Montana","Nebraska","Nevada",
        "New Hampshire","New Jersey","New Mexico","New York","North Carolina","North Dakota",
        "Ohio","Oklahoma","Oregon","Pennsylvania","Rhode Island","South Carolina","South Dakota",
        "Tennessee","Texas","Utah","Vermont","Virginia","Washington","West Virginia","Wisconsin","Wyoming"
    ]
    #Chosen States to keep track of
    chosen = ("Washington", "Kansas", "Arkansas", "New Hampshire", "Pennsilvania", "Texas", "Massachusetts", "Colorado", "New York", "California", "Florida", "Hawaii")
    assert len(feature_names) == len(cat_sizes), f"feature_names {feature_names} and cat_sizes {cat_sizes} mismatch!"

    for idx, client_data in enumerate(test_sets):
        if 0 <= idx < len(states):
            stat = states[idx]
        else:
            stat = "N/A"
        if stat in chosen:
            # Build TensorDataset for this client
            feats = torch.tensor([x for x, _ in client_data], dtype=torch.float32)
            labs  = torch.tensor([y for _, y in client_data], dtype=torch.long)
            ds    = TensorDataset(feats, labs)
            # Compute metrics
            loss_c, acc_c, recall_c, f1_c, auc_c = evaluate_model(net, ds)

            client_metric_history[stat]["accuracy"].append(acc_c) #Save Client Metrics
            client_metric_history[stat]["recall"].append(recall_c)
            client_metric_history[stat]["f1"].append(f1_c)
            client_metric_history[stat]["auc"].append(auc_c)

        if server_round == ro: #final round save data
            #write communication costs to text file
            comms_path = os.path.join(os.path.dirname(csv_path), "communication_costs.txt")
            client_costs = defaultdict(lambda: {"upload": 0})
            for cid in range(51):
                client_costs[cid]  # Access to trigger defaultdict initialization
            with open(tmp_file, "r") as f:
                for line in f:
                    cid, typ, val = line.strip().split(",")
                    client_costs[int(cid)][typ] += float(val)

            plotCommunication(client_costs=client_costs, in_mb=True) #Communication graph
            with open(comms_path, "w") as f:
                f.write("Client Communication Costs (in KB):\n\n")
                total_upload = 0
                total_download = 0
                for cid in sorted(client_costs):
                    up = client_costs[cid]["upload"]
                    total_upload += up
                    upload_mb = client_costs[cid]['upload'] / 1024
                f.write(f"\nTotal Upload: {total_upload:.2f} KB\n")
                f.write(f"Total Communication: {total_upload:.2f} KB\n")

            # Build TensorDataset for this client
            feats = torch.tensor([x for x, _ in client_data], dtype=torch.float32)
            labs  = torch.tensor([y for _, y in client_data], dtype=torch.long)
            ds    = TensorDataset(feats, labs)
            # Compute metrics
            loss_c, acc_c, recall_c, f1_c, auc_c = evaluate_model(net, ds)
            finalMetrics["accuracy"].append(acc_c) #Save Client Metrics
            finalMetrics["recall"].append(recall_c)
            finalMetrics["f1"].append(f1_c)
            finalMetrics["auc"].append(auc_c)

    log(INFO, "test accuracy on all States: %.4f", accuracy)
    log(INFO, "test recall on all States: %.4f", recall)
    log(INFO, "test F1 on all States: %.4f", f1)
    log(INFO, "test AUC on all States: %.4f", auc)


    if server_round == ro: #Final Round computations
        o = os.path.dirname(csv_path)
        #Write communication costs to text file
        cm = compute_confusion_matrix(net, full_test_dataset)
        plot_confusion_matrix(cm, "Final Global Model: Predicting DCA Visits")
        plot_all(round_accuracies, round_recalls, round_f1s, round_aucs, "FL Metrics Per Round")

        plot_shap_feature_importance(
            model=net,
            dataset=full_test_dataset,
            feature_names=feature_names,
            title="SHAP Feature Importance",
        )

        plot_feature_bin_summary_grouped_owen(
            model=net,
            dataset=full_test_dataset,
            max_examples=500,
        )

        plot_feature_summary_grouped_owen(
            model=net,
            dataset=full_test_dataset,
            max_examples=500,
        )

        plot_choropleth(states, finalMetrics["accuracy"], title="Final Round Precision Map", name="choropleth_acc.pdf")
        plot_choropleth(states, finalMetrics["recall"], title="Final Round Recall Map", name="choropleth_rec.pdf")
        plot_choropleth(states, finalMetrics["f1"], title="Final Round F1 Map", name="choropleth_f1.pdf")
        plot_choropleth(states, finalMetrics["auc"], title="Final Round AUC Map", name="choropleth_auc.pdf")

        fp = os.path.join(o, "client_metrics.txt")
        accs = []
        recs = []
        ffs = []
        auccs = []
        # Write per-client metrics to file
        with open(fp, "w") as f:
            f.write(f"Per-client metrics for final global model (round {ro}):\n\n")
            for idx, client_data in enumerate(test_sets):
                # Build TensorDataset for this client
                feats = torch.tensor([x for x, _ in client_data], dtype=torch.float32)
                labs  = torch.tensor([y for _, y in client_data], dtype=torch.long)
                ds    = TensorDataset(feats, labs)

                # Compute metrics
                loss_c, acc_c, recall_c, f1_c, auc_c = evaluate_model(net, ds)

                if 0 <= idx < len(states):
                    stat = states[idx]
                else:
                    stat = "N/A"

                f.write(
                    f"{stat} | "
                    f"loss: {loss_c:.4f}, "
                    f"acc: {acc_c:.4f}, "
                    f"recall: {recall_c:.4f}, "
                    f"F1: {f1_c:.4f}, "
                    f"AUC: {auc_c:.4f}\n"
                )
                accs.append(acc_c)
                recs.append(recall_c)
                ffs.append(f1_c)
                auccs.append(auc_c)
                #Plot State-Specific Data
                if stat in chosen :
                    plot_shap_feature_importance_client(net,ds,feature_names,f"{stat} SHAP Importances",stat)
                    plot_all_client(stat, client_metric_history[stat])


        # Finally, build the scatterplots:
        plot_scatter(states,accs,recs,"Pre/Rec Scatterplot","Precision","Recall", "Scatter1.pdf")
        plot_scatter(states,ffs,auccs,"F1/AUC Scatterplot","F1","AUC", "Scatter2.pdf")
        

        

    # 8) Return metrics for Flower
    return loss, {"accuracy": accuracy, "recall": recall, "f1": f1, "auc": auc}

#Define network model
net = HighwayModel2(n_features, n_classes)
params = ndarrays_to_parameters(get_weights(net))

def server_fn(context: Context): #Federated Averaging, server-side computations
    strategy_no_dp = FedAvg(
        fraction_fit=0.25, #Fraction of avaliable clients selected for training, ~12
        fraction_evaluate=0.4, #Fraction of avaliable clients selected for evaluation
        initial_parameters=params, #initial model weights
        evaluate_fn=evaluate, #function to use for server-side evaluation
        on_fit_config_fn=fit_config, #Epoch decleration
    )
    config=ServerConfig(num_rounds=ro)
    return ServerAppComponents(
        strategy=strategy_no_dp,
        config=config,
    )

#Create an instance of severapp
server = ServerApp(server_fn=server_fn)
# Initiate the simulation passing the server and client apps

# Specify the number of super nodes that will be selected on every round & run as a simulation
run_simulation(
    server_app=server,
    client_app=client,
    num_supernodes=51, #number of clients
    backend_config=backend_setup,
)