"""
Utility functions and classes for Jupyter Notebooks lessons. 
"""

from collections import OrderedDict, defaultdict
from typing import List, Tuple, Dict, Optional
from flwr.common import Metrics, NDArrays, Scalar
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader, random_split, WeightedRandomSampler
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("Agg")        # ← must come before pyplot is imported, suppresses errors when tearing down (which are harmless)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
os.environ["JUPYTER_PLATFORM_DIRS"] = "1"
from sklearn.metrics import confusion_matrix, recall_score, f1_score, roc_auc_score
import seaborn as sns
import numpy as np
import logging
from flwr.common.logger import console_handler, log
from logging import INFO, ERROR
import random
from FLT_Loader2 import n_features, n_classes, cat_sizes
import shap


ro = 50 #Number of rounds
csv_path = '/data/LorenzoData/NFCS2021StateData220627.csv' #CSV Path

feature_names = [
    "Gender/Age Bin",
    "Education",
    "Marital Status",
    "Living Arrangements",
    "Financial Children",
    "Annual Income",
    "Employment",
    "Web/App Help",
    "Stock Investments",
    "Health Insurance",
    "Financial Education",
]

client_metric_history = defaultdict(lambda: {
    "accuracy": [],
    "recall": [],
    "f1": [],
    "auc": [],
})

class InfoFilter(logging.Filter):
    def filter(self, record):
        return record.levelno == INFO


console_handler.setLevel(INFO)
console_handler.addFilter(InfoFilter())

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

# To filter logging coming from the Simulation Engine
# so it's more readable in notebooks
from logging import ERROR
backend_setup = {"init_args": {"logging_level": INFO, "log_to_driver": True, "local_mode": False}} #run everything in-process


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        # 38 inputs → 128 hidden
        self.fc1 = nn.Linear(n_features, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        # x will already be shape [batch, n_features], no flatten needed
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ImprovedModel(nn.Module): #more powerful and agressive with 4 layers and LeakyRelu instead to prevent zeroing of values
    def __init__(self):
        super(ImprovedModel, self).__init__()
        # 38 inputs → 128 hidden, 4 layers
        self.fc1 = nn.Linear(n_features, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, n_classes)
        #Batch Normaliation
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)
        # 30% Dropout to prevent overfitting
        self.dropout = nn.Dropout(0.3)
        #Leaky Relu
        self.act = nn.LeakyReLU(0.1)


    def forward(self, x):
        # x will already be shape [batch, n_features], no flatten needed, go through neural network
        x = self.act(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.act(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.act(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)
        return x




class ResidualModel(nn.Module): #Model with a Skip Connection to learn identity mappings better, stops gradients from vanishing
    def __init__(self):
        super(ResidualModel, self).__init__()
        # first projection
        self.fc_in = nn.Linear(n_features, 128)
        self.bn_in = nn.BatchNorm1d(128)
        # residual block 1: 128 → 128 → 128
        self.res1 = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
        )
        # residual block 2: 128 → 128 → 128
        self.res2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
        )
        # final classifier
        self.fc_out = nn.Linear(128, n_classes)

    def forward(self, x):
        # input → hidden
        x = F.leaky_relu(self.bn_in(self.fc_in(x)), 0.1)
        # block 1 with skip
        r1 = self.res1(x)
        x = F.leaky_relu(x + r1, 0.1)
        # block 2 with skip
        r2 = self.res2(x)
        x = F.leaky_relu(x + r2, 0.1)
        # to logits
        return self.fc_out(x)



class HighwayLayer(nn.Module):
    def __init__(self, size, f=F.relu):
        super().__init__()
        self.transform = nn.Linear(size, size)
        self.gate      = nn.Linear(size, size)
        self.activation = f

    def forward(self, x):
        T = torch.sigmoid(self.gate(x))          # transform gate, a sigmoid between (0,1)
        H = self.activation(self.transform(x))    # candidate transform, the 'new features'
        return H * T + x * (1 - T)                # highway combination, x is the carry input and (1 - T) is the carry gate (opposite of transform kinda)

class HighwayModel(nn.Module): # an evolved version of ResidualModel by learning, for each feature, whether to transform it or carry forward
    def __init__(self, num_highways=2):
        super(HighwayModel, self).__init__()
        self.input = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
        )
        self.highways = nn.ModuleList([
            HighwayLayer(128, f=F.leaky_relu) for _ in range(num_highways)
        ])
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.input(x)
        for hw in self.highways:
            x = hw(x)
            x = self.dropout(x)
        return self.out(x)


class HighwayLayer2(nn.Module):
    def __init__(self, size: int, activation: nn.Module):
        """
        A single highway layer: learns for each dimension whether to transform (H) or carry (x).
        """
        super().__init__()
        self.transform   = nn.Linear(size, size)
        self.gate        = nn.Linear(size, size)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = torch.sigmoid(self.gate(x))           # transform gate
        H = self.activation(self.transform(x))    # transformed features
        return H * T + x * (1 - T)                # combine

class HighwayModel2(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_classes:  int,
        hidden_size: int = 128,
        num_highways: int = 12, #number of layers
        activation:   nn.Module = nn.PReLU(),   # default to PReLU (learnable slope)
        dropout:      float = 0.3,
    ):
        super().__init__()
        # initial projection
        self.input = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.BatchNorm1d(hidden_size),
            activation,
        )
        # stack of highway layers
        self.highways = nn.ModuleList([
            HighwayLayer2(hidden_size, activation)
            for _ in range(num_highways)
        ])
        self.dropout = nn.Dropout(dropout)
        self.out     = nn.Linear(hidden_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input(x)
        for hw in self.highways:
            x = hw(x)
            x = self.dropout(x)
        return self.out(x)




class TabularTransformer(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_classes: int,
        embed_dim: int = 32,
        n_heads: int = 4,
        mlp_ratio: float = 2.0,
        depth: int = 2,
        dropout: float = 0.1,
    ):
        """
        A Transformer-style encoder over features-as-tokens.

        Args:
            n_features: number of input features (continuous columns).
            n_classes: number of output classes.
            embed_dim: embedding dimension per feature token.
            n_heads: number of attention heads.
            mlp_ratio: ratio for the Feed-Forward hidden dim (embed_dim * mlp_ratio).
            depth: number of Transformer encoder layers.
            dropout: dropout rate.
        """
        super().__init__()
        self.n_features = n_features
        # Project each scalar feature into a token embedding
        self.feature_embed = nn.Linear(1, embed_dim)
        # Learnable positional embeddings to distinguish feature indices
        self.pos_embed = nn.Parameter(torch.zeros(1, n_features, embed_dim))
        # Stack of Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,      # since PyTorch 1.9: inputs are (B, S, E)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        # Final classifier: flatten all tokens
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim * n_features),
            nn.Linear(embed_dim * n_features, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, n_features)
        """
        B, F = x.size()
        assert F == self.n_features

        # turn (B, F) → (B, F, 1) → embed → (B, F, embed_dim)
        x = x.unsqueeze(-1)
        x = self.feature_embed(x)
        # add pos embeddings
        x = x + self.pos_embed

        # Transformer over the feature tokens
        # shape remains (B, F, embed_dim)
        x = self.encoder(x)

        # flatten tokens, classify
        x = x.reshape(B, -1)
        return self.classifier(x)


def train_model(model, train_set, epochz):
    batch_size = 64
    num_epochs = epochz

    
    #train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    # ——— handle class imbalance ———
    # 1) Extract all labels from the TensorDataset
    #    train_set.tensors == (features_tensor, labels_tensor)
    labels_tensor = train_set.tensors[1]               # shape [N]

    # 2) Compute class counts and invert
    class_counts = torch.bincount(labels_tensor, minlength=n_classes)
    zero_ids = (class_counts == 0).nonzero().flatten().tolist()
    #if zero_ids:
    #    print(f"[train_model] warning: no data for classes {zero_ids}, setting weight=0")
    class_weights = torch.zeros_like(class_counts, dtype=torch.float)
    # 3) Only invert the nonzero counts:
    nonzero = class_counts > 0
    class_weights[nonzero] = 1.0 / class_counts[nonzero].float()
    #class_weights = class_weights.clamp(max=30)

    # 3a) Weighted loss, boosted to counteract oversampled data
    boosted_weights = (class_weights**1.5).clamp(max=50.0) # boost weight power prevent from going to infinity
    criterion = nn.CrossEntropyLoss(weight=boosted_weights)

    # 3b) (optional) Oversample minority classes via sampler
    sample_weights = class_weights[labels_tensor]       # shape [N]
    #sampler = WeightedRandomSampler(
    #    weights=sample_weights,
    #    num_samples=len(sample_weights),
    #    replacement=True,
    #)

    # Use the sampler instead of shuffle for balanced batches
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        #sampler=sampler,
        shuffle = True,
    )


    #criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    model.train()
    best_loss = float("inf")
    stagnant_epochs = 0
    patience = 5  # stop after 3 non‐improving epochs
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)

        if avg_loss < best_loss:
            best_loss = avg_loss
            stagnant_epochs = 0
            #log(INFO, f"Early Stopper: Improvement Made! Reset patience")
        else:
            stagnant_epochs += 1
            #log(INFO, f"Early Stopper: No improvement for {stagnant_epochs} epoch(s)")
            if stagnant_epochs >= patience:
                #log(INFO, f"Early Stopper: Stopping on epoch {epoch+1}")
                break #stop going through Epochs






def evaluate_model(model, test_set):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0

    allPred = []
    allLab = []
    allProb = []

    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            allPred.extend(predicted.cpu().tolist())
            allLab.extend(labels.cpu().tolist())
            pr = F.softmax(outputs, dim=1)[:, 1] #probabilities for positive class
            allProb.extend(pr.cpu().tolist())

    accuracy = correct / total
    average_loss = total_loss / len(test_loader)
    allLab = [1 if y==1 else 0 for y in allLab] #Make binary for learning objective labels of 1 = Positive cases?
    allPred = [1 if y==1 else 0 for y in allPred]
    recall = recall_score(allLab, allPred, average="binary", pos_label=1)
    f1 = f1_score(allLab, allPred, average="binary", pos_label=1)
    auc = roc_auc_score(allLab, allProb)

    # print(f"Test Accuracy: {accuracy:.4f}, Average Loss: {average_loss:.4f}")
    return average_loss, accuracy, recall, f1, auc


def include_digits(dataset, included_digits):
    including_indices = [
        idx for idx in range(len(dataset)) if dataset[idx][1] in included_digits
    ]
    return torch.utils.data.Subset(dataset, including_indices)


def exclude_digits(dataset, excluded_digits):
    including_indices = [
        idx for idx in range(len(dataset)) if dataset[idx][1] not in excluded_digits
    ]
    return torch.utils.data.Subset(dataset, including_indices)


def compute_confusion_matrix(model, testset):
    # Initialize lists to store true labels and predicted labels
    true_labels = []
    predicted_labels = []

    # Iterate over the test set to get predictions
    for image, label in testset:
        # Forward pass through the model to get predictions
        output = model(image.unsqueeze(0))  # Add batch dimension
        _, predicted = torch.max(output, 1)

        # Append true and predicted labels to lists
        true_labels.append(label)
        predicted_labels.append(predicted.item())

    # Convert lists to numpy arrays
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    return cm


def plot_confusion_matrix(cm, title, path=None):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d",xticklabels = ["Yes", "No"],yticklabels = ["Yes", "No"], linewidths=0.5)
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
    if path is None:
        o = os.path.dirname(csv_path)
    else:
        o = path

    save_path = os.path.join(o, "confusion.pdf")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_accuracy_graph(round_accuracies, title, path=None):
    rounds = list(range(1, len(round_accuracies)))
    fig, ax = plt.subplots(figsize=(max(min(ro/4,14), 4), 4))
    ax.plot(rounds, round_accuracies[1:], marker="o")
    ax.set_title(title)
    ax.set_xlabel("Round")
    ax.set_ylabel("Global Accuracy")
    ax.grid(True)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_xticks(rounds)
    ax.set_xlim(rounds[0], rounds[-1]) #CLAMPER
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    plt.show()
    if path is None:
        o = os.path.dirname(csv_path)
    else:
        o = path

    save_path = os.path.join(o, "accuracy.pdf")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()



def plot_recall_graph(round_recalls, title, path=None):
    rounds = list(range(1, len(round_recalls)))
    fig, ax = plt.subplots(figsize=(max(min(ro/4,14), 4), 4))
    ax.plot(rounds, round_recalls[1:], marker="o")
    ax.set_title(title)
    ax.set_xlabel("Round")
    ax.set_ylabel("Global Recall")
    ax.grid(True)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_xticks(rounds)
    ax.set_xlim(rounds[0], rounds[-1]) #CLAMPER
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    plt.show()
    if path is None:
        o = os.path.dirname(csv_path)
    else:
        o = path

    save_path = os.path.join(o, "recall.pdf")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_f1_graph(round_f1s, title, path=None):
    rounds = list(range(1, len(round_f1s)))
    fig, ax = plt.subplots(figsize=(max(min(ro/4,14), 4), 4))
    ax.plot(rounds, round_f1s[1:], marker="o")
    ax.set_title(title)
    ax.set_xlabel("Round")
    ax.set_ylabel("Global F1")
    ax.grid(True)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_xticks(rounds)
    ax.set_xlim(rounds[0], rounds[-1]) #CLAMPER
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    plt.show()
    if path is None:
        o = os.path.dirname(csv_path)
    else:
        o = path

    save_path = os.path.join(o, "f1.pdf")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_auc_graph(round_aucs, title, path=None):
    rounds = list(range(1, len(round_aucs)))
    fig, ax = plt.subplots(figsize=(max(min(ro/4,14), 4), 4))
    ax.plot(rounds, round_aucs[1:], marker="o")
    ax.set_title(title)
    ax.set_xlabel("Round")
    ax.set_ylabel("Global AUC")
    ax.grid(True)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_xticks(rounds)
    ax.set_xlim(rounds[0], rounds[-1]) #CLAMPER
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    plt.show()
    if path is None:
        o = os.path.dirname(csv_path)
    else:
        o = path

    save_path = os.path.join(o, "auc.pdf")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_all(round_accuracies, round_recalls, round_f1s, round_aucs, title, path=None):
    rounds = list(range(1, len(round_f1s)))
    fig, ax = plt.subplots(figsize=(max(min(ro/4,14), 4), 4))
    ax.plot(rounds, round_accuracies[1:], marker="o", label="Precision", color="tab:red")
    ax.plot(rounds, round_recalls[1:], marker="s", label="Recall", color="tab:blue")
    ax.plot(rounds, round_f1s[1:], marker="^", label="F1", color="tab:green")
    ax.plot(rounds, round_aucs[1:], marker="d", label="AUC", color="tab:purple")
    ax.set_title(title)
    ax.set_xlabel("Round")
    ax.set_ylabel("Global Metric")
    ax.grid(True)
    ax.legend(loc="lower left", title="Metrics")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_xticks(rounds)
    ax.set_xlim(rounds[0], rounds[-1]) #CLAMPER
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    plt.show()
    o = None
    if path is None:
        o = os.path.dirname(csv_path)
    else:
        o = path



    save_path = os.path.join(o, "graph.pdf")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_all_client(
    state_name: str,
    history: Dict[str, List[float]],
    path: Optional[str] = None,
) -> None:
    """
    Plot accuracy, recall, F1 and AUC for a single state over rounds.
    `history` should be a dict with keys "accuracy","recall","f1","auc"
    each mapping to a list of length = num_rounds+1 (including round 0).
    """
    # Rounds 0,1,2,...,N
    rounds = list(range(1, len(history["accuracy"])))
    fig, ax = plt.subplots(figsize=(min(len(rounds)/4, 14), 4))
    ax.plot(rounds, history["accuracy"][1:], marker="o", label="Accuracy")
    ax.plot(rounds, history["recall"][1:],    marker="s", label="Recall")
    ax.plot(rounds, history["f1"][1:],        marker="^", label="F1")
    ax.plot(rounds, history["auc"][1:],       marker="d", label="AUC")

    ax.set_title(f"{state_name} Metrics Per Round")
    ax.set_xlabel("Round")
    ax.set_ylabel("Value")
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.grid(True)
    ax.legend(loc="lower right")

    # Save if requested
    if path is None:
        o = os.path.dirname(csv_path)
    else:
        o = path

    save_path = os.path.join(o, f"{state_name}_metrics.pdf")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_shap_feature_importance(
    model: nn.Module,
    dataset: torch.utils.data.TensorDataset,
    feature_names: List[str],
    title: str,
    path: Optional[str] = None,
    background_size: int = 100,
    nsample: int = 100,
) -> None:
    """
    Compute SHAP values (for the *positive* class in a binary setting)
    and bar-plot the sum of mean |SHAP| over each one-hot group.
    """
    o = None
    if path is None:
        o = os.path.dirname(csv_path)
    else:
        o = path
    path = o

    # 1) grab all features as a NumPy array
    X = dataset.tensors[0].cpu().numpy()  # shape (N, D)
    # 2) sample a small “background” set
    if background_size < X.shape[0]:
        idx = np.random.choice(len(X), background_size, replace=False)
        background = X[idx]
    else:
        background = X

    # 3) define a prediction function returning probability of class “1”
    def f(X_batch: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            logits = model(torch.from_numpy(X_batch).float())
            probs = torch.softmax(logits, dim=1)[:, 1]      # positive-class prob
        return probs.cpu().numpy()

    # 4) kernel explainer
    explainer = shap.KernelExplainer(f, background)
    # 5) compute shap values on the *background* itself (or a small subset)
    shap_vals = explainer.shap_values(background, nsamples=nsample)
    # shap_vals is a list (one per output class), for binary shap_vals[1] is what we want:
    v = np.abs(shap_vals) if not isinstance(shap_vals, list) else np.abs(shap_vals[1])

    # 6) collapse one-hot groups down to 11 features
    #    => sum mean‐abs across each group
    groups = []
    start = 0
    for size in cat_sizes:  # cat_sizes = [12,7,6,5,7,11,9,4,3,3,4]
        group_mean = v[:, start : start + size].mean(axis=0).sum()
        groups.append(group_mean)
        start += size

    # 7) plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(feature_names, groups)
    ax.set_xticklabels(feature_names, rotation=45, ha="right")
    ax.set_title(title)
    ax.set_ylabel("Sum of mean |SHAP|")
    ax.grid(True, axis="y")
    fig.tight_layout()

    os.makedirs(path, exist_ok=True)
    fname = os.path.join(path, "feature_importance.pdf")
    fig.savefig(fname, bbox_inches="tight")

    plt.close(fig)


def plot_shap_feature_importance_client(
    model: nn.Module,
    dataset: torch.utils.data.TensorDataset,
    feature_names: List[str],
    title: str,
    state_name: Optional[str] = None,        # NEW: which client/state
    path: Optional[str] = None,
    background_size: int = 100,
    nsample: int = 100,
) -> None:
    """
    Compute SHAP values (for the *positive* class) and
    return & (if requested) plot the sum of mean |SHAP| over each one-hot group.
    If `state_name` is given, we also record groups into client_shap_history[state_name].
    """
    # ——— determine output folder ———
    if path is None:
        out_dir = os.path.dirname(csv_path)
    else:
        out_dir = path
    os.makedirs(out_dir, exist_ok=True)

    # ——— 1) grab all features as NumPy ———
    X = dataset.tensors[0].cpu().numpy()  # shape (N, D)

    # ——— 2) sample background ———
    if background_size < len(X):
        idx = np.random.choice(len(X), background_size, replace=False)
        background = X[idx]
    else:
        background = X

    # ——— 3) define positive‐class probability fn ———
    def f(X_batch: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            logits = model(torch.from_numpy(X_batch).float())
            return torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

    # ——— 4–5) explain on background ———
    explainer  = shap.KernelExplainer(f, background)
    shap_vals  = explainer.shap_values(background, nsamples=nsample)
    v = (
        np.abs(shap_vals)
        if not isinstance(shap_vals, list)
        else np.abs(shap_vals[1])
    )

    # ——— 6) collapse one‐hot groups into 11 real features ———
    groups = []
    start = 0
    for size in cat_sizes:  # e.g. [12,7,6,5,7,11,9,4,3,3,4]
        groups.append(v[:, start : start + size].mean(axis=0).sum())
        start += size

    # ——— 7) plot + save ———
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(feature_names, groups)
    ax.set_xticklabels(feature_names, rotation=45, ha="right")
    ax.set_title(title)
    ax.set_ylabel("Sum of mean |SHAP|")
    ax.grid(True, axis="y")
    fig.tight_layout()

    fname = os.path.join(out_dir, f"{state_name}_feature_importance.pdf")
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)





def plot_scatter(
    state_names: list[str],
    accs: list[float],
    recs: list[float],
    title: str,
    xName: str,
    yName: str,
    grName: str,
    path: Optional[str] = None,
) -> None:
    """
    Scatterplot of per-client.

    Args:
        state_names: List of client labels (e.g., state names).
        accs: List of x values.
        recs: List of y values.
        title: Plot title.
        path: Directory to save the plot. If None, skips saving.
    """

    o = None
    if path is None:
        o = os.path.dirname(csv_path)
    else:
        o = path
    path = o

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(accs, recs)

    # Annotate each point
    for x, y, label in zip(accs, recs, state_names):
        ax.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=(5, 5),
            ha="right",
            fontsize=8,
        )

    ax.set_xlabel(xName)
    ax.set_ylabel(yName)
    ax.set_title(title)
    ax.grid(True)
    fig.tight_layout()

    os.makedirs(path, exist_ok=True)
    fp = os.path.join(path, grName)
    fig.savefig(fp, bbox_inches="tight")
    plt.close(fig)