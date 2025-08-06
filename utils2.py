"""
Utility functions and classes used in Server for brunt of computation, including FL network, round/client specifications, and graph creation
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
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
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
import math
import pandas as pd
import plotly.express as px
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform

ro = 200 #Number of rounds
csv_path = '/data/LorenzoData/NFCS2021StateData220627.csv' #CSV Path for data reading

feature_names = [
    "Gender/Age Bin by Gender",
    "Gender/Age Bin by Age",
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

finalMetrics = {
    "accuracy": [],
    "recall":   [],
    "f1":       [],
    "auc":      [],
}


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
backend_setup = {"init_args": {"logging_level": ERROR, "log_to_driver": False, "local_mode": False}} #run everything in-process


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

class HighwayModel2(nn.Module): #Highway model implementation
    def __init__(
        self,
        n_features: int,
        n_classes:  int,
        hidden_size: int = 128,
        num_highways: int = 8, #number of layers
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


def train_model(model, train_set, epochz): #code to handle model training
    batch_size = 64
    num_epochs = epochz

    # ——— handle class imbalance ———
    # 1) Extract all labels from the TensorDataset
    labels_tensor = train_set.tensors[1]               # shape [N]

    # 2) Compute class counts and invert
    class_counts = torch.bincount(labels_tensor, minlength=n_classes)
    zero_ids = (class_counts == 0).nonzero().flatten().tolist()
    class_weights = torch.zeros_like(class_counts, dtype=torch.float)

    # 3) Only invert the nonzero counts:
    nonzero = class_counts > 0
    class_weights[nonzero] = 1.0 / class_counts[nonzero].float()

    boosted_weights = (class_weights**1.1835).clamp(max=100.0) #Weight boosting

    criterion = nn.CrossEntropyLoss(weight=boosted_weights)

    # Use the sampler instead of shuffle for balanced batches
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle = True,
    )

    optimizer = optim.Adam(model.parameters(), lr=0.005) #Adam optimizer

    model.train()
    best_loss = float("inf")
    stagnant_epochs = 0
    patience = 5  # Epoch Stopping: stop after 5 non‐improving epochs
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
        else:
            stagnant_epochs += 1
            if stagnant_epochs >= patience:
                break #stop going through Epochs

def evaluate_model(model, test_set): #Model evaluation code, calculates 4 metrics per round as well as average loss
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

    return average_loss, accuracy, recall, f1, auc

def compute_confusion_matrix(model, testset): #Function to compute matrix used in confusion plot
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


def plot_confusion_matrix(cm, title, path=None): #Plot using matrix calculated above
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

def plot_all(round_accuracies, round_recalls, round_f1s, round_aucs, title, path=None): #Plot all metrics of global model across all rounds
    step = max(1, math.ceil(ro//25)) #Step over some rounds for spacing
    rounds = list(range(1, len(round_f1s), step))
    # Subsample metrics to match 'rounds'
    accs = round_accuracies[1::step]
    recalls = round_recalls[1::step]
    f1s = round_f1s[1::step]
    aucs = round_aucs[1::step]

    fig, ax = plt.subplots(figsize=(max(min(lon(rounds)/4,14), 4), 4))
    ax.plot(rounds, accs, marker="o", label="Precision", color="tab:red")
    ax.plot(rounds, recalls, marker="s", label="Recall", color="tab:blue")
    ax.plot(rounds, f1s, marker="^", label="F1", color="tab:green")
    ax.plot(rounds, aucs, marker="d", label="AUC", color="tab:purple")
    ax.set_title(title)
    ax.set_xlabel("Round")
    ax.set_ylabel("Global Metric")
    ax.grid(True)
    ax.legend(loc="lower right", title="Metrics")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(max(1, math.ceil(ro//25)))) #Instead of every round do every tot/25 rounds
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

def plot_all_client( #Client-specific metric plot
    state_name: str,
    history: Dict[str, List[float]],
    path: Optional[str] = None,
) -> None:
    
    step = max(1, math.ceil(ro//25))
    rounds = list(range(1, len(history["accuracy"]), step))
    # Subsample metrics to match 'rounds'
    accs = history["accuracy"][1::step]
    recalls = history["recall"][1::step]
    f1s = history["f1"][1::step]
    aucs = history["auc"][1::step]

    fig, ax = plt.subplots(figsize=(min(len(rounds)/4, 14), 4))
    ax.plot(rounds, accs, marker="o", label="Precision")
    ax.plot(rounds, recalls,    marker="s", label="Recall")
    ax.plot(rounds, f1s,        marker="^", label="F1")
    ax.plot(rounds, aucs,       marker="d", label="AUC")

    ax.set_title(f"{state_name} Metrics Per Round")
    ax.set_xlabel("Round")
    ax.set_ylabel("Value")
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(max(1, math.ceil(ro//25))))
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


def plot_shap_feature_importance( #Plot SHAP feature importance and STD for global model
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
    #v = shap_vals if not isinstance(shap_vals, list) else shap_vals[1]

    # 6) collapse one-hot groups down to 11 features
    #    => sum mean‐abs across each group
    groups = []
    stds = []
    start = 0

    for size in cat_sizes:  # cat_sizes = [12,12,7,6,5,7,11,9,4,3,3,4]
        group_sample = v[:, start : start + size]
        groups.append(group_sample.mean(axis=0).sum())
        stds.append(group_sample.std(axis=0).sum())
        start += size

    assert len(groups) == len(feature_names), "Mismatch groups vs feature_names"

    covs = [#Relative variation, is probably not to-scale due to very large STD value compared to SHAP numbers
        (s / m) if m != 0 else 0.0
        for m, s in zip(groups, stds)
    ]
    #covs = covs / max(covs) #optional to make as unit figure


    # 7) plot
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(
    feature_names,
    groups,
)
    ax.set_xticklabels(feature_names, rotation=45, ha="right")
    ax.set_title(title)
    ax.set_ylabel("Sum of mean SHAP")
    ax.grid(True, axis="y")
    fig.tight_layout()

    #8) put the numeric std above each bar
    for bar, s in zip(bars, covs):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.01 * (ax.get_ylim()[1] - ax.get_ylim()[0]),  # a little padding
            f"{s:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    os.makedirs(path, exist_ok=True)
    fname = os.path.join(path, "feature_importance.pdf")
    fig.savefig(fname, bbox_inches="tight")

    plt.close(fig)


def plot_shap_feature_importance_client( #Client-specific SHAP plot
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

    # ——— 6) collapse one‐hot groups into 12 real features ———
    groups = []
    stds = []
    start = 0
    for size in cat_sizes:  # e.g. [12,12,7,6,5,7,11,9,4,3,3,4]
        groups.append(v[:, start : start + size].mean(axis=0).sum())
        stds.append(v[:, start : start + size].std(axis=0).sum())
        start += size

    covs = [#Relative variation
        (s / m) if m != 0 else 0.0
        for m, s in zip(groups, stds)
    ]
    #covs = covs / max(covs) #make as a unit figure?

    # ——— 7) plot + save ———
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(
    feature_names,
    groups,
)
    ax.set_xticklabels(feature_names, rotation=45, ha="right")
    ax.set_title(title)
    ax.set_ylabel("Sum of mean |SHAP|")
    ax.grid(True, axis="y")
    fig.tight_layout()

    #8) put the numeric std above each bar
    for bar, s in zip(bars, covs):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.01 * (ax.get_ylim()[1] - ax.get_ylim()[0]),  # a little padding
            f"{s:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fname = os.path.join(out_dir, f"{state_name}_feature_importance.pdf")
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)

def plot_scatter( #Scatterplot, used for Client Plotting to see general trend and outliers
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

def plot_choropleth( #Basically same thing as scatterplot but cooler
    state_names: list[str],
    vals: list[float],
    title: str = "Federated Learning by State",
    path: Optional[str] = None,
    name: Optional[str] = "choropleth",
) -> None:
    """
    Draw and save a U.S. choropleth map where each state is colored
    by its accuracy value.

    Args:
        state_names: Full state names + "District of Columbia".
        vals: Same-length list of floats in [0,1].
        title: Title for the map.
        path: Directory to save the output. Defaults next to csv_path.
    """
    if path is None:
        out_dir = os.path.dirname(csv_path)
    else:
        out_dir = path
    os.makedirs(out_dir, exist_ok=True)
    # 1) Map full state names to postal codes
    state_abbrev = {
        'Alabama':'AL','Alaska':'AK','Arizona':'AZ','Arkansas':'AR','California':'CA',
        'Colorado':'CO','Connecticut':'CT','Delaware':'DE','District of Columbia':'DC',
        'Florida':'FL','Georgia':'GA','Hawaii':'HI','Idaho':'ID','Illinois':'IL',
        'Indiana':'IN','Iowa':'IA','Kansas':'KS','Kentucky':'KY','Louisiana':'LA',
        'Maine':'ME','Maryland':'MD','Massachusetts':'MA','Michigan':'MI',
        'Minnesota':'MN','Mississippi':'MS','Missouri':'MO','Montana':'MT',
        'Nebraska':'NE','Nevada':'NV','New Hampshire':'NH','New Jersey':'NJ',
        'New Mexico':'NM','New York':'NY','North Carolina':'NC','North Dakota':'ND',
        'Ohio':'OH','Oklahoma':'OK','Oregon':'OR','Pennsylvania':'PA',
        'Rhode Island':'RI','South Carolina':'SC','South Dakota':'SD',
        'Tennessee':'TN','Texas':'TX','Utah':'UT','Vermont':'VT','Virginia':'VA',
        'Washington':'WA','West Virginia':'WV','Wisconsin':'WI','Wyoming':'WY'
    }

    df = pd.DataFrame({
        "state_name": state_names,
        "abbr": [state_abbrev.get(s, "") for s in state_names],
        "Percentage": vals,
    })
    if df["abbr"].eq("").any():
        missing = df.loc[df["abbr"]=="", "state_name"].tolist()
        raise ValueError(f"Unknown state names: {missing}")

    # 2) Build and show the choropleth
    lo, hi = df["Percentage"].min()-0.005, df["Percentage"].max()+0.005 #Clamping Values
    ticks = np.linspace(lo, hi, 6)  # 6 labels incl. top
    fig = px.choropleth(
        df,
        locations="abbr",
        locationmode="USA-states",
        color="Percentage",
        scope="usa",
        color_continuous_scale="Viridis",
        range_color=(lo, hi),
        #labels={"accuracy": "Accuracy"},
        title=title,
    )

    fig.update_layout(
        coloraxis_colorbar=dict(
            title="Percentage",
            tickmode="array",
            tickvals=ticks,
            ticktext=[f"{t:.3f}" for t in ticks],  # or format as %: f"{t:.1%}"
            ticks="outside",
        )
    )

    # 3) Save to file
    save_path = os.path.join(out_dir, name)
    #try:
    #    fig.write_image(save_path, width=800, height=500)
    #except Exception as e:
    html_path = os.path.splitext(save_path)[0] + ".html"
    fig.write_html(html_path)

def plot_feature_bin_summary_grouped_owen( #Plots Feature Summary using Owen Values instead, per each Feature bin
    model: nn.Module,
    dataset: torch.utils.data.TensorDataset,
    max_examples: int = 2000,
    background_size: int = 200,
    nsamples: int = 200,
    path: Optional[str] = None,
    max_display: Optional[int] = None,
    file_name_prefix: str = "shap_summary_owen",
    base_cmap: str = "tab20",
) -> None:

    groupings = [
        ["Gender/Age Bin by Gender", "Gender/Age Bin by Age", "Marital Status", "Living Arrangements"],
        ["Employment", "Web/App Help", "Health Insurance", "Annual Income"],
        ["Education", "Financial Children", "Stock Investments", "Financial Education"],
    ]

    def generate_detailed_feature_names(cat_sizes, feature_names):
        all_feature_names = []
        for group_name, size in zip(feature_names, cat_sizes):
            for i in range(size):
                all_feature_names.append(f"{group_name}__{i}")

            # Step 2: Create a mapping from old names to new, descriptive names
        rename_map = {
            "Gender/Age Bin by Gender__0": "Male 18-24 (Gender)",
            "Gender/Age Bin by Gender__1": "Male 25-34 (Gender)",
            "Gender/Age Bin by Gender__2": "Male 35-44 (Gender)",
            "Gender/Age Bin by Gender__3": "Male 45-54 (Gender)",
            "Gender/Age Bin by Gender__4": "Male 55-64 (Gender)",
            "Gender/Age Bin by Gender__5": "Male 65+ (Gender)",
            "Gender/Age Bin by Gender__6": "Female 18-24 (Gender)",
            "Gender/Age Bin by Gender__7": "Female 25-34 (Gender)",
            "Gender/Age Bin by Gender__8": "Female 35-44 (Gender)",
            "Gender/Age Bin by Gender__9": "Female 45-54 (Gender)",
            "Gender/Age Bin by Gender__10": "Female 55-64 (Gender)",
            "Gender/Age Bin by Gender__11": "Female 65+ (Gender)",
            "Gender/Age Bin by Age__0": "Male 18-24 (Age)",
            "Gender/Age Bin by Age__1": "Female 18-24 (Age)",
            "Gender/Age Bin by Age__2": "Male 25-34 (Age)",
            "Gender/Age Bin by Age__3": "Female 25-34 (Age)",
            "Gender/Age Bin by Age__4": "Male 35-44 (Age)",
            "Gender/Age Bin by Age__5": "Female 35-44 (Age)",
            "Gender/Age Bin by Age__6": "Male 45-54 (Age)",
            "Gender/Age Bin by Age__7": "Female 45-54 (Age)",
            "Gender/Age Bin by Age__8": "Male 55-64 (Age)",
            "Gender/Age Bin by Age__9": "Female 55-64 (Age)",
            "Gender/Age Bin by Age__10": "Male 65+ (Age)",
            "Gender/Age Bin by Age__11": "Female 65+ (Age)",
            # ... continue for all your other feature names ...
            "Education__0": "Did not complete high school",
            "Education__1": "High school graduate - regular diploma",
            "Education__2": "High school graduate - GED or alternative credential",
            "Education__3": "Some college, no degree",
            "Education__4": "Associate's degree",
            "Education__5": "Bachelor's degree",
            "Education__6": "Post graduate degree",
            "Education__7": "Prefer not to say (Education)",
            "Living Arrangements__0": "I am only adult in household",
            "Living Arrangements__1": "I live with my spouse/significant other",
            "Living Arrangements__2": "I live in my parent's home",
            "Living Arrangements__3": "I live with other family, friends, or roomates",
            "Living Arrangements__4": "Prefer not to say (Living Arrangements)",
            "Financial Children__0": "I have 1 child who is financially dependent",
            "Financial Children__1": "I have 2 children who are financially dependent",
            "Financial Children__2": "I have 3 children who are financially dependent",
            "Financial Children__3": "I have 4+ children who are financially dependent",
            "Financial Children__4": "I have 0 children who are financially dependent",
            "Financial Children__5": "I do not have children",
            "Financial Children__6": "Prefer not to say (Financial Children)", 
            "Annual Income__0": "Annual Income < $15,000",
            "Annual Income__1": "$15,000 <= Annual Income < $25,000",
            "Annual Income__2": "$25,000 <= Annual Income < $35,000",
            "Annual Income__3": "$35,000 <= Annual Income < $50,000",
            "Annual Income__4": "$50,000 <= Annual Income < $75,000",
            "Annual Income__5": "$75,000 <= Annual Income < $100,000",
            "Annual Income__6": "$100,000 <= Annual Income < $150,000",
            "Annual Income__7": "$150,000 <= Annual Income < $200,000",
            "Annual Income__8": "$200,000 <= Annual Income < $300,000",
            "Annual Income__9": "$300,000 <= Annual Income",
            "Annual Income__10": "I don't know my Annual Income",
            "Annual Income__11": "Prefer not to say (Annual Income)",
            "Employment__0": "Self-Employed",
            "Employment__1": "Work full-time for employer or military",
            "Employment__2": "Work part-time for employer or military",
            "Employment__3": "Homemaker",
            "Employment__4": "Full-time student",
            "Employment__5": "Permanently unable to work",
            "Employment__6": "Unemployed or laid off",
            "Employment__7": "Retired",
            "Employment__8": "Prefer not to say (Employment)",
            "Web/App Help__0": "I frequently get Web/App Help",
            "Web/App Help__1": "I sometimes get Web/App Help",
            "Web/App Help__2": "I never get Web/App Help",
            "Web/App Help__3": "I don't know if I get Web/App Help",
            "Web/App Help__4": "Prefer not to say (Web/App Help)", 
            "Stock Investments__0": "I have investments in stocks/bonds/mutual funds",
            "Stock Investments__1": "I do not have investments in stocks/bonds/mutual funds",
            "Stock Investments__2": "I do not know if I have investments in stocks/bonds/mutual funds",
            "Stock Investments__3": "Prefer not to say (Stock Investments)",
            "Health Insurance__0": "I have Health Insurance",
            "Health Insurance__1": "I do not have Health Insurance",
            "Health Insurance__2": "I don't know if I have Health Insurance",
            "Health Insurance__3": "Prefer not to say (Health Insurance)",
            "Financial Education__0": "My school had financial education but I did not attend",
            "Financial Education__1": "My school had financial education and I attended",
            "Financial Education__2": "My school did not have financial education",
            "Financial Education__3": "I don't know if my school had financial education",
            "Financial Education__4": "Prefer not to say (Financial Education)",
            "Marital Status__0": "?arried",
            "Marital Status__1": "Single",
            "Marital Status__2": "Separated",
            "Marital Status__3": "Divorced",
            "Marital Status__4": "Widowed/Widower",
            "Marital Status__5": "Prefer not to say (Marital Status)",
        }

        # Step 3: Create and return a new list with the renamed features.
        # The .get(name, name) method looks up the new name in the map.
        # If a name isn't in the map, it defaults to using the original name.
        renamed_features = [rename_map.get(name, name) for name in all_feature_names]

        return renamed_features

    all_feature_names = generate_detailed_feature_names(cat_sizes, feature_names)

    if path is None:
        path = os.path.dirname(csv_path)

    # Sample data
    X = dataset.tensors[0].cpu().numpy()
    X = X[:max_examples]
    if background_size < len(X):
        idx = np.random.choice(len(X), background_size, replace=False)
        background = X[idx]
    else:
        background = X

    X_df = pd.DataFrame(X)
    background_df = pd.DataFrame(background)

    # Feature clustering
    corr = X_df.corr().fillna(0)
    distance = 1 - np.abs(corr)
    distance = distance.replace([np.inf, -np.inf], 1).fillna(1)
    linkage = sch.linkage(squareform(distance.values, checks=False), method="ward")

    # SHAP Partition masker
    masker = shap.maskers.Partition(X_df, clustering=linkage)

    # Prediction function
    def predict(X_numpy):
        with torch.no_grad():
            inputs = torch.from_numpy(X_numpy).float()
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)
            return probs[:, 1].cpu().numpy()

    explainer = shap.explainers.Partition(predict, masker)
    shap_values = explainer(X_df, max_evals=nsamples)
    shap_values.feature_names = all_feature_names

    # Plot by group
    start_idx = 0
    for i, group in enumerate(groupings):
        indices = []
        for g in group:
            base_idx = feature_names.index(g)
            group_size = cat_sizes[base_idx]
            indices.extend(range(start_idx, start_idx + group_size))
            start_idx += group_size

        # Subset manually
        values = shap_values.values[:, indices]
        data = shap_values.data[:, indices]
        names = [shap_values.feature_names[i] for i in indices]

        # Create new explanation object
        subset = shap.Explanation(
            values,
            data=data,
            feature_names=names,
            base_values=shap_values.base_values,
        )

        plt.figure(figsize=(10, 6))
        shap.plots.beeswarm(subset, max_display=max_display, color=plt.get_cmap("coolwarm"))
        out_path = os.path.join(path, f"{file_name_prefix}_group{i+1}.pdf")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()

def plot_feature_summary_grouped_owen( #Plots owen feature summary for overall features instead of feature bins as above, basically aggregates them and puts them similar to SHAP
    model: nn.Module,
    dataset: torch.utils.data.TensorDataset,
    max_examples: int = 2000,
    background_size: int = 200,
    nsamples: int = 200,
    path: Optional[str] = None,
    max_display: Optional[int] = None,
    file_name: str = "shap_summary_owen.pdf",
    base_cmap: str = "tab20",
) -> None:
    if path is None:
        path = os.path.dirname(csv_path)

    # Sample data
    X = dataset.tensors[0].cpu().numpy()
    X = X[:max_examples] #beeswarm
    if background_size < len(X):
        idx = np.random.choice(len(X), background_size, replace=False)
        background = X[idx]
    else:
        background = X

    # Convert background and X to DataFrame for clustering
    X_df = pd.DataFrame(X)
    background_df = pd.DataFrame(background)

    # Step 1: Create correlation-based feature clustering
    corr = X_df.corr()
    # Replace NaNs with 0
    corr = corr.fillna(0)
    distance = 1 - np.abs(corr)
    # Clip any remaining NaNs
    distance = distance.replace([np.inf, -np.inf], 1)
    distance = distance.fillna(1)
    condensed_distance = squareform(distance.values, checks=False)
    linkage = sch.linkage(squareform(condensed_distance), method="ward")


    # Step 2: Create SHAP Partition masker with clustering
    masker = shap.maskers.Partition(X_df, clustering=linkage)

    # Step 3: Define prediction function
    def predict(X_numpy):
        with torch.no_grad():
            inputs = torch.from_numpy(X_numpy).float()
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)
            return probs[:, 1].cpu().numpy()  # Class 1 probability

    # Step 4: Create Owen-value-based Partition explainer
    explainer = shap.explainers.Partition(predict, masker)#creates a Partition SHAP/Owen explainer using your prediction function and feature clustering.
    shap_values = explainer(X_df, max_evals=nsamples)# actually runs the Owen value computation for each sample in X_df.

    #Step 4b: Group the Owen values categorically (by each feature)
    grouped_vals = group_owen_values(shap_values.values)
    # Melt grouped values for seaborn
    records = []
    for i, fname in enumerate(feature_names):
        for val in grouped_vals[:, i]:
            records.append((fname, val))
    df = pd.DataFrame(records, columns=["Feature", "Owen Value"])

    # Step 5: Plot Owen-value-based SHAP summary (beeswarm)
    # Beeswarm plot (violin-based)
    plt.figure(figsize=(10, 6))
    sns.violinplot(
        data=df,
        y="Feature",
        x="Owen Value",
        scale="width",
        inner="point",
        palette=base_cmap,
        linewidth=1
    )
    plt.title("Grouped Owen Value Summary Plot")
    plt.tight_layout()

    # Save to PDF
    out_path = os.path.join(path, file_name)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def plotCommunication( #Communication Costs Graph, kinda bad but it's generally used to see values
    client_costs: List[Dict[str, float]],
    path: Optional[str] = None,
    file_name: str = "client_communication_costs.pdf",
    in_mb: bool = False
) -> None:
    """
    Plots a stacked bar chart of upload and download communication costs per client.

    Args:
        client_costs: A list of dicts where each dict has 'upload' and 'download' in KB.
        path: Optional output path. Defaults to current directory.
        file_name: Output PDF filename.
        in_mb: If True, converts KB to MB for plotting.
    """
    state_abbrev = {
        'Alabama':'AL','Alaska':'AK','Arizona':'AZ','Arkansas':'AR','California':'CA',
        'Colorado':'CO','Connecticut':'CT','Delaware':'DE','District of Columbia':'DC',
        'Florida':'FL','Georgia':'GA','Hawaii':'HI','Idaho':'ID','Illinois':'IL',
        'Indiana':'IN','Iowa':'IA','Kansas':'KS','Kentucky':'KY','Louisiana':'LA',
        'Maine':'ME','Maryland':'MD','Massachusetts':'MA','Michigan':'MI',
        'Minnesota':'MN','Mississippi':'MS','Missouri':'MO','Montana':'MT',
        'Nebraska':'NE','Nevada':'NV','New Hampshire':'NH','New Jersey':'NJ',
        'New Mexico':'NM','New York':'NY','North Carolina':'NC','North Dakota':'ND',
        'Ohio':'OH','Oklahoma':'OK','Oregon':'OR','Pennsylvania':'PA',
        'Rhode Island':'RI','South Carolina':'SC','South Dakota':'SD',
        'Tennessee':'TN','Texas':'TX','Utah':'UT','Vermont':'VT','Virginia':'VA',
        'Washington':'WA','West Virginia':'WV','Wisconsin':'WI','Wyoming':'WY'
    }
    unit = "MB" if in_mb else "KB"
    scale = 1 / 1024 if in_mb else 1

    # Generate state abbreviation labels
    state_labels = list(state_abbrev.values())
    if path is None:
        path = os.path.dirname(csv_path)

    # Get corresponding abbreviations (used as x-axis labels)
    clients = list(range(len(client_costs)))
    uploads = [client_costs[c]["upload"] * scale for c in clients]
    #downloads = [client_costs[c]["download"] * scale for c in clients]

    plt.figure(figsize=(18, 6))
    plt.bar(clients, uploads, label="1-Way Communication Cost", alpha=0.7)
    #plt.bar(clients, downloads, label="Download", alpha=0.7, bottom=uploads)
    plt.xlabel("Client ID")
    plt.ylabel(f"Communication ({unit})")
    plt.title("Communication Cost per Client")
    # Set x-axis ticks to state abbreviations
    plt.xticks(clients, state_labels, rotation=90, fontsize=8)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(path, file_name)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()