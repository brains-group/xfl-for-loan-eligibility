# utils.py
from collections import OrderedDict
from typing import List, Tuple, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
from sklearn.metrics import confusion_matrix, recall_score, f1_score, roc_auc_score
import seaborn as sns
import numpy as np
import math
import shap
from FLT_LoaderC import *

# --- Localized Configuration ---
r = 200 # number of epochs
CSV_PATH = '/data/NFCS2021StateData220627.csv' # pathway to CSV file, update as necessary

feature_names = [
    "Gender/Age Bin by Gender", "Gender/Age Bin by Age", "Education",
    "Marital Status", "Living Arrangements", "Financial Children",
    "Annual Income", "Employment", "Web/App Help",
    "Stock Investments", "Health Insurance", "Financial Education",
]

# Model Definition (Highway Network)
class HighwayLayer2(nn.Module):
    def __init__(self, size: int, activation: nn.Module):
        super().__init__()
        self.transform = nn.Linear(size, size)
        self.gate = nn.Linear(size, size)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = torch.sigmoid(self.gate(x))
        H = self.activation(self.transform(x))
        return H * T + x * (1 - T)

class HighwayModel2(nn.Module):
    def __init__(
        self,
        n_features: int, n_classes: int, hidden_size: int = 128,
        num_highways: int = 8, activation: nn.Module = nn.PReLU(),
        dropout: float = 0.5,
    ):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.BatchNorm1d(hidden_size),
            activation,
        )
        self.highways = nn.ModuleList(
            [HighwayLayer2(hidden_size, activation) for _ in range(num_highways)]
        )
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input(x)
        for hw in self.highways:
            x = hw(x)
            x = self.dropout(x)
        return self.out(x)

# Training and Evaluation Functions
def train_model(model, train_set, epochs_to_run):
    batch_size = 64
    labels_tensor = train_set.tensors[1]
    class_counts = torch.bincount(labels_tensor, minlength=n_classes)
    
    class_weights = torch.zeros_like(class_counts, dtype=torch.float)
    nonzero = class_counts > 0
    class_weights[nonzero] = 1.0 / class_counts[nonzero].float() # class weighting
    boosted_weights = (class_weights**1.182).clamp(max=100.0) # weight factor

    criterion = nn.CrossEntropyLoss(weight=boosted_weights)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(epochs_to_run):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        print(f"  Training Loss: {avg_loss:.4f}")

def evaluate_model(model, test_set):
    model.eval()
    correct, total, total_loss = 0, 0, 0
    allPred, allLab, allProb = [], [], []

    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += criterion(outputs, labels).item()

            allPred.extend(predicted.cpu().tolist())
            allLab.extend(labels.cpu().tolist())
            pr = F.softmax(outputs, dim=1)[:, 1]
            allProb.extend(pr.cpu().tolist())

    accuracy = correct / total
    average_loss = total_loss / len(test_loader)
    allLab_binary = [1 if y == 1 else 0 for y in allLab]
    allPred_binary = [1 if y == 1 else 0 for y in allPred]

    recall = recall_score(allLab_binary, allPred_binary, average="binary", pos_label=1, zero_division=0)
    f1 = f1_score(allLab_binary, allPred_binary, average="binary", pos_label=1, zero_division=0)
    auc = roc_auc_score(allLab_binary, allProb) if len(np.unique(allLab_binary)) > 1 else 0.0

    return average_loss, accuracy, recall, f1, auc

def compute_confusion_matrix(model, testset):
    true_labels, predicted_labels = [], []
    loader = DataLoader(testset, batch_size=64)
    model.eval()
    with torch.no_grad():
        for features, labels in loader:
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())
    return confusion_matrix(true_labels, predicted_labels)

# Plotting Functions, used to create result files
def plot_confusion_matrix(cm, title, path=None):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    if path is None:
        path = os.path.dirname(CSV_PATH)
    save_path = os.path.join(path, "confusion_centralized.pdf") # file name, saves in the same location as the CSV file
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_all_metrics(metrics_history, title, path=None):
    total_points = len(metrics_history["f1s"])
    
    points_to_plot = 25
    step = max(1, total_points // points_to_plot)

    rounds_subsampled = list(range(1, total_points + 1))[::step]
    
    accuracies_subsampled = metrics_history["accuracies"][::step]
    recalls_subsampled = metrics_history["recalls"][::step]
    f1s_subsampled = metrics_history["f1s"][::step]
    aucs_subsampled = metrics_history["aucs"][::step]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(rounds_subsampled, accuracies_subsampled, marker="o", label="Accuracy", color="tab:red")
    ax.plot(rounds_subsampled, recalls_subsampled, marker="s", label="Recall", color="tab:blue")
    ax.plot(rounds_subsampled, f1s_subsampled, marker="^", label="F1", color="tab:green")
    ax.plot(rounds_subsampled, aucs_subsampled, marker="d", label="AUC", color="tab:purple")
    
    # Formatting the plot
    ax.set_title(title)
    ax.set_xlabel("Rounds")
    ax.set_ylabel("Global Metric")
    ax.grid(True)
    ax.legend(loc="best", title="Metrics")
    
    ax.set_xticks(rounds_subsampled) 
    ax.tick_params(axis='x', rotation=45)
    
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    
    # Saving the file
    plt.tight_layout()
    if path is None:
        path = os.path.dirname(CSV_PATH)
    save_path = os.path.join(path, "metrics_graph_centralized.pdf") # results file name
    plt.savefig(save_path, bbox_inches='tight')
    
    plt.show()
    plt.close()