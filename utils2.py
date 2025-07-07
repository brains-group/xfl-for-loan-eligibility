"""
Utility functions and classes for Jupyter Notebooks lessons. 
"""

from collections import OrderedDict
from typing import List, Tuple, Dict, Optional
from flwr.common import Metrics, NDArrays, Scalar
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader, random_split, WeightedRandomSampler
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import logging
from flwr.common.logger import console_handler, log
from logging import INFO, ERROR
import random
from FLT_Loader2 import n_features, n_classes


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
backend_setup = {"init_args": {"logging_level": ERROR, "log_to_driver": False}}


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
    class_weights = 1.0 / class_counts.float()          # length = n_classes

    # 3a) Weighted loss
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 3b) (optional) Oversample minority classes via sampler
    sample_weights = class_weights[labels_tensor]       # shape [N]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    # Use the sampler instead of shuffle for balanced batches
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=sampler,
    )


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()


def evaluate_model(model, test_set):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0

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

    accuracy = correct / total
    average_loss = total_loss / len(test_loader)
    # print(f"Test Accuracy: {accuracy:.4f}, Average Loss: {average_loss:.4f}")
    return average_loss, accuracy


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


def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", linewidths=0.5)
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
