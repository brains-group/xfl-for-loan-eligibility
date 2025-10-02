import random
import os
import torch
from torch.utils.data import TensorDataset
import numpy as np

# Import from local modules
from FLT_LoaderC2 import *
from utilsC2 import *

def main():
    random.seed(42)
    torch.manual_seed(42)

    # Load and split the dataset by state
    print("Loading and splitting dataset by state...")
    raw_data = load_dataset(CSV_PATH)
    random.shuffle(raw_data)

    split_idx = int(0.8 * len(raw_data))
    train_data_raw = raw_data[:split_idx]
    test_data_raw = raw_data[split_idx:]

    # Split to training and testing sets
    train_sets = split_data_across_clients(train_data_raw)
    test_sets = split_data_across_clients(test_data_raw)
    print(f"Dataset split into {len(train_sets)} partitions.")

    # Initialize 51 separate models
    num_states = len(train_sets)
    models = [HighwayModel2(n_features=n_features, n_classes=n_classes) for _ in range(num_states)]
    print(f"Initialized {len(models)} separate models.")

    # Training Loop: This dictionary will store the average metrics across all models per epoch
    avg_metrics_history = {"accuracies": [], "recalls": [], "f1s": [], "aucs": []}

    print(f"\nStarting per-state training for {r} epochs...")
    for epoch in range(1, r + 1):
        print(f"--- Epoch {epoch}/{r} ---")
        
        # Store metrics from all models for the current epoch
        epoch_metrics = {"accuracies": [], "recalls": [], "f1s": [], "aucs": []}

        # Inner loop: Train and evaluate each state's model
        for i in range(num_states):
            state_train_data = train_sets[i]
            state_test_data = test_sets[i]

            # Skip states with no training data, failsafe
            if not state_train_data:
                continue

            # Convert this state's data to tensors
            train_features = torch.tensor([x for x, y in state_train_data], dtype=torch.float32)
            train_labels = torch.tensor([y for x, y in state_train_data], dtype=torch.long)
            state_train_dataset = TensorDataset(train_features, train_labels)

            test_features = torch.tensor([x for x, y in state_test_data], dtype=torch.float32)
            test_labels = torch.tensor([y for x, y in state_test_data], dtype=torch.long)
            state_test_dataset = TensorDataset(test_features, test_labels)

            # Train this state's model once
            train_model(models[i], state_train_dataset, epochs_to_run=1)
            
            # Evaluate and store its metrics
            loss, acc, recall, f1, auc = evaluate_model(models[i], state_test_dataset)
            epoch_metrics["accuracies"].append(acc)
            epoch_metrics["recalls"].append(recall)
            epoch_metrics["f1s"].append(f1)
            epoch_metrics["aucs"].append(auc)

        # After training all models for one epoch, calculate and store the average metrics
        avg_acc = np.mean(epoch_metrics["accuracies"])
        avg_recall = np.mean(epoch_metrics["recalls"])
        avg_f1 = np.mean(epoch_metrics["f1s"])
        avg_auc = np.mean(epoch_metrics["aucs"])

        avg_metrics_history["accuracies"].append(avg_acc)
        avg_metrics_history["recalls"].append(avg_recall)
        avg_metrics_history["f1s"].append(avg_f1)
        avg_metrics_history["aucs"].append(avg_auc)
        
        print(f"  Average Across All States - Acc: {avg_acc:.4f}, Recall: {avg_recall:.4f}, F1: {avg_f1:.4f}, AUC: {avg_auc:.4f}")

    print("\nTraining complete.")
    
    # Final Plotting of Averaged Metrics
    print("Generating final plot of average metrics...")
    plot_all_metrics(
        avg_metrics_history,
        title="Average Model Metrics Across All States"
    )
    print("All tasks finished.")

if __name__ == "__main__":
    main()