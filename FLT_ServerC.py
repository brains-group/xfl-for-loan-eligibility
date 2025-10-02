import random
import os
import torch
from torch.utils.data import TensorDataset

from FLT_LoaderC import *
from utilsC import *

def main():
    

    #random.seed(42)
    #torch.manual_seed(42) # Optional seeds

    # Load and prepare the dataset
    raw_data = load_dataset(CSV_PATH)
    random.shuffle(raw_data)

    # Split data into training and testing sets
    split_idx = int(0.8 * len(raw_data))
    train_data = raw_data[:split_idx]
    test_data = raw_data[split_idx:]
    print(f"Fixed - Total samples: {len(raw_data)} -> Train: {len(train_data)}, Test: {len(test_data)}")

    # Create unified TensorDatasets for centralized training
    train_features = torch.tensor([x for x, y in train_data], dtype=torch.float32)
    train_labels = torch.tensor([y for x, y in train_data], dtype=torch.long)
    train_dataset = TensorDataset(train_features, train_labels)

    test_features = torch.tensor([x for x, y in test_data], dtype=torch.float32)
    test_labels = torch.tensor([y for x, y in test_data], dtype=torch.long)
    test_dataset = TensorDataset(test_features, test_labels)
    
    # Initialize the model
    net = HighwayModel2(n_features=n_features, n_classes=n_classes)

    metrics_history = {"accuracies": [], "recalls": [], "f1s": [], "aucs": []}

    print(f"\nStarting centralized training for {r} rounds...")
    for roundd in range(1, r + 1):
        print(f"--- Round {roundd}/{r} ---")
        
        # Train the model for one round
        train_model(net, train_dataset, epochs_to_run=5)
        
        # Evaluate the model on the entire test set
        loss, acc, recall, f1, auc = evaluate_model(net, test_dataset)
        print(f"  Evaluation - Loss: {loss:.4f}, Acc: {acc:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        
        # Store metrics
        metrics_history["accuracies"].append(acc)
        metrics_history["recalls"].append(recall)
        metrics_history["f1s"].append(f1)
        metrics_history["aucs"].append(auc)

    print("\nTraining complete.")
    
    # Final Evaluation and Plotting
    print("Generating final reports and plots...")

    # Plot metrics over all epochs
    plot_all_metrics(
        metrics_history,
        title="Centralized Model Metrics Per Round"
    )

    # Plot confusion matrix
    cm = compute_confusion_matrix(net, test_dataset)
    plot_confusion_matrix(
        cm,
        title="Final Centralized Model Confusion Matrix"
    )

    print("All tasks finished. Check for output PDF files.")


if __name__ == "__main__":
    # Ensure the data path exists
    if not os.path.exists(CSV_PATH):
        print(f"Error: The data file was not found at {CSV_PATH}")
        print("Please update the CSV_PATH variable in utils.py")
    else:
        main()