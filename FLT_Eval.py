import random
from collections import Counter, defaultdict
from FLT_Loader import load_dataset, split_data_across_clients, n_classes, n_features
from FLT_Model   import federated_avg, predict_proba


def main():
    path = '/data/LorenzoData/NFCS2021StateData220627.csv'
    random.seed(42)

    # Load and shuffle
    raw_data = load_dataset(path)
    random.shuffle(raw_data)
    split1 = int(0.8 * len(raw_data))
    train_all = raw_data[:split1]
    test_data = raw_data[split1:]
    print(f"Total samples: {len(raw_data)}  → Train+Val: {len(train_all)}, Test: {len(test_data)}")

    # Train/Val split
    split2 = int(0.75 * len(train_all))
    train_data = train_all[:split2]
    val_data = train_all[split2:]
    print(f"→ After split:  Train: {len(train_data)},  Val: {len(val_data)},  Test: {len(test_data)}")

    # Class weights
    labels = [y.index(1) for _, y in train_data]
    counts = Counter(labels)
    total_train = len(labels)
    class_weights = {c: total_train / (n_classes * counts.get(c, 1)) for c in range(n_classes)}

    # Oversample training data
    by_class = defaultdict(list)
    for x, y in train_data:
        by_class[y.index(1)].append((x, y))
    max_size = max(len(v) for v in by_class.values())
    balanced = []
    for cls, examples in by_class.items():
        extra = random.choices(examples, k=max_size - len(examples))
        balanced.extend(examples + extra)
    random.shuffle(balanced)
    print("Balanced train size per class:", {c: len(by_class[c]) for c in by_class})

    # Grid‐search hyper‐parameters
    best = {"val_acc": -1.0}
    for K in [100, 110]:
        for E in [1, 3, 4]:
            for C in [0.05, 0.06]:
                for lr in [0.0125, 0.013, 0.015]:
                    clients = split_data_across_clients(balanced, K)
                    W = [[0.0] * n_features for _ in range(n_classes)]
                    b = [0.0] * n_classes
                    Wf, bf = federated_avg(W, b, clients, E=E, lr=lr, rounds=250, C=C, class_weights=class_weights)
                    correct = sum(
                        1 for x, y in val_data
                        if max(range(n_classes), key=lambda c: predict_proba(x, Wf, bf)[c]) == y.index(1)
                    )
                    val_acc = correct / len(val_data)
                    print(f"[VAL] K={K} E={E} C={C:.4f} lr={lr:.5f} → val_acc={val_acc:.5f}")
                    if val_acc > best["val_acc"]:
                        best = {"K": K, "E": E, "C": C, "lr": lr, "val_acc": val_acc}

    print("→ Best on validation:", best)

    # Final training and evaluation on test set
    combined = train_all
    clients = split_data_across_clients(combined, best["K"])
    W = [[0.0] * n_features for _ in range(n_classes)]
    b = [0.0] * n_classes
    Wf, bf = federated_avg(W, b, clients, E=best["E"], lr=best["lr"], rounds=2000, C=best["C"], class_weights=class_weights)

    correct = sum(
        1 for x, y in test_data
        if max(range(n_classes), key=lambda c: predict_proba(x, Wf, bf)[c]) == y.index(1)
    )
    test_acc = correct / len(test_data)
    print(f"*** FINAL TEST ACCURACY: {test_acc:.6f} ***")

if __name__ == "__main__":
    main()
