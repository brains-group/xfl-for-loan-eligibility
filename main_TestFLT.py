import random
import csv
import math
from collections import Counter
from collections import defaultdict

#global
cat_sizes = [3, 102, 6, 7, 9]
n_classes = 4
n_features = sum(cat_sizes)

def softmax(z):
    ex = [math.exp(v) for v in z]
    s  = sum(ex)
    return [v/s for v in ex]

def predict_proba(x, W, b):
    #z_c = W[c] * x + b[c]
    z = [ sum(W[c][i]*x[i] for i in range(n_features)) + b[c]
          for c in range(n_classes) ]
    return softmax(z)

def one_hot(idx, length):
    vec = [0]*length
    vec[idx] = 1
    return vec

def load_dataset(path):
    data = []

    with open(path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            try:
                # parse each as int category in its known range
                f1 = int(row[4])   # Gender, should be in 0..2
                f2 = int(row[5])   # Age, 0..101
                f3 = int(row[9])   # Marital Status, 0..5
                f4 = int(row[12])  # Financially Dependent Children, 0..6
                f5 = int(row[16])  # employment or work status, 0..8

                # clamp if out-of-range
                f1 = f1 if 0 <= f1 < 3   else 0
                f2 = f2 if 0 <= f2 < 102 else 0
                f3 = f3 if 0 <= f3 < 6   else 0
                f4 = f4 if 0 <= f4 < 7   else 0
                f5 = f5 if 0 <= f5 < 9   else 0

                # one-hot each
                vec = []
                vec += one_hot(f1, cat_sizes[0])
                vec += one_hot(f2, cat_sizes[1])
                vec += one_hot(f3, cat_sizes[2])
                vec += one_hot(f4, cat_sizes[3])
                vec += one_hot(f5, cat_sizes[4])
                
                # load label 0..3, clamp then one-hot
                y = int(float(row[84])) # Learning Objective, do you have an auto loan?
                y = y if 0 <= y < n_classes else 0
                label = one_hot(y, n_classes)

                data.append((vec, label))
            except ValueError:
                continue
    return data


def split_data_across_clients(data, K):
    #K clients
    random.shuffle(data)
    chunk_size = len(data) // K
    return [data[i*chunk_size:(i+1)*chunk_size] for i in range(K)]

def federated_avg(W, b, clients, E, lr, rounds, C, class_weights):
    globalW, globalb = W, b
    K = len(clients)
    for _ in range(1, rounds+1):
        #print(f"\n--- Round {_}/{rounds} ---")
        chosen = random.sample(clients, max(1, int(C*K)))
        #print(f" Sampling {len(chosen)} clients out of {K}")
        updates = [client_update(globalW, globalb, c, E, lr, class_weights) for c in chosen]
        total = sum(sz for _,_,sz in updates)
        # average W
        newW = [[0.0]*n_features for _ in range(n_classes)]
        newb = [0.0]*n_classes
        for Wc, bc, sz in updates:
            for c in range(n_classes):
                newb[c] += bc[c] * sz
                for i in range(n_features):
                    newW[c][i] += Wc[c][i] * sz
            #print("Averaged Weights: ",newW)
        for c in range(n_classes):
            newb[c] /= total
            for i in range(n_features):
                newW[c][i] /= total
        globalW, globalb = newW, newb
        #print(" Updated global weights norm:",
        #      sum(abs(w) for row in globalW for w in row)**0.5,
        #      " bias norm:", sum(abs(v) for v in globalb)**0.5)
    return globalW, globalb




def client_update(W, b, dataset, epochs=1, lr=0.01, class_weights=None):
    localW = [row[:] for row in W]
    localb = b[:]
    for _ in range(epochs):
        for x, y_true in dataset:
            p = predict_proba(x, localW, localb)
            # gradient and update for each class
            for c in range(n_classes):
                #scale error by inverse-frequency weight
                weight = class_weights.get(c, 1.0)
                diff = (p[c] - y_true[c]) * weight
                for i in range(n_features):
                    localW[c][i] -= lr * diff * x[i]
                localb[c]   -= lr * diff
    return localW, localb, len(dataset)



def average_weights(weight_list):#TODO
    if not weight_list:
        print("Not a Weight\n")
    num_weights = len(weight_list[0][0])    
    total_size = sum(size for _, size in weight_list)

    averaged = [0.0]*num_weights
    for weights, size in weight_list:
        for i in range(num_weights):
            averaged[i] += (weights[i]*size)

    #normalize
    for i in range(num_weights):
        averaged[i] /= total_size

    return averaged


def predict(x, weights):
    """
    predict output for input vector using computed weights
    x = list of 5 feature values
    weights = list of 6 values (5 weights + bias)
    """
    return sum(weights[i] * x[i] for i in range(5)) + weights[5]



def main():
    path = 'NFCS 2021 State Data 220627.csv'
    random.seed(42)

    # 1) Load & one-hot encode
    raw_data = load_dataset(path)

    # 2) Shuffle once
    random.shuffle(raw_data)

    # 3) Split off TEST (20%)
    split1 = int(0.8 * len(raw_data))
    train_all = raw_data[:split1]
    test_data = raw_data[split1:]
    print(f"Total samples: {len(raw_data)}  → Train+Val: {len(train_all)}, Test: {len(test_data)}")

    # 4) Split train_all into TRAIN (60%) and VAL (20%)
    split2 = int(0.75 * len(train_all))   # 0.75 * 80% = 60% of total
    train_data = train_all[:split2]
    val_data   = train_all[split2:]
    print(f"→ After split:  Train: {len(train_data)},  Val: {len(val_data)},  Test: {len(test_data)}")

    # 5) (Optional) Compute class_weights on train_data
    labels      = [y.index(1) for _, y in train_data]
    counts      = Counter(labels)
    total_train = len(labels)
    class_weights = {
        c: total_train / (n_classes * counts.get(c, 1))
        for c in range(n_classes)
    }

    # 6) Oversample ONLY train_data for balancing
    by_class = defaultdict(list)
    for x,y in train_data:
        by_class[y.index(1)].append((x,y))
    max_size = max(len(v) for v in by_class.values())

    balanced = []
    for cls, examples in by_class.items():
        extra = []
        if len(examples) < max_size:
            extra = random.choices(examples, k=max_size-len(examples))
        balanced.extend(examples + extra)
    random.shuffle(balanced)
    print("Balanced train size per class:", {c: len(by_class[c]) for c in by_class})

    # 7) Grid‐search hyper-parameters using VAL
    best = {"val_acc": -1.0}
    for K in [100, 110]: #clients
      for E in [1, 3, 4]: #epochs
        for C in [0.05, 0.06]: #sample size
          for lr in [0.0125, 0.013, 0.015]: #learning rate
            # split balanced TRAIN into K clients
            clients = split_data_across_clients(balanced, K)

            # init
            W = [[0.0]*n_features for _ in range(n_classes)]
            b = [0.0]*n_classes

            # federated train on TRAIN
            Wf, bf = federated_avg(
                W, b, clients,
                E=E, lr=lr, rounds=250, C=C,
                class_weights=class_weights
            )

            # evaluate on VAL
            correct = 0
            for x, y in val_data:
                pred = max(range(n_classes),
                           key=lambda c: predict_proba(x, Wf, bf)[c])
                if pred == y.index(1):
                    correct += 1
            val_acc = correct / len(val_data)

            print(f"[VAL] K={K} E={E} C={C:.4f} lr={lr:.5f} → val_acc={val_acc:.5f}")

            if val_acc > best["val_acc"]:
                best = {"K":K,"E":E,"C":C,"lr":lr,"val_acc":val_acc,"b":bf}

    print("→ Best on validation:", best)

    # 8) Final training on TRAIN+VAL with best hyper‐params
    combined = train_all  # 60%+20%=80%
    # (you could rebalance combined again if you like)
    clients = split_data_across_clients(combined, best["K"])
    W   = [[0.0]*n_features for _ in range(n_classes)]
    b   = [0.0]*n_classes
    Wf, bf = federated_avg(
        W, b, clients,
        E=best["E"], lr=best["lr"],
        rounds=2000, C=best["C"],
        class_weights=class_weights
    )

    # 9) Evaluate on TEST
    correct = 0
    for x, y in test_data:
        pred = max(range(n_classes),
                   key=lambda c: predict_proba(x, Wf, bf)[c])
        if pred == y.index(1):
            correct += 1
        #print("Actual: ",y.index(1),"    Predicted: ",pred)
    test_acc = correct / len(test_data)
    print(f"*** FINAL TEST ACCURACY: {test_acc:.6f} ***")


if __name__ == "__main__":
    main()