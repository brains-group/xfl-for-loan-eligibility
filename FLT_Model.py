import math
import random
from FLT_Loader import load_dataset, split_data_across_clients, n_classes, n_features


def softmax(z):
    ex = [math.exp(v) for v in z]
    s = sum(ex)
    return [v / s for v in ex]


def predict_proba(x, W, b): #calculate probabilities via softmax
    z = [
        sum(W[c][i] * x[i] for i in range(n_features)) + b[c]
        for c in range(n_classes)
    ]
    return softmax(z)


def client_update(W, b, dataset, epochs=1, lr=0.01, class_weights=None): #perform an update
    localW = [row[:] for row in W]
    localb = b[:]
    for _ in range(epochs):
        for x, y_true in dataset:
            p = predict_proba(x, localW, localb)
            for c in range(n_classes):
                weight = class_weights.get(c, 1.0) if class_weights else 1.0
                diff = (p[c] - y_true[c]) * weight
                for i in range(n_features):
                    localW[c][i] -= lr * diff * x[i]
                localb[c] -= lr * diff
    return localW, localb, len(dataset)


def federated_avg(W, b, clients, E, lr, rounds, C, class_weights): #W-Weights, b-Bias, E-Epochs, lr-Learning Rate, C-Sampling Rate
    globalW, globalb = W, b
    K = len(clients)
    for _ in range(1, rounds + 1):
        chosen = random.sample(clients, max(1, int(C * K)))
        updates = [
            client_update(globalW, globalb, c, epochs=E, lr=lr, class_weights=class_weights)
            for c in chosen
        ]
        total = sum(sz for _, _, sz in updates)
        # aggregate
        newW = [[0.0] * n_features for _ in range(n_classes)]
        newb = [0.0] * n_classes
        for Wc, bc, sz in updates:
            for c in range(n_classes):
                newb[c] += bc[c] * sz
                for i in range(n_features):
                    newW[c][i] += Wc[c][i] * sz
        for c in range(n_classes):
            newb[c] /= total
            for i in range(n_features):
                newW[c][i] /= total
        globalW, globalb = newW, newb
    return globalW, globalb


def predict(x, weights):
    return sum(weights[i] * x[i] for i in range(len(weights) - 1)) + weights[-1]


def average_weights(weight_list): #take the average of the weights
    if not weight_list:
        return []
    num_weights = len(weight_list[0][0])
    total_size = sum(size for _, size in weight_list)
    averaged = [0.0] * num_weights
    for weights, size in weight_list:
        for i in range(num_weights):
            averaged[i] += weights[i] * size
    return [w / total_size for w in averaged]