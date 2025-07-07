import csv
import random
from collections import Counter, defaultdict

# Dimensions of categorical features and classes
cat_sizes = [3, 13, 6, 7, 9]
n_classes = 4
n_features = sum(cat_sizes)

def one_hot(idx, length):
    vec = [0] * length
    vec[idx] = 1
    return vec


def load_dataset(path):
    data = []
    with open(path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            try:
                # parse categories from raw CSV
                identifier = int(row[1]) # StateQ
                feature1 = int(row[4])   # Gender
                feature2 = int(row[6])   # Age Bin
                feature3 = int(row[9])   # Marital Status
                feature4 = int(row[12])  # Financially Dependent Children
                feature5 = int(row[16])  # Employment status

                # clamp if out-of-range
                feature1 = feature1 if 0 <= feature1 < cat_sizes[0] else 0
                feature2 = feature2 if 0 <= feature2 < cat_sizes[1] else 0
                feature3 = feature3 if 0 <= feature3 < cat_sizes[2] else 0
                feature4 = feature4 if 0 <= feature4 < cat_sizes[3] else 0
                feature5 = feature5 if 0 <= feature5 < cat_sizes[4] else 0

                # one-hot encode features
                vec = []
                vec += one_hot(feature1, cat_sizes[0])
                vec += one_hot(feature2, cat_sizes[1])
                vec += one_hot(feature3, cat_sizes[2])
                vec += one_hot(feature4, cat_sizes[3])
                vec += one_hot(feature5, cat_sizes[4])

                # parse label
                learningObj = int(float(row[84]))
                learningObj = learningObj if 0 <= learningObj < n_classes else 0

                data.append((identifier, (vec, learningObj)))
            except ValueError:
                continue
    return data


def split_data_across_clients(data): #Code to get all datapoints and split them across 51 clients by state
    # 1) Pre-allocate one list per client
    clients = [[] for _ in range(51)]

    # 2) Distribute each sample into the right bucket
    for identifier, sample in data:
        #print(identifier)
        if not (1 <= identifier <= 51):
            raise ValueError(f"Identifier {identifier} out of range")
        clients[identifier - 1].append(sample)

    # 3) (Optional) shuffle within each client
    for c in clients:
        random.shuffle(c)
    
    return clients