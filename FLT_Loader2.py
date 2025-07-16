import csv
import random
from collections import Counter, defaultdict

# Dimensions of categorical features and classes
cat_sizes = [12, 8, 6, 5, 7, 11, 9, 4, 3, 3, 4]
n_classes = 3
n_features = sum(cat_sizes)

def one_hot(idx, length):
    vec = [0] * length
    vec[idx] = 1
    return vec


def load_dataset(path): #Inwon hyperparameter optimization, use raytune to run exhaustively to run
    data = []
    with open(path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            try:
                # parse categories from raw CSV
                identifier = int(row[1]) # StateQ

                feature1 = int(row[6])-1   # Gender/Age Bin
                feature2 = int(row[8])-1   # Education
                feature3 = int(row[9])-1   # Marital Status
                feature4 = int(row[10])-1  # Living Arrangements
                feature5 = int(row[12])-1  # Financial Children
                feature6 = int(row[13])-1  # Annual Income
                feature7 = int(row[16])-1  # Employment
                feature8 = int(row[60])-1  # Web/App help for Financial Tasks
                feature9 = int(row[68])-1  # Stock investments
                feature10 = int(row[104])-1# Health Insurance
                feature11 = int(row[111])-1# Financial Education


                if feature1 < 0:
                    raise ValueError(f"Feature 1 is negative!")
                if feature2 < 0:
                    raise ValueError(f"Feature 2 is negative!")
                if feature3 < 0:
                    raise ValueError(f"Feature 3 is negative!")
                if feature4 < 0:
                    raise ValueError(f"Feature 4 is negative!")
                if feature5 < 0:
                    raise ValueError(f"Feature 5 is negative!")
                if feature6 < 0:
                    raise ValueError(f"Feature 6 is negative!")
                if feature7 < 0:
                    raise ValueError(f"Feature 7 is negative!")
                if feature8 < 0:
                    raise ValueError(f"Feature 8 is negative!")
                if feature9 < 0:
                    raise ValueError(f"Feature 9 is negative!")
                if feature10 < 0:
                    raise ValueError(f"Feature 10 is negative!")
                if feature11 < 0:
                    raise ValueError(f"Feature 11 is negative!")

                # clamp if out-of-range
                feature1 = feature1 if 0 <= feature1 < cat_sizes[0] else 0
                feature2 = feature2 if 0 <= feature2 < cat_sizes[1] else 0
                feature3 = feature3 if 0 <= feature3 < cat_sizes[2] else 0
                feature4 = feature4 if 0 <= feature4 < cat_sizes[3] else 0
                feature5 = feature5 if 0 <= feature5 < cat_sizes[4] else 0
                feature6 = feature6 if 0 <= feature6 < cat_sizes[5] else 0
                feature7 = feature7 if 0 <= feature7 < cat_sizes[6] else 0
                feature8 = feature8 if 0 <= feature8 < cat_sizes[7] else 0
                feature9 = feature9 if 0 <= feature9 < cat_sizes[8] else 0
                feature10 = feature10 if 0 <= feature10 < cat_sizes[9] else 0
                feature11 = feature11 if 0 <= feature11 < cat_sizes[10] else 0

                # one-hot encode features
                vec = []
                vec += one_hot(feature1, cat_sizes[0])
                vec += one_hot(feature2, cat_sizes[1])
                vec += one_hot(feature3, cat_sizes[2])
                vec += one_hot(feature4, cat_sizes[3])
                vec += one_hot(feature5, cat_sizes[4])
                vec += one_hot(feature6, cat_sizes[5])
                vec += one_hot(feature7, cat_sizes[6])
                vec += one_hot(feature8, cat_sizes[7])
                vec += one_hot(feature9, cat_sizes[8])
                vec += one_hot(feature10, cat_sizes[9])
                vec += one_hot(feature11, cat_sizes[10])

                # parse label
                learningObj = int(float(row[102])) #Debt Collection
                #print(learningObj)
                #learningObj = learningObj if 0 <= learningObj < n_classes else 0

                if learningObj > n_classes: #skip certain cases
                    continue

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