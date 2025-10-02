import csv
import random
from collections import Counter, defaultdict

# Dimensions of categorical features and classes
cat_sizes = [12, 12, 8, 6, 5, 7, 11, 9, 4, 3, 3, 4]
n_classes = 3 #total number of learnin objective classes, unused as eval() expects and filters itself
n_features = sum(cat_sizes)

def one_hot(idx, length): #one-hot function
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

                feature1 = int(row[6])-1   # Gender/Age Bin ordered by gender
                feature1b = int(row[6])-1   #Gender/Age Bin ordered by age
                if feature1==1:#Categorically sort 1b
                    feature1b = 2
                if feature1==2:
                    feature1b = 4
                if feature1==3:
                    feature1b = 6
                if feature1==4:
                    feature1b = 8
                if feature1==5:
                    feature1b = 10
                if feature1==6:
                    feature1b = 1
                if feature1==7:
                    feature1b = 3
                if feature1==8:
                    feature1b = 5
                if feature1==9:
                    feature1b = 7
                if feature1==10:
                    feature1b = 9

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


                if feature1 < 0: #sanity check for negative values
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
                feature1b = feature1b if 0 <= feature1b < cat_sizes[1] else 0
                feature2 = feature2 if 0 <= feature2 < cat_sizes[2] else 0
                feature3 = feature3 if 0 <= feature3 < cat_sizes[3] else 0
                feature4 = feature4 if 0 <= feature4 < cat_sizes[4] else 0
                feature5 = feature5 if 0 <= feature5 < cat_sizes[5] else 0
                feature6 = feature6 if 0 <= feature6 < cat_sizes[6] else 0
                feature7 = feature7 if 0 <= feature7 < cat_sizes[7] else 0
                feature8 = feature8 if 0 <= feature8 < cat_sizes[8] else 0
                feature9 = feature9 if 0 <= feature9 < cat_sizes[9] else 0
                feature10 = feature10 if 0 <= feature10 < cat_sizes[10] else 0
                feature11 = feature11 if 0 <= feature11 < cat_sizes[11] else 0

                # one-hot encode features
                vec = []
                vec += one_hot(feature1, cat_sizes[0])
                vec += one_hot(feature1b, cat_sizes[1])
                vec += one_hot(feature2, cat_sizes[2])
                vec += one_hot(feature3, cat_sizes[3])
                vec += one_hot(feature4, cat_sizes[4])
                vec += one_hot(feature5, cat_sizes[5])
                vec += one_hot(feature6, cat_sizes[6])
                vec += one_hot(feature7, cat_sizes[7])
                vec += one_hot(feature8, cat_sizes[8])
                vec += one_hot(feature9, cat_sizes[9])
                vec += one_hot(feature10, cat_sizes[10])
                vec += one_hot(feature11, cat_sizes[11])

                # parse label
                learningObj = int(float(row[102])) #Debt Collection

                if learningObj > n_classes: #filtering
                    continue

                data.append((vec, learningObj))#Add data point
            except ValueError:
                continue
    return data
