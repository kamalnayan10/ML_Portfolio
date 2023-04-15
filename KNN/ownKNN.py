import numpy as np
from math import sqrt
import warnings
from collections import Counter
import pandas as pd
import random


# def k_nearest_neighbors(data , predict , k = 3):
#     """
#     In this model I calculate euclidean distance of every node and store it in the distances list and sort the distances
#     list in ascending oreder and take the first k elements to calculate vote 
#     adding O(N) extra space for storing distances and O(nlogn) extra time for sorting distances
#     """

#     if len(data) >= k:
#         warnings.warn("K is set to a value less than total voting groups!!")

#     distances = []

#     for group in data:
#         for features in data[group]:
#             euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
#             distances.append([euclidean_distance , group])

#     votes = [i[1] for i in sorted(distances)[:k]]
#     vote_result = Counter(votes).most_common(1)[0][0]
#     confidence = (Counter(votes).most_common(1)[0][1]/k)*100

#     return vote_result,confidence

def k_nearest_neighbors(data , predict , k = 3):
    """
    In this model I calculate euclidean distance of every node but store the distances of only the closest k nodes
    and sort those k nodes having an extra space of O(1) and sort only those k distances having 
    O(N) extra time {O(1) for individual sorting done N times so total complexity is O(N)}
    """

    if len(data) >= k:
        warnings.warn("K is set to a value less than total voting groups!!")

    distances = [[float("inf") , 0]]*k

    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            for i in range(k):
                if euclidean_distance < distances[i][0]:
                    distances.append([euclidean_distance , group])
                    del distances[i]
                    break
                
    votes = [i[1] for i in sorted(distances)]
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = (Counter(votes).most_common(1)[0][1]/k)*100

    return vote_result,confidence

accuracies = []
for i in range(25):
    df = pd.read_csv("KNN/breast-cancer-wisconsin.data.txt")
    df.replace("?" , -99999 , inplace = True)
    df.drop(["id"],1 , inplace = True) 

    # SPLITTING TRAINING AND TESTING DATA

    full_data = df.astype(float).values.tolist()

    random.shuffle(full_data)

    test_size = 0.2

    train_set = {2:[] , 4:[]}
    test_set = {2:[] , 4:[]}

    train_data = full_data[:-int(test_size*len(full_data))]
    test_data = full_data[-int(test_size*len(full_data)):]

    for i in train_data:
        train_set[i[-1]].append(i[:-1])

    for i in test_data:
        test_set[i[-1]].append(i[:-1])

    correct = 0
    total = 0

    # BUILDING MODEL AND CALCULATING ACCURACY

    for group in test_set:
        for data in test_set[group]:
            vote ,confidence= k_nearest_neighbors(train_set , data , k = 5)
            if group == vote:
                correct += 1
            total += 1

    accuracy = (correct/total)*100
    accuracies.append(accuracy)

print(sum(accuracies)/len(accuracies))