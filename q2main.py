# Elif Gamze Güliter
# 21802870
# knn classifier codes

import numpy as numpy
import math
from numpy import genfromtxt
import clock
import time
import csv

data = list()
count = 0
with open('diabetes_train_features.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        count = count + 1
        if count != 1:
            data.append([row[0],row[1], row[2], row[3], row[8],row[7]])
count = 0


def calculate_distance(test_data, train_data, index1):
    d = 0.0
    count_i = 1

    while count_i < len(train_data[1]):
        #print(count_i)
        d = d + (pow((float(test_data[count_i]) - float(train_data[index1][count_i])), 2))
        count_i = count_i + 1
    d = math.sqrt(d)
    return d


def find_Neigbours(test_data, train_data):
    count_index = 0
    neighbours = []
    while count_index < len(data):
        d = 0
        d = calculate_distance(test_data, train_data, count_index)
        node = [d, count_index]
        neighbours.append(node)

        count_index = count_index + 1
    # print(neighbours)
    return neighbours


def kNN_Classifier(test_data, train_data, k):
    neigbours = find_Neigbours(test_data, train_data)
    neigbours.sort()
    k_count = 0
    k_neigbours = []

    while k_count < k:
        k_neigbours.append(neigbours[k_count])
        k_count = k_count + 1

    count = 0
    indexes = []
    while count < k:
        indexes.append(k_neigbours[count][1])
        count = count + 1

    labels = []
    train_labels = []
    count = 0
    with open('diabetes_train_labels.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            count = count + 1
            if count != 1:
                train_labels.append(row)
    count = 0
    while count < k:
        labels.append(train_labels[indexes[count]][1])
        count = count + 1
    most_common = max(labels, key=labels.count)

    return most_common


if __name__ == '__main__':
    test_data = []
    count = 0
    test_labels = []
    TP = 0
    FN = 0
    TN = 0
    FP = 0
    accuracy = 0
    t1 = time.time()
    with open('diabetes_test_labels.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            count = count + 1
            if count != 1:
                test_labels.append(row)

    with open('diabetes_test_features.csv', 'r') as file:
        reader = csv.reader(file)
        x = 0
        matrix = []
        predicted = []
        for row in reader:
            x = x + 1
            if x != 1:
                if test_labels[x - 2][1] == kNN_Classifier([row[0],row[1], row[2], row[3], row[8], row[7]], data, 9):
                    accuracy = accuracy + 1
                    if int(kNN_Classifier(row, data, 9)) == 0:
                        TN = TN + 1
                    else:
                        TP = TP + 1
                else:
                    if int(kNN_Classifier(row, data, 9)) == 1:
                        FP = FP + 1
                    else:
                        FN = FN + 1
    print("Accuracy", accuracy * 100 / len(test_labels) - 1)
    print("TN = ", TN, "FP = ", FP, "TP = ", TP, "FN = ", FN, )

    t2 = time.time()
    print("Running time in seconds =  ", t2 - t1)
