# elif gamze güliter
# bernouli and multinomial naive bayes
import numpy as numpy
import math
import time
from numpy import genfromtxt
import csv

sms_data = list()
count = 0
number_of_sms = 0
occurrences_ham = 0
occurrences_spam = 0

# read train data
with open('sms_train_features.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        count = count + 1
        if count != 1:
            sms_data.append(row)
count = 0

number_of_sms = len(sms_data)

# read train labels
sms_label = []
count = 0
sms_spam = []
sms_ham = []
with open('sms_train_labels.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        count = count + 1
        if count != 1:
            sms_label.append(row)
count = 0
# divide train data to spam and ham
while count < len(sms_label):
    if float(sms_label[count][1]) == 1:
        sms_spam.append(sms_data[count])

    if float(sms_label[count][1]) == 0:
        sms_ham.append(sms_data[count])
    count = count + 1

# read test data
sms_test_data = []
count = 0
with open('sms_test_features.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        count = count + 1
        if count != 1:
            sms_test_data.append(row)
count = 0

# read vocabulary
vocab = []
with open('vocabulary.txt', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        vocab.append(row)
count = 0
number_of_word = len(vocab)

# count the number of spam and ham messages
while count < number_of_sms:
    if float(sms_label[count][1]) == 0:
        occurrences_ham = occurrences_ham + 1
    else:
        occurrences_spam = occurrences_spam + 1
    count = count + 1

count_column = 0
vocab_counts_spam = []
vocab_counts_ham = []

ones_Spam = 0
prob_spam = 1
countWords = 0
count_2 = 0
count_3 = 0


# counting number of occurrences of a word in spam and ham
def find_Count(index):
    count_1 = 0
    ones_Spam = 0
    ones_Ham = 0
    while count_1 < len(sms_spam):
        if float(sms_spam[count_1][index]) != 0:
            ones_Spam = ones_Spam + float(sms_spam[count_1][index])
        count_1 = count_1 + 1

    count_1 = 0
    while count_1 < len(sms_ham):
        if float(sms_ham[count_1][index]) != 0:
            ones_Ham = ones_Ham + float(sms_ham[count_1][index])
        count_1 = count_1 + 1
    return [ones_Spam, ones_Ham]


# counting occurrences of a word in spam and ham
def find_Count_Bernoulli(index):
    count_1 = 0
    ones_Spam = 0
    ones_Ham = 0
    while count_1 < len(sms_spam):
        if float(sms_spam[count_1][index]) > 0:
            ones_Spam = ones_Spam + 1
        count_1 = count_1 + 1
    count_1 = 0
    while count_1 < len(sms_ham):
        if float(sms_ham[count_1][index]) > 0:
            ones_Ham = ones_Ham + 1
        count_1 = count_1 + 1
    return [ones_Spam, ones_Ham]


# multinomial classifier
def predict_Multinomial(test_data):
    count_1 = 0
    false_Count = 0
    count = 0
    # read vocabulary
    labels = []
    with open('sms_test_labels.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            labels.append(row)
    count = 0

    while count_1 < len(test_data):
        count = 0
        prob_spam = 1
        prob_ham = 1

        while count < len(test_data[1]):
            if float(test_data[count_1][count]) > 0:
                temp = find_Count(count)

                prob_spam = prob_spam * ((int(temp[0]) + 1) / (number_of_word + len(sms_spam)))
                prob_ham = prob_ham * ((int(temp[1]) + 1) / (number_of_word + len(sms_ham)))

            count = count + 1

        prior_spam = (float(len(sms_spam) / len(sms_data)))
        prior_ham = (float(len(sms_ham) / len(sms_data)))
        prob_ham = (prior_ham * prob_ham)
        prob_spam = (prior_spam * prob_spam)

        if prob_ham > prob_spam:
            if (labels[count_1][1]) == "1":
                false_Count = false_Count + 1
            # print(count_1,0)
        else:

            if (labels[count_1][1]) == "0":
                false_Count = false_Count + 1
            # print(count_1,1)

        count_1 = count_1 + 1
    print(((len(sms_test_data) - false_Count) * 100) / len(labels))


# Bernoulli classifier
def predict_Bernoulli(test_data):
    count_1 = 0
    false_Count = 0
    labels = []

    with open('sms_test_labels.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            labels.append(row)

    while count_1 < len(test_data):
        count = 0
        prob_spam = 1
        prob_ham = 1

        while count < len(test_data[1]):
            if float(test_data[count_1][count]) > 0:
                temp = find_Count_Bernoulli(count)

                prob_spam = prob_spam * (int(temp[0]) + 1 / 2 + len(sms_spam))
                prob_ham = prob_ham * (int(temp[1]) + 1 / 2 + len(sms_ham))

            count = count + 1

        prior_spam = float(len(sms_spam) / len(sms_data))
        prior_ham = float(len(sms_ham) / len(sms_data))
        prob_ham  = (prior_ham * prob_ham)
        prob_spam = (prior_spam * prob_spam)

        if prob_ham > prob_spam:
            if (labels[count_1][1]) == "1":
                false_Count = false_Count + 1
        else:
            if (labels[count_1][1]) == "0":
                false_Count = false_Count + 1
        count_1 = count_1 + 1

    print(((len(sms_test_data) - false_Count) * 100) / len(labels))


if __name__ == '__main__':
    t1 = time.time()
    predict_Multinomial(sms_test_data)
    predict_Bernoulli(sms_test_data)
    t2 = time.time()
    print("Running time in seconds =  ", t2 - t1)
