# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 12:34:36 2021

@author: Nikita
"""
import os
import re
import sys
from math import exp
from numpy import *
from math import log
from math import e
from collections import Counter

# Extract the list of stopwords
def getStopWords(path):
    stop_words = []
    f = open(path, 'r')
    for l in f.readlines():
        stop_words.append(l.strip())
    return stop_words

# Read in the data set and perform extraction of words file by file
# Skip the stopwords to form a list for each file
# Form the vocabulary, which is a set of distinct words
def readStopWords(path, stop_words):
    nums = 0
    vocabulary = [] 
    files = os.listdir(path)
    
    # Create an empty dictionary
    d = {}
    for file in files:
        f = open(path+"/"+file,encoding = "ISO-8859-1")
        nums += 1
        text = f.read()
        text = re.sub('[^a-zA-Z]', ' ', text)
        words_list = text.strip().split()
        words_new = []
        for w in words_list:
            if w not in stop_words:
                words_new.append(w)
        d[file] = words_new
        vocabulary.extend(words_new)
    return vocabulary, d

# Read in the data set and extract words in a list for each file
def readFile(path):
    files = os.listdir(path)
    nums = 0
    vocabulary = []
    d = {}
    for file in files:
        f = open(path+"/"+file,encoding = "ISO-8859-1")
        nums += 1
        text = f.read()
        text = re.sub('[^a-zA-Z]', ' ', text)
        words_list = text.strip().split()
        d[file] = words_list
        vocabulary.extend(words_list)
    return vocabulary, d


# Find distinct words
def extractKeys(train_spam_list,train_ham_list):
    return list(set(train_spam_list)|set(train_ham_list))

# Spam - 0, Ham - 1
# Insert all the words into a list
def getClassLabels(num_spam_file, num_ham_file):
    assigned_class = []
    for i in range(num_spam_file):
        assigned_class.append(0)
    for j in range(num_ham_file):
        assigned_class.append(1)
    return assigned_class

# Compare each feature(word) and check if it is the whole training list
# If the feature exits, then you mark it as 1 else mark it as 0
def featureList(words, dic1):
    w = list(words)
    result = []
    for f in dic1:
        row = [0] * (len(w))
        for word in w:
            if word in dic1[f]:
                row[w.index(word)] = 1
        # x0 = 1, so insert it first
        row.insert(0,1)
        result.append(row)
    return result

# Merge spam data and ham data 
def mergeData(dataset1, dataset2):
    dic3 = dataset1.copy()
    dic3.update(dataset2)
    return dic3

# Sigmoid function
def sigmoid(z):
    return 1.0/(1+exp(-z))

# Training Logistic Regression with L2 regularization
# Perform gradient ascent, go batch by batch
# Keep updating weights and return them
def trainLR(data_matrix,class_labels,lam):
    data_set = mat(data_matrix)
    label_set = mat(class_labels).transpose()
    m,n = shape(data_set)
    alpha = 0.1
    num_of_iterations = 100
    weights = zeros((n,1))
    for k in range(num_of_iterations):
        print("Iteration : ", k)
        h = sigmoid(data_set*weights)
        error = (label_set - h)
        weights = weights + alpha * data_set.transpose()* error - alpha*lam*weights
    return weights

# Classify the test data and compute the accuracy
def classify(weight,data,num_spam,num_ham):
    matrix = mat(data)
    wx = matrix * weight
    correct = 0
    total = num_spam + num_ham

    for i in range(num_spam):
        if wx[i][0] < 0.0:
            correct += 1
    for j in range(num_spam+1,total):
        if wx[j][0] > 0.0:
            correct += 1

    print(1.0 * correct/total)
    return wx

if __name__ == "__main__":
    train_spam_path = r'train/spam'
    train_ham_path = r'train/ham'
    test_spam_path = r'test/spam'
    test_ham_path = r'test/ham'
    stop_words_path = r'./stopwords.txt'
    lam = float(sys.argv[1])
    will_remove_stop_words = sys.argv[2]

    stopwords = getStopWords(stop_words_path)
    if will_remove_stop_words == 'yes':
        train_spam_list, train_spam_dict = readStopWords(train_spam_path,stopwords)
        train_ham_list, train_ham_dict = readStopWords(train_ham_path,stopwords)
    else:
        train_spam_list, train_spam_dict = readFile(train_spam_path)
        train_ham_list, train_ham_dict = readFile(train_ham_path)

    test_spam_list, test_spam_dict = readFile(test_spam_path)
    test_ham_list, test_ham_dict = readFile(test_ham_path)

    words = extractKeys(train_spam_list,train_ham_list)
    train = mergeData(train_spam_dict,train_ham_dict)
    test = mergeData(test_spam_dict,test_ham_dict)

    num_train_spam = len(train_spam_dict)
    num_test_spam = len(test_spam_dict)
    num_train_ham= len(train_ham_dict)
    num_test_ham = len(test_ham_dict)

    train_labels = getClassLabels(num_train_spam,num_train_ham)

    train_data_list = featureList(words, train)
    test_data_list = featureList(words, test)

    weight = trainLR(train_data_list, train_labels, lam)
    test = classify(weight, test_data_list, num_test_spam, num_test_ham)