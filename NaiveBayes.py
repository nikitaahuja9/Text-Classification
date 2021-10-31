# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 12:34:18 2021

@author: Nikita
"""

import os
import re
import sys
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

# Read in the data set and perform extraction of words for a single file
# Skip the stopwords to form a list of words for the file
def readStopWords(path, stop_words):
    files = os.listdir(path)
    vocabulary = []
    d = {}
    for file in files:
        f = open(path+"/"+file,encoding = "ISO-8859-1")
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
    vocabulary = []
    d = {}
    for file in files:
        f = open(path+"/"+file,encoding = "ISO-8859-1")
        text = f.read()
        text = re.sub('[^a-zA-Z]', ' ', text)
        words_list = text.strip().split()
        d[file] = words_list
        vocabulary.extend(words_list)
    return vocabulary, d

# Find distinct words
def extractKeys(train_spam_list,train_ham_list):
    return list(set(train_spam_list)|set(train_ham_list))

# Calculate the prior probability and conditional probability
# Store these in a dictionary
# Perform add one Laplace Smoothing (MAP)
def train_NB(count_spam, count_ham, spam_fre_dic, ham_fre_dic, spam_list, ham_list, key_list):
    prior_spam = 1.0 * count_spam / (count_spam + count_ham)
    prior_ham = 1.0 * count_ham / (count_spam + count_ham)
    total_spam = len(spam_list)
    total_ham = len(ham_list)
    total_type = len(key_list)
    add_one = 1
    cond_pro_spam = {}
    cond_pro_ham = {}

    for w in key_list:
        occurrence = 0
        if w in spam_fre_dic:
            occurrence = spam_fre_dic[w]
        cond_prob1 = 1.0 * (occurrence + add_one) / (total_spam + total_type)
        cond_pro_spam[w] = cond_prob1

    for w in key_list:
        occurrence = 0
        if w in ham_fre_dic:
            occurrence = ham_fre_dic[w]
        cond_prob2 = 1.0 * (occurrence + add_one) / (total_ham + total_type)
        cond_pro_ham[w] = cond_prob2

    return prior_spam, prior_ham, cond_pro_spam, cond_pro_ham

# Classify the test data and compute the accuracy
def apply_NB(prior_spam, prior_ham, cond_pro_spam, cond_pro_ham, spam_dic, ham_dic, key_list):
    set_dic = [spam_dic, ham_dic]
    total = len(spam_dic) + len(ham_dic)
    correct = 0
    for i in range(len(set_dic)):
        for f_name in set_dic[i]:
            score1 = log(prior_spam)
            score2 = log(prior_ham)
            for word in set_dic[i][f_name]:
                if word in key_list:
                    score1 += log(cond_pro_spam[word])
                    score2 += log(cond_pro_ham[word])
            if score1 >= score2 and i == 0:
                correct +=1
            elif score1 <= score2 and i == 1:
                correct +=1
    return 1.0 * correct/total

if __name__ == "__main__":
    train_spam_path = r'train/spam'
    train_ham_path = r'train/ham'
    test_spam_path = r'test/spam'
    test_ham_path = r'test/ham'
    stop_words_path = r'./stopwords.txt'
    stop_words = getStopWords(stop_words_path)
    will_remove_stop_words = sys.argv[1]
    
    if will_remove_stop_words == 'yes':
        train_spam_list, train_spam_dict = readStopWords(train_spam_path,stop_words)
        train_ham_list, train_ham_dict = readStopWords(train_ham_path,stop_words)
    else:
        train_spam_list, train_spam_dict = readFile(train_spam_path)
        train_ham_list, train_ham_dict = readFile(train_ham_path)
        
    test_spam_list, test_spam_dict = readFile(test_spam_path)
    test_ham_list, test_ham_dict = readFile(test_ham_path)

    num_spam = len(train_spam_dict)
    num_ham = len(train_ham_dict)

    spam_freq = Counter(train_spam_list)
    ham_freq = Counter(train_ham_list)
    key_list = extractKeys(spam_freq, ham_freq)

    prior_spam, prior_ham, cond_prob1, cond_prob2 = train_NB(num_spam, num_ham,spam_freq, ham_freq, train_spam_list, train_ham_list,key_list)
    funct = apply_NB(prior_spam, prior_ham, cond_prob1, cond_prob2, test_spam_dict, test_ham_dict, key_list)
    print (funct)