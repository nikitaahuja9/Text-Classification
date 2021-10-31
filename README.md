# Text-Classification

There are two folders -train and test, each of which have spam and ham folders. As their name suggests, the spam folder has the spam files, and the ham folder has the ham files. Two algorithms – Naïve Bayes and Logistic Regression have been used to perform text classification.

1. Naïve Bayes: We first form a vocabulary – set of distinct words and use two dictionaries for the spam and ham files. We read each file only once. We then find the MAP estimate (add-one Laplace smoothing) of the occurrence of a word in a file. This can help us understand if a test file is spam or ham.
To run the Naïve Bayes code, provide the following command line argument:
‘yes’ or ‘no’ to remove or retain stop words from the stop_words.txt file respectively.

2. Logistic Regression: We first form a vocabulary (similar to Naïve Bayes) – which is a set of distinct words and go file by file to calculate the occurrence of each word in a file. We create a matrix that can help us see how many times a word occurs in each file. We then find the weight (w) values and keep updating them batch by batch (Gradient Ascent). Here, L2 regularization is used, and I’ve used different values of lambda for the two cases that the stop words are included or not. The last step is to classify a test case as spam or ham.
Hard Limit of the number of iterations – 100 for each lambda value for this report only. It can be changed and increased to any desired value.
lambda = 0.001, 0.01, 0.1, 0.2, 0.3
To run the Logistic Regression code, provide two command line arguments:
1. lambda value as a float number followed by,
2. ‘yes’ or ‘no’ to remove or retain stop words from the stop_words.txt file respectively.

It is generally a good practice to remove the stop words because they are of no use in our classification procedure. However, it is not to be taken for granted that the removal of a stop word will increase the accuracy value. The accuracy value also depends on lambda which is used for regularization. The purpose of regularization is to penalize high weights and avoid overfitting. Report the accuracy values for the two cases of including/not including stop words.
