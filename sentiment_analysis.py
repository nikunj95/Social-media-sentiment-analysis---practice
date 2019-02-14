# This file is created with the help of https://medium.com/@martinpella/customers-tweets-classification-41cdca4e2de
# It is a project intended to classify a tweet as neutral, positive or negative. Could achieve a 80% accuracy on the model
# using logistic regression approach.

# First pre-processing steps are taken to clean the text and any unwanted noise in the data. Then the data is split
# into test and training sets in order for the model to learn appropriately. Text vectorization using TF-IFD approach
# is performing and then using Logistic regression is used to train the model.

import numpy
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from utils import *
import snowballstemmer
import string
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# Read the csv file and check out its distribution
data = pd.read_csv('data/Tweets.csv')
# print(data.shape)
# print(data.sample(5))
# print(data.isnull().sum())
# print(data['airline_sentiment'].value_counts())

required_data = data[['text', 'airline_sentiment']].copy()
# print(plot_data.shape)

# Plot the data to see the distribution
# plt.figure
# required_data.groupby(['airline_sentiment']).count().plot(kind='bar', title='No. of tweets by class', legend=False)
# plt.show()
# print(required_data.sample(5))

# Pre-processing steps
# Encode labels
# Text cleaning
# Tokenizer
label_encoder = LabelEncoder()
required_data['target'] = label_encoder.fit_transform(required_data['airline_sentiment'])

text_cleaner = TextCleaner()
required_data['clean_text'] = text_cleaner.transform(required_data['text'])

# segment text into tokens(words)
tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
stemmer = snowballstemmer.EnglishStemmer() # words are reduced to their root words

def tokenize(s):
    tokens = tok.sub(r' \1 ', s).split()

    return stemmer.stemWords(tokens)

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(required_data['clean_text'].values, required_data['target'].values,
                                                    test_size=0.25, random_state=0)

# Text vectorization
# Transfrom text into vectors

# TF-IDF: Term frequency inverse document frequency
# used to give weights to words depending on number of times it occurs. Also offsets the word by number of time it occurs
# in the entire document
# It will not preserve the order of the words. Hence used for shallow learning - logistic regression

# max_df: terms that have document frequency higher than
# min_df: ignore terms that have document frequency lower than
# n_gram_range: groups of 1 and 2 consecutive words are extracted
vector = TfidfVectorizer(strip_accents='unicode', tokenizer=tokenize, ngram_range=(1, 2), max_df=0.75, min_df=3, sublinear_tf=True)
tfidf_train = vector.fit_transform(X_train)
tfidf_test = vector.transform(X_test)

# Logistic regression is used
# It also plots most important coefficients in order to identify the words that are being considered to make predictions
# for each target class
model = LogisticRegression(C=4, dual=True)
model.fit(tfidf_train, y_train)

pred_y = model.predict(tfidf_test)
print((pred_y==y_test).mean())

print(metrics.classification_report(y_test, pred_y, target_names=label_encoder.classes_))

"""
Output:
0.8087431693989071
              precision    recall  f1-score   support

    negative       0.83      0.94      0.88      2327
     neutral       0.71      0.54      0.62       772
    positive       0.81      0.63      0.71       561

   micro avg       0.81      0.81      0.81      3660
   macro avg       0.78      0.70      0.73      3660
weighted avg       0.80      0.81      0.80      3660

"""

