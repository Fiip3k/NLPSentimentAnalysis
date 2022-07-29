"""
Teach models with the data from database and pickle the best one
"""
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score
import pickle

from databaseHandler import getPreprocessedData

# Finish preprocessing data for model input
df = getPreprocessedData()
X = np.array(df.review.values)
y = np.array(df.sentiment.values)
cv = CountVectorizer(max_features=1000)
X = cv.fit_transform(df.review).toarray()

# Split for training and testing
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.1)

# Create, fit and test 3 Naive Bayes models to pick from
gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()

gnb.fit(train_x, train_y)
mnb.fit(train_x, train_y)
bnb.fit(train_x, train_y)

gaussy = gnb.predict(test_x)
multinomialy = mnb.predict(test_x)
bernoulliy = bnb.predict(test_x)

gacc = accuracy_score(test_y, gaussy)
macc = accuracy_score(test_y, multinomialy)
bacc = accuracy_score(test_y, bernoulliy)

print(gacc)
print(macc)
print(bacc)

# Pick best model based on accuracy score
best = 0
if gacc >= macc and gacc >= bacc:
    best = gnb
elif macc >= gacc and macc >= bacc:
    best = mnb
else:
    best = bnb

# Save best model to file
pickle.dump(best, open('model1.pkl', 'wb'))

# Save vocabulary to file
pickle.dump(cv.vocabulary_, open('bow.pkl', 'wb'))
