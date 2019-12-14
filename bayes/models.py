import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv('../data/names_dataset.csv')

df_names = df

# Replacing All F and M with 0 and 1 respectively
df_names.sex.replace({'F':0,'M':1},inplace=True)

Xfeatures = df_names['name']

# Feature Extraction 
cv = CountVectorizer()
X = cv.fit_transform(Xfeatures)

cv.get_feature_names()

# Train/test split
y = df_names.sex

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Naive Bayes Classifier
clf = MultinomialNB()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)

# Accuracy of the model
print("Accuracy Naive Bayes Model",clf.score(X_test,y_test)*100,"%")


# Prediction function
def genderpredictor(a):
    test_name = [a]
    vector = cv.transform(test_name).toarray()
    if clf.predict(vector) == 0:
        print("Female")
    else:
        print("Male")